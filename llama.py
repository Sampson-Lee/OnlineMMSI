import os
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from datasets import load_dataset
from qwen_vl_utils import process_vision_info

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
import wandb
from trl import SFTTrainer
from huggingface_hub import login
from trl import SFTConfig
import gc
from tqdm import tqdm
import re
import csv
import copy
import argparse

def test_image(
    dataset_name,
    model_name,
    image_folder,
    log_dir,
    max_new_tokens: int = 1024,
):
    split = 'test'

    def format_data(sample):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": sample["user"],
                    },
                ],
            },
        ]
    
    def extract_between_tags(text, start_tag, end_tag):
        try:
            start_idx = text.index(start_tag) + len(start_tag)
            end_idx = text.index(end_tag, start_idx)
            return text[start_idx:end_idx].strip()
        except ValueError:
            return None

    def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
        text_input = processor.apply_chat_template(
            format_data(sample), tokenize=False, add_generation_prompt=True  # Use the sample without the system message
        )

        model_inputs = processor(
            text=[text_input],
            images=[sample['image'].convert("RGB")],
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        output = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        generated_text = processor.decode(output[0])

        question_start_tag = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        answer_start_tag = "<|start_header_id|>assistant<|end_header_id|>"
        end_tag = "<|eot_id|>"
        answer_text = extract_between_tags(generated_text, answer_start_tag, end_tag)
        
        return answer_text

    test_dataset = load_dataset("imagefolder", data_dir=image_folder, split=split)
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto",)
    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
    model = PeftModel.from_pretrained(model, f'ckpts/{log_dir}')

    correct_count = 0
    os.makedirs(f'ckpts/{log_dir}', exist_ok=True)
    log_dir_ = log_dir.replace('/', '_')
    with open(f'ckpts/{log_dir}/{log_dir_}_output.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'pd_referent', 'gt_referent', 'output', 'gt', 'input'])

        processed_count = 0
        for index, test_item in tqdm(enumerate(test_dataset), total=len(test_dataset)):
            processed_count += 1

            output = generate_text_from_sample(
                model, processor, test_item, max_new_tokens=max_new_tokens
            )
            if output is None:
                output = ' '

            pd_referent = ' '
            gt_referent = test_item["assistant"][:7]
            
            pattern = r"Player\d+"
            gt_speaker = re.findall(pattern, test_item['user'])[-1]
            players = re.findall(pattern, output)
            if len(players) == 0: 
                players = [gt_speaker]

            if len(players) == 1:
                pd_referent = players[0]

            if len(players) >= 2:
                for i in range(1, len(players)):
                    if players[i - 1] == gt_speaker:
                        pd_referent = players[i]
                        break

            if 'forecast' in log_dir: pd_referent = players[0]

            writer.writerow([index, pd_referent, gt_referent, output, test_item['assistant'], test_item['user']])

            print(index, pd_referent, gt_referent, output, test_item['assistant'])
            if pd_referent == gt_referent: correct_count += 1

        accuracy = (correct_count / processed_count) * 100 if processed_count else 0.0
        message = f"dataset: {dataset_name} accuracy: {accuracy:.2f}%"
        print(message)
        writer.writerow([])
        writer.writerow(['Accuracy', accuracy])

def train_image(dataset_name, model_name, learning_rate, lr_strategy, image_folder, log_dir):
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16,)
    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)

    def format_data(sample):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": sample["user"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text", 
                        "text": sample["assistant"],
                    }
                ],
            },
        ]

    train_dataset = load_dataset("imagefolder", data_dir=image_folder, split='train')

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=512,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    def collate_fn(examples):
        
        texts = []; images = []
        for example in examples: 
            texts += [
                processor.apply_chat_template(format_data(example), tokenize=False)
            ]
            images += [example['image'].convert("RGB")]   # Process the images to extract inputs
        
        batch = processor(
            text=texts, 
            images=images, 
            return_tensors="pt", padding=True
        )
            
        def check_header(targets,seq):
            for i in range(len(seq)-3):
                if seq[i:i+3] in targets:
                    return True
            return False

        def replace_target(target,seq):
            for i in range(len(seq)-3):
                if seq[i:i+3] == target:
                    seq[i],seq[i+1],seq[i+2] = -100,-100,-100
            return seq

        label_list = []
        for i in range(len(batch["input_ids"])):  # The size of batch["input_ids"] is (batch_size, sequence_length)

            dialog_tokens = batch["input_ids"][i].tolist()
            labels = copy.copy(dialog_tokens)

            # Find end-of-turn (eot) tokens
            eot_indices = [i for i, n in enumerate(labels) if n == 128009]
            last_idx = 0

            # System and user prompt headers
            prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
            for n, idx in enumerate(eot_indices):
                current_seq = labels[last_idx:idx + 1]
                if check_header(prompt_header_seqs, current_seq):
                    # Found prompt header, indicating that this seq should be masked
                    labels[last_idx:idx + 1] = [-100] * (idx - last_idx + 1)
                else:
                    last_idx = idx + 1

            # Mask the assistant header prompt
            assistant_header_seq = [128006, 78191, 128007]  # Tokenized "<|start_assistant|>"
            labels = replace_target(assistant_header_seq, labels)

            # Mask the padding token and image token 128256
            for j in range(len(labels)):
                if labels[j] == processor.tokenizer.pad_token_id or labels[j] == 128256:  # 128256 is image token index
                    labels[j] = -100

            label_list.append(labels)

        batch["labels"] = torch.tensor(label_list)
        return batch  # Return the prepared batch

    # Configure training arguments
    per_device_train_batch_size=1
    gradient_accumulation_steps=4
    steps_per_epoch = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
    N_EPOCH = 5
    training_args = SFTConfig(
        output_dir='ckpts/'+log_dir,  # Directory to save the model
        num_train_epochs=N_EPOCH,  # Number of training epochs
        per_device_train_batch_size=per_device_train_batch_size,  # Batch size for training
        gradient_accumulation_steps=gradient_accumulation_steps,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=learning_rate,  # Learning rate for training
        lr_scheduler_type=lr_strategy,  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=100,  # Steps interval for logging
        save_strategy="steps",  # Strategy for saving the model
        save_steps=int(steps_per_epoch),  # Steps interval for saving
        greater_is_better=False,  # Whether higher metric values are better
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": True},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        weight_decay=0.01,
        # max_seq_length=1024  # Maximum sequence length for input
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset
    wandb.init(
        project=log_dir,  # change this
        name=log_dir,  # change this
        config=training_args,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":

    api_token = os.getenv('HF_API_TOKEN')
    login(token=api_token)

    parser = argparse.ArgumentParser(description='Train and test a model with a specified dataset.')
    parser.add_argument('--dataset_name', type=str, required=True, help='[STI_youtube_image_forecast_text_rect_point, ]')
    parser.add_argument('--learning_rate', type=float, required=True, help='[1e-3, 1e-4, 1e-5]')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Max new tokens for generation')
    parser.add_argument('--image_folder', type=str, default=None, help='Path to the dataset folder')
    args = parser.parse_args()

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    lr_strategy = 'linear'
    log_dir = args.dataset_name + '_' + model_name.split('/')[-1] + f'_{lr_strategy}_{args.learning_rate}'
    image_folder =  f'{args.image_folder}/{args.dataset_name}'
    
    train_image(args.dataset_name, model_name, args.learning_rate, lr_strategy, image_folder, log_dir)
    test_image(args.dataset_name, model_name, image_folder, log_dir, max_new_tokens=args.max_new_tokens)
