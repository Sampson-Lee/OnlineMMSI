import os
import torch
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from datasets import load_dataset

from qwen_vl_utils import process_vision_info

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig
import wandb
from trl import SFTTrainer
from huggingface_hub import login

import time
from tqdm import tqdm
import re
import csv
import argparse
from IPython import embed

def test_video(dataset_name, model_name, video_folder, log_dir, fps=1.0, lora_path: str = None):
    split = 'test'
    def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
        text_input = processor.apply_chat_template(
            sample, tokenize=False, add_generation_prompt=True  # Use the sample without the system message
        )
        image_inputs, video_inputs = process_vision_info(sample)

        model_inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs, 
            return_tensors="pt",
        ).to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]  # Return the first decoded output text

    def format_data(sample):
        
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f'{video_folder}/{split}/{sample["video_name"]}',
                        "max_pixels": 36 * 42 * 10,
                        "fps": fps,
                    },
                    {
                        "type": "text",
                        "text": sample["user"],
                        # "text": "Describe the social interaction cues in the video.",
                    },
                ],
            },

        ]
    
    test_dataset = load_dataset("videofolder", data_dir=video_folder, split=split)
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto",)
    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
    
    model = PeftModel.from_pretrained(model, f'ckpts/{log_dir}')

    correct_count = 0
    os.makedirs(f'ckpts/{log_dir}', exist_ok=True); log_dir_ = log_dir.replace('/', '_')
    with open(f'ckpts/{log_dir}/{log_dir_}_output.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['video_name', 'pd_referent', 'gt_referent', 'output', 'gt', 'input'])
        for index, test_item in tqdm(enumerate(test_dataset)):
            embed()
            output = generate_text_from_sample(model, processor, format_data(test_item), max_new_tokens=1024)
            
            pd_referent = ' '
            gt_referent = test_item["assistant"][:7]
            
            pattern = r"Player\d+"
            gt_speaker = re.findall(pattern, test_item['user'])[-1]
            players = re.findall(pattern, output)
            if len(players) == 0: 
                players = [gt_referent]

            if len(players) == 1:
                pd_referent = players[0]

            if len(players) >= 2:
                for i in range(1, len(players)):
                    if players[i - 1] == gt_speaker:
                        pd_referent = players[i]
                        break

            if 'forecast' in log_dir: pd_referent = players[0]

            writer.writerow([test_item['video_name'], pd_referent, gt_referent, output, test_item['assistant'], test_item['user']])

            print(test_item['video_name'], pd_referent, gt_referent, output, test_item['assistant'])
            if pd_referent == gt_referent: correct_count += 1

        accuracy = (correct_count / (index + 1)) * 100
        message = f"dataset: {dataset_name} accuracy: {accuracy:.2f}%"
        print(message)
        writer.writerow([])
        writer.writerow(['Accuracy', accuracy])

def train_video(dataset_name, model_name, learning_rate, lr_strategy, video_folder, log_dir, fps=1.0):
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",)
    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)
    split = 'train'
    def format_data(sample):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f'{video_folder}/{split}/{sample["video_name"]}',
                        "max_pixels": 36 * 42 * 10,
                        "fps": fps,
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
    
    train_dataset = load_dataset("videofolder", data_dir=video_folder, split=split)
    train_dataset = [format_data(sample) for sample in train_dataset]
    
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
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]

        video_inputs = []
        for example in examples:
            video_inputs += process_vision_info(example)[1]   # Process the images to extract inputs
        
        batch = processor(
            text=texts, 
            videos=video_inputs, 
            return_tensors="pt", padding=True
        )

        labels = batch["input_ids"].clone() 
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the visual token index in the loss computation (model specific)
        for visual_token in ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",]: #, "<|im_start|>", "<|im_end|>"]:
            visual_token_id = processor.tokenizer.convert_tokens_to_ids(visual_token)
            labels[labels == visual_token_id] = -100

        batch["labels"] = labels
        
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
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., mmsi_forecast)')
    parser.add_argument('--learning_rate', type=float, required=True, help='[1e-3, 1e-4, 1e-5]')
    parser.add_argument('--fps', type=float, required=True, help='[1, 2, 4]')
    parser.add_argument('--video_folder', type=str, default=None, help='Path to the dataset folder')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to LoRA weights')
    args = parser.parse_args()

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    lr_strategy = 'linear'
    log_dir = args.dataset_name + '_' + model_name.split('/')[-1] + f'_{lr_strategy}_{args.learning_rate}'
    video_folder = f'{args.video_folder}/{args.dataset_name}'
    
    train_video(args.dataset_name, model_name, args.learning_rate, lr_strategy, video_folder, log_dir, args.fps)
    test_video(args.dataset_name, model_name, video_folder, log_dir, args.fps, args.lora_path)
