# Towards Online Multi-Modal Social Interaction Understanding [TMLR 2026]

## Online-MMSI-VLM



## Installation
Clone the repo
```
git clone git@github.com:Sampson-Lee/OnlineMMSI.git
cd OnlineMMSI
```

Create an enviroment:
```
conda create -n online_mmsi python=3.11 -y
conda activate online_mmsi
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```
Or
```
conda env create -f environment.yml
```

## Dataset
You can access the dataset from [Box](https://utdallas.box.com/s/yd097xmtw236w8ta6jh8da128z6xxfh9) with password: OnlineMMSI2026.

Each folder follows this structure:
```text
<folder_name>/
├── train/
│   ├── xxx.mp4               # historical video clip
│   ├── metadata.csv          # sample annotations
│   └── mllm_video_train.json # instruction-formatted training data for LLaMA-Factory
└── test/
    ├── xxx.mp4               # historical video clip
    └── metadata.csv          # sample annotations
```


#### Loading with HuggingFace
```
from datasets import load_dataset

dataset = load_dataset(
    "videofolder",
    data_dir="online_mmsi_STI_youtube_video_forecast_text_rect_point",
    split="train"
)
```
Change data_dir to any of the six folders as needed.


#### Training with LLaMA-Factory
Use the provided file: train/mllm_video_train.json in configuration:
```
dataset: mllm_video_train.json
```
This file contains multimodal instruction-formatted training samples.

## Train & Test

#### Qwen2.5-VL-7B
```
TASK = STI # Select one from ["STI", "PCR", "MPP"]
DATASET = youtube # Select one from ["youtube", ego4d]
CUDA_VISIBLE_DEVICES=0 python qwen.py --dataset_name ${TASK}_${DATASET}_video_text_rect_point --video_folder YOUR_PATH --learning_rate 1e-4
```

#### LLaMA-3.2-V
```
TASK = STI # Select one from ["STI", "PCR", "MPP"]
DATASET = youtube # Select one from ["youtube", ego4d]
CUDA_VISIBLE_DEVICES=0 python llama.py --dataset_name ${TASK}_${DATASET}_image_text_rect_point --video_folder YOUR_PATH --learning_rate 1e-4
```


## Citation
If the project is helpful for you, consider citing it.
```
@article{li2025towards,
  title={Towards online multi-modal social interaction understanding},
  author={Li, Xinpeng and Deng, Shijian and Lai, Bolin and Pian, Weiguo and Rehg, James M and Tian, Yapeng},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2026},
}
```
