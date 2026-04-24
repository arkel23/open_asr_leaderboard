# Fleurs-SLU: Belebele-Fleurs


## Setup

### Structure

```text
open_asr_leaderboard/
├── normalizer/
│   └── data_utils_v2.py
└── qwen2_slu/
    └── run_eval_ml.py
```

**Tested Python version:** Python 3.10

### Installation

Follow the Open ASR README for installation.

```
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_qwen.txt
```
Additional required packages:
```
pip install accelerate bitsandbytes torchcodec torchaudio wandb
```



### Usage

```
python run_eval_ml.py \
  --model_id Qwen/Qwen2-Audio-7B \
  --dataset_path WueNLP/belebele-fleurs \
  --dataset eng_Latn \
  --language eng_Latn \
  --strategy best \
  --split test \
  --batch_size 6 \
  --max_new_tokens 10 \
  --max_eval_samples 20 \
  --max_audio_seconds 30 \
  --warmup_steps 2 \
  --quant_config bnb \
  --quant_dtype_weights 8 \
  --device 0 \
  --debug

```

**CLI  W&B Logging**
```
python run_eval_ml.py \
  --model_id Qwen/Qwen2-Audio-7B \
  --dataset_path WueNLP/belebele-fleurs \
  --dataset eng_Latn \
  --language eng_Latn \
  --strategy best \
  --split test \
  --batch_size 6 \
  --max_new_tokens 10 \
  --max_eval_samples 20 \
  --max_audio_seconds 30 \
  --warmup_steps 2 \
  --quant_config bnb \
  --quant_dtype_weights 8 \
  --device 0 \
  --debug
  --wandb_entity <entity name> \
  --wandb_project <project name>

```