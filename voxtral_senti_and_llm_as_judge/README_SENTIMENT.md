# Fleurs-SLU: TwinkStart/MELD

> evaluation tasks: Sentiment classification and emotion evaluation
> This README covers only sentiment evaluation

TwinkStart/MELD given audio, task is to classify the sentence into "positive" , "negative" and "neutral"


---

## Setup

### Structure

```text
open_asr_leaderboard/
├── normalizer/
│   └── data_utils_v2.py
│   └── eval_utils.py 
└── voxtral_senti_and_llm/
    └── run_eval_sentiment.py
    └── run_voxtral_sentiment.sh
    └── results # all outjson files will be stored here   
```

**Tested Python version:** Python 3.12

### Installation

Follow the Open ASR README for installation.

```
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_voxtral.txt
```

### Usage

```bash run_eval_sentiment.sh
```

Optional logging to Weights & Biases:

```
--wandb_project <project name> \
--wandb_entity <entity name>
```

---

## Dataset: TwinkStart/MELD

Evaluation dataset: [Hugging Face: TwinkStart/MELD](https://huggingface.co/datasets/TwinkStart/MELD)

### Sample Structure

**relevant columns** — each sample contains:
1. `sr no`
2. `Utterance` → user text
3. `Speaker` 
4. `Emotion`
5. `Sentiment`
6. `video path` 
7. `WavPath`
8. `audio`

**output → /results forlder jsonl format**
**metrics → accuracy**

### Output sample structure

references → ground truth /given labels
pred_text  → predicted by voxtral

```
{"user_text": "Why do all youre coffee mugs have numbers on the bottom?", "references": "positive", "pred_text": "neutral"},

{"user_text": "Oh. Thats so Monica can keep track. That way if one on them is missing, she can be like, Wheres number 27?!", "references": "negative", "pred_text": "neutral"}
```