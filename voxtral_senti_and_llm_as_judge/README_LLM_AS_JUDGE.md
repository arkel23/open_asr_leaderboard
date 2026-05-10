# TwinkStart/speech-triavia-qa
> evaluation tasks: Sentiment classification and emotion evaluation
> This README covers only sentiment evaluation

TwinkStart/speech-triavia-qa given audio, task is to first generate the answer for given question by voxtral and then score into "1" → totally incorrect , "2" → paritally correct and "3" → correct


---

## Setup

### Structure

```text
open_asr_leaderboard/
├── normalizer/
│   └── data_utils_v2.py
│   └── eval_utils.py
│   └── llm_as_judge_eval_utils.py
└── voxtral_senti_and_llm/
│   └── run_eval_llm.py
│   └── run_voxtral_llm.sh
│.  └── results #all outjson files will be stored here   
└── .env  #create .env file locally once pulled the repo
```

**Tested Python version:** Python 3.12

### Installation

Follow the Open ASR README for installation.

```
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_voxtral.txt
pip install -r requirements/llm_as_judge.txt #sarvam as is optional
.env # create .env file and put llm api secret key in this 
    COHERE_API_KEY=<api secret key> 
```

### Usage

```bash run_eval_llm.sh
```

Optional logging to Weights & Biases:

```
wandb_entity=<project name>, wandb_project=<entity name>

```

---

## Dataset: TwinkStart/speech-triavia-qa

Evaluation dataset: [Hugging Face: TwinkStart/speech-triavia-qa](https://huggingface.co/datasets/TwinkStart/speech-triavia-qa)

### Sample Structure

**relevant columns** — each sample contains:
1. `question` → user text
2. `answer` → groundtruth
6. `save_name` or
7. `WavPath`or
8. `audio` 

**output → /results forlder jsonl format**
**metrics → accuracy**

**output → /results forlder jsonl format**
/result folder will generate two files containing 1st_file. voxtral predicted responses,  2nd_file. llm scores predicted by llm api on voxtral responses

**metrics → accuracy**

### Output sample structure for 1st_file

question → user text
prediction  → predicted by voxtral

```
{"question": "Who was the man behind The Chipmunks?", "prediction": "the man behind the chipmunks is ross bagdasarian senior"}

{"question": "What star sign is Jamie Lee Curtis?", "prediction": "jamie lee curtis was born on november 22 1958 her star sign is scorpio"}
```

### Output sample structure for 2nd_file

question → user text
prediction  → predicted by voxtral
score → score given by llm on prediction

```
{"question": "What star sign is Jamie Lee Curtis?", "prediction": "jamie lee curtis was born on november 22 1958 her star sign is scorpio", "score": "3"}

{"question": "Which Lloyd Webber musical premiered in the US on 10th December 1993?", "prediction": "the musical that premiered in the us on the 10th of december 1993 is rent", "score": "1"}
```






