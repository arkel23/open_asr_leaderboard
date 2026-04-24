# Fleurs-SLU: Belebele-Fleurs

> [Fleurs-SLU (Schmidt et al., 2025)](https://arxiv.org/abs/2501.06117) introduces two SLU
> evaluation tasks: SIB-Fleurs (topical classification) and Belebele-Fleurs (listening
> comprehension). This README covers the Belebele-Fleurs evaluation only.

Belebele-Fleurs links each Belebele reading comprehension sample to its corresponding FLEURS
audio recordings. The underlying datasets are FLORES (text), FLEURS (speech), and Belebele (MCQ).
The authors apply preprocessing such as removing silent audios, discarding passages without
a complete set of audios, and other data quality filters. The final concatenation of sentence-level
audio chunks into a single passage audio is handled in this repo
([`normalizer/data_utils_v2.py#L229-L306`](https://github.com/arkel23/open_asr_leaderboard/blob/main/normalizer/data_utils_v2.py#L229-L306)).

When used as an evaluation task, the passage is presented as audio while the question and
4 answer options are provided as text.

---

## Setup

### Structure

```text
open_asr_leaderboard/
├── normalizer/
│   └── data_utils_v2.py
└── voxtral_slu/
    └── run_eval_ml.py
```

**Tested Python version:** Python 3.10

### Installation

Follow the Open ASR README for installation.

```
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_voxtral.txt
```
Additional required packages:
```
pip install accelerate
pip install torchcodec
pip install torchaudio
pip install wandb
```

### Usage

```bash
python run_eval_ml.py \
    --model_id mistralai/Voxtral-Mini-3B-2507 \
    --dataset_path WueNLP/belebele-fleurs \
    --dataset eng_Latn \
    --split test \
    --strategy best \
    --language eng_Latn \
    --batch_size 4 \
    --max_new_tokens 300 \
    --max_eval_samples 20 \
    --max_audio_seconds 30 \
    --warmup_steps 2 \
    --device 0 \
    --debug
```

Optional logging to Weights & Biases:

```bash
--wandb_project <project name> \
--wandb_entity <entity name>
```

---

## Dataset: Belebele-Fleurs

Evaluation dataset: [Hugging Face: WueNLP/belebele-fleurs](https://huggingface.co/datasets/WueNLP/belebele-fleurs)

### Sample Structure

**Top level (Belebele)** — each sample contains:
1. `flores_passage` → full reference passage
2. `question`
3. `mc_answer1`, `mc_answer2`, `mc_answer3`, `mc_answer4`
4. `question_number`
5. `correct_answer_num`
6. `sentence_data` → nested list of dictionaries, each representing a sentence chunk from the `flores_passage`

**Nested level (FLEURS)** — `sentence_data` 

Selected fields:
1. `id` → indicates the order of the sentence chunks
2. `sentence` → canonical sentence text for that chunk
3. `audio` → list of audio recordings for that sentence (1-2 speakers)
4. `whisper_asr_cer` → CER scores from Whisper ASR
5. `seamlessm4t_asr_cer` → CER scores from SeamlessM4T ASR

### Full Example
``````
WueNLP/belebele-fleurs | cfg=eng_Latn | split=test
{'correct_answer_num': '4',
 'dialect': 'eng_Latn',
 'ds': datetime.datetime(2023, 5, 3, 0, 0),
 'flores_passage': 'Ancient China had a unique way of showing different time periods; each stage of China or each '
                   'family that was in power was a distinctive dynasty. Also between each dynasty was an unstable age '
                   'of divided provinces. The best-known of these periods was the Three Kingdoms epoch taking place '
                   'for 60 years between the Han and the Jin Dynasty. During these periods fierce warfare took place '
                   'between many nobles fighting for the throne. The Three Kingdoms was one of the bloodiest eras in '
                   'Ancient China’s history thousands of people died fighting to sit in the highest seat in the grand '
                   'palace at Xi’an.',
 'link': 'https://en.wikibooks.org/wiki/Ancient_China/Government',
 'mc_answer1': 'The Jin Dynasty',
 'mc_answer2': 'The Xi’an era',
 'mc_answer3': 'The Han Dynasty',
 'mc_answer4': 'The Three Kingdoms era',
 'question': 'According to the passage, which of the following was one of China’s most violent eras?',
 'question_number': 1,
 'sentence_data': [{'URL': 'https://en.wikibooks.org/wiki/Ancient_China/Government',
                    'audio': [<datasets.features._torchcodec.AudioDecoder object at 0x7f333282f550>,
                              <datasets.features._torchcodec.AudioDecoder object at 0x7f333282d450>],
                    'domain': 'wikibooks',
                    'filename': ['7963255664903240942.wav', '1447510838326873839.wav'],
                    'fleurs_id': 1782,
                    'full_paragraph': True,
                    'gender': ['FEMALE', 'FEMALE'],
                    'has_hyperlink': 0,
                    'has_image': 0,
                    'id': 452,
                    'num_samples': [188160, 189120],
                    'raw_transcription': 'Also between each dynasty was an unstable age of divided provinces. The '
                                         'best-known of these periods was the Three Kingdoms epoch taking place for 60 '
                                         'years between the Han and the Jin Dynasty.',
                    'seamlessm4t_asr': ['also between each dynasty was an unstable age of divided provinces the best '
                                        'known of these periods was the three kingdoms epoch taking place for sixty '
                                        'years between the han and the jin dynasty',
                                        'also between each dynasty was an unstable age of divided provinces the best '
                                        'known of these periods was the three kingdoms epoch taking place for sixty '
                                        'years between the han and the jin dynasty'],
                    'seamlessm4t_asr_cer': [0.07853403141361257, 0.07853403141361257],
                    'seamlessm4t_asr_translation': ['also between each dynasty was an unstable age of divided '
                                                    'provinces the best known of these periods was the three kingdoms '
                                                    'epoch taking place for sixty years between the han and the jin '
                                                    'dynasty',
                                                    'also between each dynasty was an unstable age of divided '
                                                    'provinces the best known of these periods was the three kingdoms '
                                                    'epoch taking place for sixty years between the han and the jin '
                                                    'dynasty'],
                    'seamlessm4t_asr_wer': [0.3333333333333333, 0.3333333333333333],
                    'sentence': 'Also between each dynasty was an unstable age of divided provinces. The best-known of '
                                'these periods was the Three Kingdoms epoch taking place for 60 years between the Han '
                                'and the Jin Dynasty.',
                    'sentence_idx': 0,
                    'speaker_id': [2, 2],
                    'split': ['test', 'test'],
                    'topic': 'ancient china/government',
                    'transcription': 'also between each dynasty was an unstable age of divided provinces the '
                                     'best-known of these periods was the three kingdoms epoch taking place for 60 '
                                     'years between the han and the jin dynasty',
                    'whisper_asr': ['Also between each dynasty was an unstable age of dividend provinces. The best '
                                    'known of these periods was the Three Kingdoms Epoch, taking place for 60 years '
                                    'between the Han and the Jin Dynasty.',
                                    'Also, between each dynasty was an unstable age of divided provinces. The best '
                                    'known of these periods was the Three Kingdoms Epoch, taking place for 60 years '
                                    'between the Han and the Jin Dynasty.'],
                    'whisper_asr_cer': [0.020942408376963352, 0.020942408376963352],
                    'whisper_asr_translation': ['Also between each dynasty was an unstable age of dividend provinces. '
                                                'The best known of these periods was the Three Kingdoms Epoch, taking '
                                                'place for 60 years between the Han and the Jin Dynasty.',
                                                'Also, between each dynasty was an unstable age of divided provinces. '
                                                'The best known of these periods was the Three Kingdoms Epoch taking '
                                                'place for 60 years between the Han and the Jin Dynasty.'],
                    'whisper_asr_wer': [0.12121212121212122, 0.12121212121212122]},
                   {'URL': 'https://en.wikibooks.org/wiki/Ancient_China/Government',
                    'audio': [<datasets.features._torchcodec.AudioDecoder object at 0x7f333282f790>,
                              <datasets.features._torchcodec.AudioDecoder object at 0x7f333282f1f0>],
                    'domain': 'wikibooks',
                    'filename': ['5407053594615190090.wav', '7366992441583559574.wav'],
                    'fleurs_id': 1819,
                    'full_paragraph': True,
                    'gender': ['MALE', 'MALE'],
                    'has_hyperlink': 0,
                    'has_image': 0,
                    'id': 451,
                    'num_samples': [207040, 195840],
                    'raw_transcription': 'Ancient China had a unique way of showing different time periods; each stage '
                                         'of China or each family that was in power was a distinctive dynasty.',
                    'seamlessm4t_asr': ['Ancient China had a unique way of showing different time periods. Each stage '
                                        'of China or each family that was in power was a distinctive dynasty.',
                                        'Ancient China had a unique way of showing different periods. Each stage of '
                                        'China, or each family that was in power, was a distinctive dynasty.'],
                    'seamlessm4t_asr_cer': [0.013793103448275862, 0.06206896551724138],
                    'seamlessm4t_asr_translation': ['Ancient China had a unique way of showing different time periods. '
                                                    'Each stage of China or each family that was in power was a '
                                                    'distinctive dynasty.',
                                                    'Ancient China had a unique way of showing different periods. Each '
                                                    'stage of China, or each family that was in power, was a '
                                                    'distinctive dynasty.'],
                    'seamlessm4t_asr_wer': [0.07692307692307693, 0.19230769230769232],
                    'sentence': 'Ancient China had a unique way of showing different time periods; each stage of China '
                                'or each family that was in power was a distinctive dynasty.',
                    'sentence_idx': 1,
                    'speaker_id': [1, 1],
                    'split': ['test', 'test'],
                    'topic': 'ancient china/government',
                    'transcription': 'ancient china had a unique way of showing different time periods each stage of '
                                     'china or each family that was in power was a distinctive dynasty',
                    'whisper_asr': ['Ancient China had a unique way of showing different time periods. Each stage of '
                                    'China or each family that was in power was a distinctive dynasty.',
                                    'Ancient China had a unique way of showing different periods. Each stage of China '
                                    'or each family that was in power was a distinctive dynasty.'],
                    'whisper_asr_cer': [0.013793103448275862, 0.04827586206896552],
                    'whisper_asr_translation': ['Ancient China had a unique way of showing different time periods. '
                                                'Each stage of China or each family that was in power was a '
                                                'distinctive dynasty.',
                                                'Ancient China had a unique way of showing different periods. Each '
                                                'stage of China or each family that was in power was a distinctive '
                                                'dynasty.'],
                    'whisper_asr_wer': [0.07692307692307693, 0.11538461538461539]},
                   {'URL': 'https://en.wikibooks.org/wiki/Ancient_China/Government',
                    'audio': [<datasets.features._torchcodec.AudioDecoder object at 0x7f333282f4c0>],
                    'domain': 'wikibooks',
                    'filename': ['6761397427575851174.wav'],
                    'fleurs_id': 1385,
                    'full_paragraph': True,
                    'gender': ['FEMALE'],
                    'has_hyperlink': 0,
                    'has_image': 0,
                    'id': 453,
                    'num_samples': [93760],
                    'raw_transcription': 'During these periods fierce warfare took place between many nobles fighting '
                                         'for the throne.',
                    'seamlessm4t_asr': ['During these periods, fierce warfare took place between many nobles fighting '
                                        'for the throne.'],
                    'seamlessm4t_asr_cer': [0.01098901098901099],
                    'seamlessm4t_asr_translation': ['During these periods, fierce warfare took place between many '
                                                    'nobles fighting for the throne.'],
                    'seamlessm4t_asr_wer': [0.07142857142857142],
                    'sentence': 'During these periods fierce warfare took place between many nobles fighting for the '
                                'throne.',
                    'sentence_idx': 2,
                    'speaker_id': [18],
                    'split': ['train'],
                    'topic': 'ancient china/government',
                    'transcription': 'during these periods fierce warfare took place between many nobles fighting for '
                                     'the throne',
                    'whisper_asr': ['During these periods fierce warfare took place between many nobles fighting for '
                                    'the throne.'],
                    'whisper_asr_cer': [0.0],
                    'whisper_asr_translation': ['During these periods fierce warfare took place between many nobles '
                                                'fighting for the throne.'],
                    'whisper_asr_wer': [0.0]},
                   {'URL': 'https://en.wikibooks.org/wiki/Ancient_China/Government',
                    'audio': [<datasets.features._torchcodec.AudioDecoder object at 0x7f333282f610>,
                              <datasets.features._torchcodec.AudioDecoder object at 0x7f333282f940>],
                    'domain': 'wikibooks',
                    'filename': ['17656538893065595043.wav', '2479496904141533095.wav'],
                    'fleurs_id': 1948,
                    'full_paragraph': True,
                    'gender': ['FEMALE', 'FEMALE'],
                    'has_hyperlink': 0,
                    'has_image': 0,
                    'id': 454,
                    'num_samples': [190080, 163200],
                    'raw_transcription': 'The Three Kingdoms was one of the bloodiest eras in Ancient China’s history '
                                         'thousands of people died fighting to sit in the highest seat in the grand '
                                         'palace at Xi’an.',
                    'seamlessm4t_asr': ["The Three Kingdoms was one of the bloodiest eras in ancient China's history. "
                                        'Thousands of people died fighting to sit on the highest seat in the Grand '
                                        'Palace at Zion.',
                                        "the three kingdoms was one of the bloodiest eras in ancient china's history "
                                        'thousands of people died fighting to sit in the highest seat in the grand '
                                        'palace at zion'],
                    'seamlessm4t_asr_cer': [0.060240963855421686, 0.060240963855421686],
                    'seamlessm4t_asr_translation': ['The Three Kingdoms was one of the bloodiest eras in ancient '
                                                    "China's history. Thousands of people died fighting to sit on the "
                                                    'highest seat in the Grand Palace at Zion.',
                                                    'the three kingdoms was one of the bloodiest eras in ancient '
                                                    "china's history thousands of people died fighting to sit in the "
                                                    'highest seat in the grand palace at zion'],
                    'seamlessm4t_asr_wer': [0.26666666666666666, 0.2],
                    'sentence': 'The Three Kingdoms was one of the bloodiest eras in Ancient China’s history thousands '
                                'of people died fighting to sit in the highest seat in the grand palace at Xi’an.',
                    'sentence_idx': 3,
                    'speaker_id': [5, 2],
                    'split': ['test', 'test'],
                    'topic': 'ancient china/government',
                    'transcription': "the three kingdoms was one of the bloodiest eras in ancient china's history "
                                     'thousands of people died fighting to sit in the highest seat in the grand palace '
                                     "at xi'an",
                    'whisper_asr': ["The Three Kingdoms was one of the bloodiest eras in ancient China's history. "
                                    'Thousands of people died fighting to sit in the highest seat in the Grand Palace '
                                    'at Zion.',
                                    "The Three Kingdoms was one of the bloodiest eras in ancient China's history. "
                                    'Thousands of people died fighting to sit in the highest seat in the Grand Palace '
                                    'at Zion.'],
                    'whisper_asr_cer': [0.05421686746987952, 0.05421686746987952],
                    'whisper_asr_translation': ["The Three Kingdoms was one of the bloodiest eras in ancient China's "
                                                'history. Thousands of people died fighting to sit in the highest seat '
                                                'in the Grand Palace at Zion.',
                                                "The Three Kingdoms was one of the bloodiest eras in ancient China's "
                                                'history. Thousands of people died fighting to sit in the highest seat '
                                                'in the Grand Palace at...Zion?'],
                    'whisper_asr_wer': [0.23333333333333334, 0.23333333333333334]}]}

```



