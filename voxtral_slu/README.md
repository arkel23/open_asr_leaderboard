## Belebele FLEURS SLU Evaluation (Voxtral)
### Structure:

```text
open_asr_leaderboard/
├── normalizer/
│   └── data_utils_v2.py
│   └── evals_utils_v2.py
└── voxtral_slu/
    └── run_eval_ml_slu.py
```

**Tested Python version:** Python 3.10

### Installation:
Follow the Open ASR README for installation.
```
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_voxtral.txt
```
Additional required packages
```
pip install accelerate
pip install torchcodec
pip install torchaudio
pip install wandb
```


#### Test English:
```bash
python ./voxtral_slu/run_eval_ml_slu.py \
--model_id mistralai/Voxtral-Mini-3B-2507 \
--dataset WueNLP/belebele-fleurs \
--config_name eng_Latn \
--split test \
--device 0 \
--batch_size 4 \
--max_eval_samples 20 \
--max_audio_seconds 30 \
--max_new_tokens 20 \
--warmup_steps 5 \
--strategy best \
--debug 
````

#### Test Hindi:
```bash
python ./voxtral_slu/run_eval_ml_slu.py \
--model_id mistralai/Voxtral-Mini-3B-2507 \
--dataset WueNLP/belebele-fleurs \
--config_name hin_Deva \
--split test \
--device 0 \
--batch_size 4 \
--max_eval_samples 20 \
--max_audio_seconds 30 \
--max_new_tokens 20 \
--warmup_steps 5 \
--strategy best \
--debug 
```
Optional logging to Weights & Biases:
```
--wandb_project <project name> \
--wandb_entity <entity name>
```

---

## SLU Evaluation Dataset: Belebele-Fleurs

Voxtral takes the audio as input and gets asked multiple choice questions about the content of the passage.
Evaluation Dataset: [Hugging Face: WueNLP/belebele-fleurs](https://huggingface.co/datasets/WueNLP/belebele-fleurs)

---

### Dataset Breakdown

**Belebele** is a multilingual reading comprehension benchmark.

**FLEURS** is a multilingual speech dataset with audio recordings of text passages, often with multiple speakers per sentence.

**Belebele-FLEURS** combines both, where the Belebele passages overlap with FLEURS text. The dataset links each Belebele reading comprehension sample to its corresponding FLEURS audio recordings.

---

### Top Level: Belebele

Each sample contains:
1. `flores_passage` → full reference passage
2. `question`
3. `mc_answer1`, `mc_answer2`, `mc_answer3`, `mc_answer4`
4. `question_number`
5. `correct_answer_num`
6. `sentence_data` → nested list of dictionaries, each dictionary represents a sentence chunk from the `flores_passage`

---

### Nested Level: FLEURS

`sentence_data` is a list of dictionaries, where each element represents a sentence chunk of the full passage.

Selected fields:
1. `id` → also indicates the order of the chunks
2. `audio` → there can be multiple audio recordings per sentence chunk
3. `sentence` → canonical sentence text for that chunk
4. `seamlessm4t_asr_cer`
5. `seamlessm4t_asr_wer`
6. `whisper_asr_cer`
7. `whisper_asr_wer`

---

## Particularities

### 1. Unordered Sentence Chunks
The `id` field is the most reliable way to order the chunks so they match the `flores_passage`.

Example:

`flores_passage`:

> Make sure your hand is as relaxed as possible while still hitting all the notes correctly - also try not to make much extraneous motion with your fingers. This way, you will tire yourself out as little as possible. Remember there's no need to hit the keys with a lot of force for extra volume like on the piano. On the accordion, to get extra volume, you use the bellows with more pressure or speed.

However, for this sample it is nested as follows:

- `fleurs_id=479`, `sentence_idx=0`
  → “Make sure your hand is as relaxed as possible…”

- `fleurs_id=481`, `sentence_idx=1`
  → “On the accordion, to get extra volume…”

- `fleurs_id=480`, `sentence_idx=2`
  → “This way, you will tire yourself out…”

If we sort by `sentence_idx`, we get

> Make sure your hand is as relaxed as possible... On the accordion, to get extra volume... This way, you will tire yourself out...

which is incorrect, as the sentence order does not match the original passage.

If we instead sort by `fleurs_id`, we get:

> Make sure your hand is as relaxed as possible... This way, you will tire yourself out... On the accordion, to get extra volume...

which matches the actual `flores_passage`. However, this alignment is not consistent, and other examples show that `fleurs_id` and `sentence_idx` are not reliable ordering signals.
Towards the end of this README, you'll find an example from `config_name='eng_Latn'`, `split='test'`, at index 11 that demonstrates this.

### 2. Which Audio to Pick?

*(N number of audios per sentence chunk)*

Each sentence chunk may contain multiple audio recordings, so a selection strategy is needed to choose a single audio per chunk.
The dataset provides ASR quality signals such as WER and CER. Since audio quality strongly impacts SLU performance
(FLEURS-SLU: [https://arxiv.org/pdf/2501.06117](https://arxiv.org/pdf/2501.06117)), CER is available in the codebase to guide selection
by averaging, then reranking based on the selected `strategy`.

The following strategies are available via the flag `--strategy`:
- **best** → lowest average CER across available ASR systems
- **worst** → highest average CER
- **random** → randomly selects an audio

---

## Debug Mode

`--debug` was added to inspect how the passage audio is being reconstructed during streaming evaluation.

It prints:
- The full `flores_passage`
- The selected sentences in concatenation order, this reflects the order the audios were concatenated.
- Chunk metadata
- The prompt sent to the model
- Final concatenated audio duration (in seconds)

Notes:
- Text fields such as `sentence`, and `flores_passage` are only there for debugging and sanity checks.
  They are not part of the actual speech input during SLU evaluation.
- In `--no_audio` mode, audio is still used for dataset filtering to ensure both audio and text-only runs evaluate
  on the same subset of samples, these traces still appear in `--debug` mode.

**Debug Mode Example:**

```
======================================================================
[DEBUG] Sample 2/4  |  question_num: 1
[DEBUG] RUN TAG: with_audio_all_lengths_best
----------------------------------------------------------------------
[DEBUG] FLORES PASSAGE (full reference passage):
Ancient China had a unique way of showing different time periods; each stage of China
or each family that was in power was a distinctive dynasty. Also between each dynasty 
was an unstable age of divided provinces. The best-known of these periods was the Three Kingdoms
epoch taking place for 60 years between the Han and the Jin Dynasty. During these periods fierce
warfare took place between many nobles fighting for the throne. The Three Kingdoms was one of the
bloodiest eras in Ancient China’s history thousands of people died fighting to sit in the highest 
seat in the grand palace at Xi’an.
----------------------------------------------------------------------
[DEBUG] SELECTED SENTENCES (concatenation order):
  [0] Ancient China had a unique way of showing different time periods; each stage of China or each family that was in power was a distinctive dynasty.
  [1] Also between each dynasty was an unstable age of divided provinces. The best-known of these periods was the Three Kingdoms epoch taking place for 60 years between the Han and the Jin Dynasty.
  [2] During these periods fierce warfare took place between many nobles fighting for the throne.
  [3] The Three Kingdoms was one of the bloodiest eras in Ancient China’s history thousands of people died fighting to sit in the highest seat in the grand palace at Xi’an.
----------------------------------------------------------------------
[DEBUG] MCQ PROMPT:
Listen to the audio passage, then answer the following question by responding with ONLY the number (1, 2, 3, or 4) of the correct answer.

Question: According to the passage, which of the following was one of China’s most violent eras?
1. The Jin Dynasty
2. The Xi’an era
3. The Han Dynasty
4. The Three Kingdoms era

Answer:
----------------------------------------------------------------------
[DEBUG] Audio shape: (679040,)  |  duration: 42.44s
[DEBUG] Input IDs shape: torch.Size([1, 838])
[DEBUG] RAW MODEL OUTPUT: '4. The Three Kingdoms era'
[DEBUG] PARSED ANSWER: 4  |  CORRECT: 4
======================================================================

```

---

## Final Setup

Each sample consists of:
- Concatenated passage audio
- A question
- Four answer choices

If the flag `--no_audio` is set, the audio is omitted and the model is evaluated from the prompt alone.

The model should answer with only `1`, `2`, `3`, or `4`.

---

### Full example:

Example from `config_name='eng_Latn'`, `split='test'`, index 11.

Sentence chunks don’t reliably follow `fleurs_id` or `sentence_idx`; `id`
was the most consistent ordering signal during debugging on the English set.
```
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



