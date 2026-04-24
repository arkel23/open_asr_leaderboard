import re

from datasets import load_dataset, Audio

from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer
# from .normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer
# from .normalizer_chinese import ChineseNormalizer
# from .preprocess_specific_datasets import preprocess_dataset


def prepare_filter_language(target_language='en'):
    target_language = 'eng' if target_language == 'en' else target_language

    def is_language(language):
        return language == target_language

    return is_language


def is_target_text_in_range(ref):
    if ref.strip() == 'ignore time segment in scoring':
        return False
    else:
        return ref.strip() != ''


def has_no_repeats(text):
    # This regex matches any character (including Chinese) that repeats 3+ times.
    # (.) matches any character.
    # \1 matches the exact same character as the first one.
    # {2,} matches two or more additional occurrences (total 3+).
    pattern = r'(.)\1{2,}'
    
    # If a match is found, the text is "bad", so we return False to filter it out.
    if re.search(pattern, text):
        return False
    return True


def contains_chinese(text):
    # This regex searches for at least one character in the CJK range
    # If no Chinese character is found, it returns False (filtering the sample out)
    if re.search(r'[\u4e00-\u9fff]', str(text)):
        return True
    return False


# filter data that is shorter than min_input_length or longer than
# max_input_length
def is_audio_in_length_range(length, min_input_length, max_input_length):
    return length > min_input_length and length < max_input_length


def get_audio_col_name(dataset_path):
    if 'dataset-hakka' in dataset_path:
        audio_col_name = 'wav'
    elif 'adi-gov-tw' in dataset_path:
        audio_col_name = 'mp3'
    elif 'ASMR' in dataset_path:
        audio_col_name = 'flac'
    else:
        audio_col_name = 'audio'
    return audio_col_name


def get_text(sample):
    if 'text' in sample:
        return sample['text']
    elif 'sentence' in sample:
        return sample['sentence']
    elif 'normalized_text' in sample:
        return sample['normalized_text']
    elif 'transcript' in sample:
        return sample['transcript']
    elif 'transcription' in sample:
        return sample['transcription']
    elif 'txt' in sample:
        return sample['txt']
    elif 'label' in sample:
        return sample['label']
    elif 'Digit' in sample:
        return sample['Digit']
    elif 'Text' in sample:
        return sample['Text']
    elif 'sentence' in sample:
        return sample['sentence']
    elif 'question_text' in sample:
        return sample['question_text']
    elif 'Questions' in sample:
        return sample['Questions']
    elif 'instruction' in sample:
        return sample['instruction']
    elif 'question' in sample:
        return sample['question']
    elif 'meta_info' in sample:
        return sample['meta_info']
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def make_normalizer(english=False, chinese=False):
    if english:
        normalizer = EnglishTextNormalizer()
    elif chinese:
        raise NotImplementedError
        normalizer = ChineseNormalizer()
    else:
        normalizer = BasicMultilingualTextNormalizer()
    return normalizer


def make_normalize_fn(normalizer):
    def normalize(batch):
        batch['original_text'] = get_text(batch)
        batch['norm_text'] = normalizer(batch['original_text'])

        return batch
    return normalize


def load_data(
        dataset_path='hf-audio/esb-datasets-test-only-sorted',
        dataset_config='tedlium',
        split='test',
        streaming=True
        ):

    # filter based on language for floras
    if 'floras' in dataset_path and 'multilingual' in dataset_config:
        dataset = load_dataset(
            dataset_path,
            'multilingual',
            split=split,
            streaming=streaming,
            token=True,
        )

        is_language = prepare_filter_language(target_language=dataset_config.split('_')[-1])
        dataset = dataset.filter(is_language, input_columns=['language'])

    else:
        dataset = load_dataset(
            dataset_path,
            dataset_config,
            split=split,
            streaming=streaming,
            token=True,
        )

    return dataset


def prepare_data(
        dataset, dataset_path, audio_col_name='audio',
        english=True, chinese=False, sampling_rate=16_000
    ):
    print('Dataset info: ', dataset, dataset_path, audio_col_name, english, chinese)

    # also convert to a uniform format and may need to process from multichannel to single
    # Re-sample to 16kHz and normalise transcriptions
    dataset = dataset.cast_column(audio_col_name, Audio(sampling_rate=sampling_rate))

    '''
    TO DO
    '''
    # preprocess text for datasets that need it
    # dataset = preprocess_dataset(dataset, dataset_path)

    # normalize text
    normalizer = make_normalizer(english, chinese)
    normalize = make_normalize_fn(normalizer)
    # the map function can take num_proc to control number of workers
    dataset = dataset.map(normalize)

    # filter
    # checks for valid text
    dataset = dataset.filter(is_target_text_in_range, input_columns=['norm_text'])
    # check text does not contain 3 or more times a repeated word (indicates noisy data)
    if chinese:
        dataset = dataset.filter(has_no_repeats, input_columns=['norm_text'])
        dataset = dataset.filter(contains_chinese, input_columns=['norm_text'])
    # checks audio is within a range
    # dataset = dataset.filter(is_audio_in_length_range, input_columns=['input_length'])

    return dataset, normalizer


def load_and_prepare_dataset(args, warmup=False):
    # get audio_col_name depending on dataset
    args.audio_col_name = get_audio_col_name(args.dataset_path)

    # load hf datset
    dataset = load_data(args.dataset_path, args.dataset, args.split, args.streaming)

    dataset, normalizer = prepare_data(
        dataset, args.dataset_path, args.audio_col_name,
        args.norm_english, args.norm_chinese, args.target_sampling_rate
    )

    if warmup:
        num = args.warmup_steps * args.batch_size
        if args.streaming:
            dataset = dataset.take(num)
        else:
            dataset = dataset.select(range(min(num, len(dataset))))

        return dataset, normalizer

    if args.max_eval_samples:
        print(f'Subsampling dataset to first {args.max_eval_samples} samples!')
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    return dataset, normalizer

# Fleurs-SLU: Belebele-fleurs evaluation
import base64
import io
import soundfile as sf
import librosa
import numpy as np


def select_audio_based_on_strategy(sentence_chunk: dict, strategy: str, language:str) -> int:

    audios = sentence_chunk.get("audio", [])
    total_audios = len(audios)

    if strategy == "random":
        random_audio_idx = int(np.random.randint(total_audios))
        selected_audio = audios[random_audio_idx]
        return selected_audio

    if strategy == "random":
        random_audio_idx = int(np.random.randint(total_audios))
        selected_audio = audios[random_audio_idx]
        return selected_audio

    # CER STRATEGY
    # https://huggingface.co/datasets/WueNLP/belebele-fleurs
    # Modifications made from the code provided in the README.
    seamless_unsupported = {"ast_Latn", "hau_Latn", "kam_Latn", "kea_Latn", "lin_Latn",
                            "mri_Latn", "nso_Latn", "oci_Latn", "tgl_Latn", "umb_Latn",
                            "wol_Latn", "xho_Latn"}

    whisper_unsupported = {"ast_Latn", "ceb_Latn", "ckb_Arab", "fuv_Latn", "gle_Latn",
                           "ibo_Latn", "kam_Latn", "kea_Latn", "kir_Cyrl", "lug_Latn",
                           "luo_Latn", "nso_Latn", "tgl_Latn", "umb_Latn", "wol_Latn",
                           "xho_Latn", "zul_Latn"}

    if language in whisper_unsupported:
        cer_scores = [sentence_chunk["seamlessm4t_asr_cer"]]
    elif language in seamless_unsupported:
        cer_scores = [sentence_chunk["whisper_asr_cer"]]
    else:
        cer_scores = [sentence_chunk["whisper_asr_cer"], sentence_chunk["seamlessm4t_asr_cer"]]

    averaged_cer_scores = np.stack(cer_scores).mean(axis=0)
    if strategy == "best":
        best_audio_idx = int(np.argmin(averaged_cer_scores))
        selected_audio = audios[best_audio_idx]
        return selected_audio

    if strategy == "worst":
        worst_audio_idx = int(np.argmax(averaged_cer_scores))
        selected_audio = audios[worst_audio_idx]
        return selected_audio


def concatenate_audios(audios: list, target_sample_rate:int) -> np.ndarray | None:
    processed_signal = []
    for audio in audios:
        signal, sr = audio["array"], audio["sampling_rate"]

        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        if sr != target_sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sample_rate)

        processed_signal.append(signal)
    concatenated_audio = np.concatenate(processed_signal, axis=0)

    return concatenated_audio

def audio_array_to_base64(concatenated_audio: np.ndarray) -> str:
    buffer_wav = io.BytesIO()
    sf.write(file=buffer_wav, data=concatenated_audio, samplerate=16_000, format="WAV")
    wav_bytes = buffer_wav.getvalue()
    audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

    return audio_base64

def preprocess_sentence_data(example:list[dict], strategy:str, language:str, target_sampling_rate: int):
    sentence_data = example["sentence_data"]
    sorted_sentence_chunks = sorted(sentence_data, key=lambda x: x["id"])
    audios = [select_audio_based_on_strategy(chunk, strategy=strategy, language=language) for chunk in sorted_sentence_chunks]
    concatenated_audio = concatenate_audios(audios, target_sample_rate=target_sampling_rate)
    example["audio"] = concatenated_audio

    return example

def load_and_prepare_dataset_slu(args, warmup=False):
    dataset = load_dataset(args.dataset_path, args.dataset, split=args.split, streaming=True)
    dataset = dataset.map(preprocess_sentence_data,fn_kwargs={"strategy": args.strategy, "language": args.language, "target_sampling_rate": args.target_sampling_rate})
    dataset = dataset.remove_columns(["sentence_data"])

    if warmup:
        num = args.warmup_steps * args.batch_size
        if args.streaming:
            dataset = dataset.take(num)
        return dataset

    if args.max_eval_samples:
        print(f'Subsampling dataset to first {args.max_eval_samples} samples!')
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)

    if args.max_audio_seconds:
        dataset = dataset.filter(lambda example: len(example["audio"]) / args.target_sampling_rate <= args.max_audio_seconds)

    return dataset