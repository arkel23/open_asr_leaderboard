from datasets import load_dataset, Audio,  IterableDataset
from normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer

from .eval_utils import read_manifest, write_manifest


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    elif "Sentiment" in sample:
        return sample["Sentiment"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )

normalizer = EnglishTextNormalizer()

ml_normalizer = BasicMultilingualTextNormalizer()


def normalize(batch):
    batch["original_text"] = get_text(batch)
    batch["norm_text"] = normalizer(batch["original_text"])
    return batch


def load_data(args):
    if args.dataset is None:
        dataset = load_dataset(
        args.dataset_path,
        split=args.split,
        streaming=args.streaming,
        token=True,
    )
    else: 
        dataset = load_dataset(
            args.dataset_path,
            args.dataset,
            split=args.split,
            streaming=args.streaming,
            token=True,
        )

    return dataset


def prepare_data(dataset):
    # Re-sample to 16kHz and normalise transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # Use writer_batch_size=1 to avoid pyarrow offset overflow with large audio blobs
    if isinstance(dataset, IterableDataset):     # this change if streaming is set to true 
        dataset = dataset.map(normalize)
        dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])
    else:
        dataset = dataset.map(normalize, writer_batch_size=1)
        dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"], writer_batch_size=1)

    return dataset


