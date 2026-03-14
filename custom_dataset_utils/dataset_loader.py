from datasets import load_dataset, Audio, Dataset



"""

Currently this works for McGuill African Celtic Shared Task Dataset
however language columns can vary: 
- ex - lang_id(with iso-2 iso-3 or iso-2_locale iso_3_locale), language.
- we can build on this.

"""
## 

def load_filtered_dataset(dataset, config, split, language=None, streaming=True, token=True):
    ds = load_dataset(dataset, config, split=split, streaming=streaming, token=token)

    if "wav" in ds.column_names:
        ds = ds.rename_column("wav", "audio")

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    ds = ds.filter(lambda x: x["duration"] is not None and x["duration"] <= 30)

    return ds



from datasets import load_dataset, Audio, IterableDataset

def load_filtered_data_slu_fleurs_audios(dataset, config, split, streaming=True, token=True):
    slu_ds = load_dataset(dataset, config, split=split, streaming=streaming, token=token)

    def generate_samples():
        for slu_sample in slu_ds:
            for sentence_dict in slu_sample["sentence_data"]:
                for i, audio_obj in enumerate(sentence_dict["audio"]):
                    yield {
                        "audio": audio_obj,
                        "text": sentence_dict["transcription"],
                    }

    fleurs_ds = IterableDataset.from_generator(generate_samples)
    fleurs_ds = fleurs_ds.cast_column("audio", Audio(sampling_rate=16000))

    fleurs_ds = fleurs_ds.filter(
        lambda x: len(x["audio"]["array"]) / x["audio"]["sampling_rate"] <= 30
    )

    return fleurs_ds