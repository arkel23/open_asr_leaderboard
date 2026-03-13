from datasets import load_dataset

"""

Currently this works for McGuill African Celtic Shared Task Dataset
however language columns can vary: 
- ex - lang_id(with iso-2 iso-3 or iso-2_locale iso_3_locale), language.
- we can build on this.

"""

def load_filtered_dataset(dataset, config, split, language=None, streaming=True):
    ds = load_dataset(dataset, config, split=split, streaming=streaming)

    if language:
        ds = ds.filter(lambda x: x["language"] == language)

    ds = ds.filter(lambda x: x["duration"] <= 30)

    return ds