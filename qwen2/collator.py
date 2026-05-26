from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def load_collator(
        processor,
        model_id,
        audio_column='audio',
        text_column='norm_text',
        language='en',
    ):
    if 'Qwen2-Audio' in model_id or 'q2a' in model_id:
        collator = QwenAudioCollator(
            processor=processor,
            audio_column=audio_column, text_column=text_column,
            language=language,
        )
    return collator

def _to_np_float32(audio_dict: dict, dtype=np.float32) -> np.ndarray:
    """Extract numpy float32 array from a HF Audio feature dict."""
    arr = audio_dict["array"]
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=dtype)
    return arr.astype(dtype)


def _pad_list_of_lists(
    sequences: List[List[int]],
    pad_value: int,
    dtype: torch.dtype = torch.long,
    pad_side: str = "left",
) -> torch.Tensor:
    """Right-pad a list of token-id lists to the longest length in the batch."""
    max_len = max(len(s) for s in sequences)
    if pad_side == "left":
        padded = [[pad_value] * (max_len - len(s)) + s for s in sequences]
    else:
        padded = [s + [pad_value] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=dtype)


@dataclass
class QwenAudioCollator:
    """
    Sequence layout:
        [PROMPT tokens] [TRANSCRIPTION tokens] [EOS]
         → labels mask:  all -100             kept       EOS kept

    Qwen2-Audio: decoder-only LLM where <|AUDIO|> placeholder positions in
    input_ids are replaced by audio encoder outputs inside the model.

    The processor handles audio feature extraction + placeholder tokenization
    in a single batched call, which we use as our prompt.
    """
    processor: Any
    sampling_rate: int = 16_000
    # max_label_length: int = 256
    audio_column: str = "audio"
    text_column: str = "text"

    pad_side: str = 'left'
    language: str = "en"

    # Subclasses set this to True to keep BOS on labels (e.g. Whisper needs it
    # stripped; decoder-only models never prepend BOS to labels at all).
    _strip_bos_from_labels: bool = field(default=False, init=False, repr=False)

    def _extract_audios(self, samples: List[dict]) -> List[np.ndarray]:
        return [_to_np_float32(s[self.audio_column]) for s in samples]

    def _extract_texts(self, samples: List[dict]) -> List[str]:
        return [s[self.text_column] for s in samples]

    def _tokenize_labels(
        self,
        texts: List[str],
        tokenizer=None,
        add_special_tokens: bool = False,
    ) -> List[List[int]]:
        """
        Tokenize transcriptions without padding (we pad manually after
        concatenating with prompt ids so the -100 boundary is exact).
        Returns a plain list-of-lists, not tensors.
        """
        tok = tokenizer or self.processor.tokenizer
        encoded = tok(
            texts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_tensors=None,
        )
        return encoded["input_ids"]

    def _build_prompt_inputs(
        self, audios: List[np.ndarray], B: int
    ) -> Dict[str, torch.Tensor]:
        """Return prompt input_ids, attention_mask, plus any audio tensors."""
        raise NotImplementedError

    @staticmethod
    def _strip_padding(ids_row: list, attn_row: list, pad_side: str):
        """Remove intra-batch padding from a single prompt row."""
        real_len = sum(attn_row)
        if real_len == 0:          # degenerate: fully padded row
            return [], []
        if pad_side == 'left':
            return ids_row[-real_len:], attn_row[-real_len:]
        return ids_row[:real_len], attn_row[:real_len]

    @property
    def _prompt_template(self) -> str:
        return self.processor.prompt_asr
        # return (
        #     f"<|audio_bos|><|AUDIO|><|audio_eos|>"
        #     f"Detect the language and recognize the speech: <|{self.language}|>"
        # )

    def _build_prompt_inputs(
        self, audios: List[np.ndarray], B: int
    ) -> Dict[str, torch.Tensor]:
        # Single batched processor call: encodes audio + tokenizes prompt together
        inputs = self.processor(
            text=[self._prompt_template] * B,
            audio=audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        # input_ids, attention_mask, input_features, feature_attention_mask

        # models like Falcon produce token_type_ids
        inputs.pop('token_type_ids', None)

        return dict(inputs)

    def __call__(self, samples: List[dict]) -> Dict[str, torch.Tensor]:
        full_ids, full_attn, full_labels = [], [], []

        audios = self._extract_audios(samples)
        texts  = self._extract_texts(samples)
        B      = len(samples)

        tok = self.processor.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        # 1. Build prompt tensors (model-specific)
        prompt_inputs = self._build_prompt_inputs(audios, B)

        prompt_ids  = prompt_inputs.pop("input_ids")   # (B, L_prompt)
        prompt_attn = prompt_inputs.pop("attention_mask")
        audio_extras = prompt_inputs  # whatever is left (features, masks, …)


        # 2. Tokenize transcriptions (no padding yet)
        label_ids_list = self._tokenize_labels(texts)
 

        for i in range(B):
            p   = prompt_ids[i].tolist()
            pa  = prompt_attn[i].tolist()
            lab = label_ids_list[i]

            p, pa = self._strip_padding(p, pa, self.pad_side)

            ids  = p + lab + [tok.eos_token_id]
            attn = pa + [1] * (len(lab) + 1)
            # Mask prompt positions; keep transcription + EOS
            lbl  = [-100] * len(p) + lab + [tok.eos_token_id]

            full_ids.append(ids)
            full_attn.append(attn)
            full_labels.append(lbl)

        batch = {
            "input_ids":      _pad_list_of_lists(full_ids,    pad_id, pad_side=self.pad_side),
            "attention_mask": _pad_list_of_lists(full_attn,   0, pad_side=self.pad_side),
            "labels":         _pad_list_of_lists(full_labels, -100, pad_side=self.pad_side),
        }
        batch.update(audio_extras)

        return batch


if __name__ == '__main__':
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor

    model_id = 'Qwen/Qwen2-Audio-7B'
    processor = AutoProcessor.from_pretrained(model_id)
    language_code = 'en'
    prompt = f'<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and recognize the speech: <|{language_code}|>'
    processor.prompt_asr = prompt

    dataset = load_dataset('hf-audio/esb-datasets-test-only-sorted', 'voxpopuli', split='test', streaming=True)

    collator = load_collator(processor, model_id, text_column='text')

    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    samples = next(iter(loader))
    print(f'Samples ({model_id} collator): ', type(samples), samples.keys())
