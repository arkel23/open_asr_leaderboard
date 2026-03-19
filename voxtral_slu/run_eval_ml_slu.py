import argparse
import os
import tempfile
import time
from argparse import Namespace
from statistics import mean, stdev
from typing import Any, Callable, Union

import soundfile as sf
import torch
import wandb
from datasets import Dataset, IterableDataset
from tqdm import tqdm
from transformers import AutoProcessor, VoxtralForConditionalGeneration

from normalizer import data_utils_v2, evals_utils_v2


UNDER_TIME_LIMIT_AUDIO_COUNTER = 0

def get_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Voxtral on the Belebele–FLEURS spoken language understanding (SLU) dataset."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier 'mistralai/Voxtral-Mini-3B-2507'."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset path 'WueNLP/belebele-fleurs'."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Dataset configuration name (e.g., eng_Latn, hin_Deva, fra_Latn, ibo_Latn)."
    )
    parser.add_argument(
        "--no_audio",
        action="store_true",
        help="Run text-only MCQ baseline (exclude audio from the prompt)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device index (-1 for CPU, >=0 for GPU)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of evaluation samples (default: use all)."
    )
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=None,
        help="Maximum audio duration in seconds (filter out longer samples)."
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable dataset streaming (enabled by default)."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum number of tokens generated per sample."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warmup steps to run before starting evaluation."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the reference passage and sentence order used when concatenating audio, as well as the MCQ Template used."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="best",
        choices=["best", "worst", "random"],
        help="Sentence chunk selection strategy based on CER for audio construction."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name where run results will be logged (optional)."
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (workspace) that owns the project where runs are logged (optional)."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Authentication token (e.g., Hugging Face access token)."
    )

    parser.set_defaults(streaming=True)
    args = parser.parse_args()
    return args

def parse_answer_num(text: str) -> int:
    for ch in text:
        if ch in "1234":
            return int(ch)
    return -1

def build_mcq_prompt(batch: dict[str, list[Any]], i: int, with_audio: bool) -> str:
    if with_audio:
        intro = (
            "Listen to the audio passage, then answer the following question "
            "by responding with ONLY the number (1, 2, 3, or 4) of the correct answer."
        )
    else:
        intro = (
            "Given the question and answer choices, respond with ONLY the number "
            "(1, 2, 3, or 4) of the correct answer."
        )

    return (
        f"{intro}\n\n"
        f"Question: {batch['question'][i]}\n"
        f"1. {batch['mc_answer1'][i]}\n"
        f"2. {batch['mc_answer2'][i]}\n"
        f"3. {batch['mc_answer3'][i]}\n"
        f"4. {batch['mc_answer4'][i]}\n\n"
        "Answer:"
    )


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def load_voxtral(args: Namespace):
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=f"cuda:{args.device}" if args.device >= 0 else "cpu",
    )
    model.eval()
    return model, processor

def run_voxtral_on_batch(
        batch: dict[str, list[Any]],
        batch_size: int,
        model,
        processor,
        debug: bool,
        args: Namespace
) -> list[int]:
    pred_answer_nums: list[int] = []

    audios = batch["audio"]
    sampling_rates = batch["sampling_rate"]

    for i in range(batch_size):
        use_audio = not args.no_audio
        mcq_prompt = build_mcq_prompt(batch, i, with_audio=use_audio)

        if not use_audio:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": mcq_prompt}],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                return_tensors="pt",
            )
            audio = None
            sampling_rate = None
        else:
            audio = audios[i]
            sampling_rate = sampling_rates[i]

            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.close()
            sf.write(tmp_wav.name, audio, sampling_rate)

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": tmp_wav.name},
                            {"type": "text", "text": mcq_prompt},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    return_tensors="pt",
                )
            finally:
                os.unlink(tmp_wav.name)

        inputs = inputs.to(model.device)

        if isinstance(inputs, dict):
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(model.device)

        if debug:
            print("\n" + "=" * 70)
            print(
                f"[DEBUG] Sample {i + 1}/{batch_size}  |  "
                f"question_num: {batch['question_num'][i]}"
            )
            print(f"[DEBUG] RUN TAG: {args.run_tag}")
            print("-" * 70)

            if "flores_passage" in batch:
                print("[DEBUG] FLORES PASSAGE (full reference passage):")
                print(batch["flores_passage"][i])
                print("-" * 70)

            if "sentences" in batch:
                print("[DEBUG] SELECTED SENTENCES (concatenation order):")
                for j, sent in enumerate(batch["sentences"][i]):
                    print(f"  [{j}] {sent}")
                print("-" * 70)

            print("[DEBUG] MCQ PROMPT:")
            print(mcq_prompt)
            print("-" * 70)

            if use_audio and audio is not None and sampling_rate is not None:
                print(
                    f"[DEBUG] Audio shape: {audio.shape}  |  "
                    f"duration: {len(audio) / sampling_rate:.2f}s"
                )

            if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
                print(f"[DEBUG] Input IDs shape: {inputs.input_ids.shape}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        prompt_len = 0
        if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
            prompt_len = inputs.input_ids.shape[1]

        generated_ids = outputs[:, prompt_len:] if prompt_len > 0 else outputs

        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        pred = parse_answer_num(decoded)
        pred_answer_nums.append(pred)

        if debug:
            print(f"[DEBUG] RAW MODEL OUTPUT: '{decoded}'")
            print(
                f"[DEBUG] PARSED ANSWER: {pred}  |  "
                f"CORRECT: {batch['correct_answer_num'][i]}"
            )
            print("=" * 70)

    return pred_answer_nums


def benchmark(batch: dict[str, list[Any]], is_warmup: bool, model, processor, args: Namespace) -> dict[str, list[Any]]:
    global UNDER_TIME_LIMIT_AUDIO_COUNTER
    debug = args.debug and not is_warmup

    audios = batch["audio"]
    sampling_rates = batch["sampling_rate"]

    if args.max_audio_seconds is not None:
        original_batch_size = len(audios)
        kept_audio_indices = []

        for idx, (audio, sampling_rate) in enumerate(zip(audios, sampling_rates)):
            audio_length_seconds = len(audio) / sampling_rate
            if audio_length_seconds <= args.max_audio_seconds:
                kept_audio_indices.append(idx)

        for key in list(batch.keys()):
            batch[key] = [batch[key][i] for i in kept_audio_indices]

        audios = batch["audio"]
        sampling_rates = batch["sampling_rate"]

        if not is_warmup:
            UNDER_TIME_LIMIT_AUDIO_COUNTER += len(audios)

        if debug:
            print(
                f"\n[DEBUG] Duration filter <= {args.max_audio_seconds}s: "
                f"kept {len(audios)} / {original_batch_size} samples"
            )
    else:
        if not is_warmup:
            UNDER_TIME_LIMIT_AUDIO_COUNTER += len(audios)

    if len(audios) == 0:
        batch["audio_length_s"] = []
        batch["inference_time_s"] = []
        batch["predictions"] = []
        batch["references"] = []
        batch["question_num"] = []
        return batch

    batch["audio_length_s"] = [
        len(audio) / sr for audio, sr in zip(audios, sampling_rates)
    ]
    minibatch_size = len(audios)
    start_time = time.time()
    pred_answer_nums = run_voxtral_on_batch(batch, minibatch_size, model, processor, debug, args)
    runtime = time.time() - start_time
    batch["inference_time_s"] = [runtime / minibatch_size] * minibatch_size
    batch["predictions"] = pred_answer_nums
    batch["references"] = [int(c) for c in batch["correct_answer_num"]]

    return batch


def start_warmup(
        dataset: Union[IterableDataset, Dataset],
        benchmark_func: Callable[..., dict[str, list[Any]]],
        model,
        processor,
        args: Namespace,
) -> None:
    dataset = iter(
        dataset.map(
            benchmark_func,
            batch_size=args.batch_size,
            batched=True,
            fn_kwargs={
                "is_warmup": True,
                "model": model,
                "processor": processor,
                "args": args,
            },
        )
    )

    for _ in tqdm(dataset, desc="Warming up..."):
        pass


def start_evaluation(
        dataset: Union[IterableDataset, Dataset],
        benchmark_func: Callable[..., dict[str, list[Any]]],
        model,
        processor,
        args: Namespace,
) -> Union[IterableDataset, Dataset]:
    dataset = dataset.map(
        benchmark_func,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
        fn_kwargs={
            "is_warmup": False,
            "model": model,
            "processor": processor,
            "args": args,
        },
    )
    return dataset

def collect_results(dataset: Union[IterableDataset, Dataset]) -> dict[str, list[Any]]:
    all_results = {
        "audio_length_s": [],
        "inference_time_s": [],
        "predictions": [],
        "references": [],
        "question_num": [],
    }

    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Batch Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    return all_results

def compute_and_log_metrics(
        all_results: dict[str, list[Any]],
        model,
        args: Namespace,
) -> dict[str, Any]:
    manifest_path = evals_utils_v2.write_manifest_slu(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.config_name,
        args.split,
        question_nums=all_results["question_num"],
        audio_length=all_results["audio_length_s"],
        inference_time=all_results["inference_time_s"],
        run_tag=args.run_tag,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    correct = sum(
        int(pred == ref)
        for pred, ref in zip(all_results["predictions"], all_results["references"])
    )
    total_predictions = len(all_results["predictions"])
    accuracy = round(100 * correct / total_predictions, 2) if total_predictions > 0 else 0.0

    total_audio_s = sum(all_results["audio_length_s"])
    total_inference_s = sum(all_results["inference_time_s"])
    rtfx = round(total_audio_s / total_inference_s, 2) if total_inference_s > 0 else 0.0

    no_params = round(count_params(model) / 1e6, 4)
    if torch.cuda.is_available():
        max_memory = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 4)
    else:
        max_memory = 0

    try:
        audio_length_s_std = stdev(all_results["audio_length_s"])
    except Exception:
        audio_length_s_std = 0.0

    if total_predictions > 0:
        audio_length_s_mean = mean(all_results["audio_length_s"])
        audio_length_s_min = min(all_results["audio_length_s"])
        audio_length_s_max = max(all_results["audio_length_s"])
    else:
        audio_length_s_mean = 0.0
        audio_length_s_min = 0.0
        audio_length_s_max = 0.0

    scores_dict = {
        "accuracy": accuracy,
        "rtfx": rtfx,
        "max_memory": max_memory,
        "no_params": no_params,
        "run_tag": args.run_tag,
        "num_samples": total_predictions,
        "under_time_limit_audio_counter": UNDER_TIME_LIMIT_AUDIO_COUNTER,
        "audio_length_s_mean": audio_length_s_mean,
        "audio_length_s_std": audio_length_s_std,
        "audio_length_s_min": audio_length_s_min,
        "audio_length_s_max": audio_length_s_max,
    }

    print(f"Accuracy: {accuracy}% ({correct}/{total_predictions})  RTFx: {rtfx}")
    print(f"Scores_dict: {scores_dict}")
    return scores_dict

def main():
    args = get_args()

    audio_tag = "no_audio" if args.no_audio else "with_audio"
    seconds_tag = (
        f"max{int(args.max_audio_seconds)}s"
        if args.max_audio_seconds is not None
        else "all_lengths"
    )
    args.run_tag = f"{audio_tag}_{seconds_tag}_{args.strategy}"

    if args.wandb_project and args.wandb_entity:
        dataset_info = f"{args.dataset_path}_{args.config_name}_{args.split}"
        args.run_name = f"{args.model_id}_{dataset_info}_{args.run_tag}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )
        wandb.run.name = args.run_name

    model, processor = load_voxtral(args)

    dataset = data_utils_v2.load_dataset_bb_fleurs_slu(args, warmup=True)
    if args.warmup_steps:
        start_warmup(dataset, benchmark, model, processor, args)

    dataset = data_utils_v2.load_dataset_bb_fleurs_slu(args)
    dataset = start_evaluation(dataset, benchmark, model, processor, args)
    all_results = collect_results(dataset)
    scores_dict = compute_and_log_metrics(all_results, model, args)

    if args.wandb_project and args.wandb_entity and wandb.run is not None:
        wandb.log(scores_dict)
        wandb.finish()

if __name__ == "__main__":
    main()