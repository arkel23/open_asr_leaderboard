import argparse
import os
import logging
import time
from typing import Any
from statistics import mean, stdev

import torch

from tqdm import tqdm
import wandb

from transformers import VoxtralForConditionalGeneration, AutoProcessor

from normalizer import data_utils, data_utils_v2

logger = logging.getLogger(__name__)

def setup_logging(debug=False):
    formatter = logging.Formatter(
        "{levelname} | {module} | {funcName} | {lineno} | {asctime} : {message}",
        style="{",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if debug:
        file_handler = logging.FileHandler("debug.log", mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def load_model_and_processor(args):

    logger.info(f"Loading model and processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
    )

    model = model.to(args.device)
    model.eval()

    model_input_name = 'input_features'

    gen_kwargs = {}
    if model.can_generate():
        gen_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'do_sample': False,
            'num_beams': 1,
        }

    return model, processor, model_input_name, gen_kwargs


def build_conversation(audio_base64, batch:dict[str, list[Any]], idx:int) -> list[dict]:
    logger.info("Building conversation")
    instruction = "Listen to the audio passage, then answer the following question by responding with ONLY the number (1, 2, 3, or 4) of the correct answer."
    prompt = (
        f"Question: {batch['question'][idx]}\n"
        f"1. {batch['mc_answer1'][idx]}\n"
        f"2. {batch['mc_answer2'][idx]}\n"
        f"3. {batch['mc_answer3'][idx]}\n"
        f"4. {batch['mc_answer4'][idx]}\n\n"
        f"{instruction}\n\n"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "base64": audio_base64,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


    return conversation


def parse_answer_num(text: str) -> int:
    letter_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    for ch in text:
        if ch in "1234":
            return int(ch)
        if ch in letter_map:
            return letter_map[ch]
    return -1

def make_benchmark_fn(model, processor, model_input_name, gen_kwargs, args):
    def benchmark(batch):
        logger.info("Initialize benchmark on batch")
        logger.debug(f"Batch keys: {list(batch.keys())}")
        logger.debug(f"Batch sizes: { {k: len(v) for k, v in batch.items()} }")

        logger.info(f"Loading audio inputs")
        # For the SLU pipeline
        # batch["audio"] is a list of numpy arrays, not an HF dict-like object
        # ({"array", "sampling_rate"}). Set by preprocess_sentence_data() in data_utils_v2.py.
        audios = batch["audio"]
        logger.debug(f"batch['audio']{type(batch['audio'][0])}")
        # logger.debug(f"batch['audio']['array'] {type(batch['audio']['array'][0])}")

        logger.debug(f"Num audios: {len(audios)}, first audio shape: {audios[0].shape}")
        logger.debug(f"Audio sample lengths (raw samples): {[len(a) for a in audios]}")

        batch["audio_length_s"] = [len(audio) / 16000 for audio in audios]

        logger.debug(f"Audio lengths (s): {batch['audio_length_s']}")
        minibatch_size = len(audios)
        start_time = time.time()
        pred_answer_nums = []
        for idx in range(minibatch_size):
            logger.debug(f"audios[idx] type {type(audios[idx])}")

            audio_base64 = data_utils_v2.audio_array_to_base64(audios[idx])
            logger.debug(f"Audio base64 Type: {type(audio_base64)}")

            conversation = build_conversation(audio_base64, batch, idx)
            logger.debug(f"conversation: audio={conversation[0]['content'][0]['base64'][:300]}... text={conversation[0]['content'][1]['text']}")
            inputs = processor.apply_chat_template(conversation)
            logger.debug(f"input_ids shape: {inputs.input_ids.shape}, dtype: {inputs.input_ids.dtype}")
            logger.debug(f"input_features shape: {inputs[model_input_name].shape}, dtype: {inputs[model_input_name].dtype}")
            logger.debug(f"input_ids tokens: {processor.batch_decode(inputs.input_ids, skip_special_tokens=False)[0][:200]}")

            inputs = inputs.to(args.device)
            inputs[model_input_name] = inputs[model_input_name].to(torch.bfloat16)
            logger.debug(f"input_features device: {inputs[model_input_name].device}, dtype after cast: {inputs[model_input_name].dtype}")

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            logger.debug(f"Outputs shape: {outputs.shape}")

            decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            logger.debug(f"Raw prediction: {decoded}")
            logger.debug(f"Reference: {batch['correct_answer_num'][idx]}")

            pred = parse_answer_num(decoded)
            pred_answer_nums.append(pred)

        runtime = time.time() - start_time
        logger.debug(f"{runtime=}")

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = pred_answer_nums
        batch["references"] = [int(answer) for answer in batch["correct_answer_num"]]

        return batch

    return benchmark


def run_warmup(dataset, benchmark, args):
    logger.info("Initializing warmup steps")
    dataset = iter(
        dataset.map(
            benchmark,
            batch_size=args.batch_size,
            batched=True,
        )
    )

    for _ in tqdm(dataset, desc='Warming up...'):
        pass

    return 0



def evaluate_dataset(dataset, benchmark, args):
    logger.info("Initializing dataset evaluation")
    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": []
    }

    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    logger.debug(f"Total samples collected: {len(all_results['predictions'])}")
    logger.debug(f"Predictions: {all_results['predictions']}")
    logger.debug(f"References: {all_results['references']}")

    return all_results


def count_params(model):
    return sum([p.numel() for p in model.parameters()])


def get_machine_and_data_stats(all_results, model, args):
    no_params = round(count_params(model) / 1e6, 4)

    if torch.cuda.is_available():
        max_memory = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 4)
    else:
        max_memory = 0

    try:
        audio_length_s_std = stdev(all_results['audio_length_s'])
    except:
        audio_length_s_std = 0

    info = {
        'max_memory': max_memory,
        'no_params': no_params,
        'num_samples': len(all_results['audio_length_s']),
        'audio_length_s_mean': mean(all_results['audio_length_s']),
        'audio_length_s_std': audio_length_s_std,
        'audio_length_s_min': min(all_results['audio_length_s']),
        'audio_length_s_max': max(all_results['audio_length_s']),
    }
    return info


def compute_and_log_metrics(all_results, model, args):
    logger.info("Computing final metrics")
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    logger.info(f"Results saved at path:  {os.path.abspath(manifest_path)}")
    correct = sum(int(pred == ref) for pred, ref in zip(all_results["predictions"], all_results["references"]))
    total_predictions = len(all_results["predictions"])
    accuracy = round(100 * correct / total_predictions, 2) if total_predictions > 0 else 0.0
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)

    logger.info(f"accuracy:, {accuracy}%, RTFx: {rtfx}")
    scores_dic = {'accuracy': accuracy, 'rtfx': rtfx}
    scores_dic.update(get_machine_and_data_stats(all_results, model, args))


    logger.info(f"{scores_dic}")
    if args.wandb_project and args.wandb_entity:
        wandb.log(scores_dic)
        wandb.finish()

    return 0

def main():
    args = parse_args()
    setup_logging(debug=args.debug)

    logger.info("Initializing Run")
    logger.info(f"Run config: {vars(args)}")
    model, processor, model_input_name, gen_kwargs = load_model_and_processor(args)

    logger.info("Loading dataset for warmup steps")
    dataset = data_utils_v2.load_and_prepare_dataset_slu(args, warmup=True)
    benchmark = make_benchmark_fn(model, processor, model_input_name, gen_kwargs, args)

    if args.warmup_steps:
        run_warmup(dataset, benchmark, args)

    logger.info("Resetting dataset for evaluation")
    dataset = data_utils_v2.load_and_prepare_dataset_slu(args)
    results = evaluate_dataset(dataset, benchmark, args)

    logger.info("Run Complete")
    compute_and_log_metrics(results, model, args)

    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Voxtral on the Fleurs-SLU: Belebele-Fleurs evaluation task."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default='mistralai/Voxtral-Mini-3B-2507',
        help="Model identifier 'mistralai/Voxtral-Mini-3B-2507'.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Dataset path 'WueNLP/belebele-fleurs'.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='eng_Latn',
        help="Dataset configuration name (e.g., eng_Latn, hin_Deva, fra_Latn, ibo_Latn).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to go through each batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of evaluation samples. If not set, all samples are evaluated.",
    )
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=None,
        help="Maximum audio duration in seconds (filter out longer samples).",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps to run before launching the timed runs.",
    )

    # Data
    parser.add_argument(
        "--strategy",
        type=str,
        default="best",
        choices=["best", "worst", "random"],
        help="Sentence chunk selection strategy based on CER for audio construction.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="eng_Latn",
        help="Dataset configuration name (e.g., eng_Latn, hin_Deva, fra_Latn, ibo_Latn).",
    )
    parser.add_argument(
        "--target_sampling_rate",
        type=int,
        default=16_000,
        help="Target sampling rate for audio.",
    )

    # Logger
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (verbose output and debug.log file).",
    )

    # Weights & Biases
    parser.add_argument(
        "--serial",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name where run results will be logged.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity (workspace) that owns the project where runs are logged.",
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device != -1:
        args.device = f"cuda:{args.device}"
    else:
        args.device = "cpu"

    if args.wandb_project and args.wandb_entity:
        ds = f'{args.dataset_path}_{args.dataset}_{args.split}'
        args.run_name = f'{args.model_id}_{ds}_{args.serial}'
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
        wandb.run.name = args.run_name

    return args


if __name__ == "__main__":
    main()