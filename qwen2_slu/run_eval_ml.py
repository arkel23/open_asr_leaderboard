import argparse
import os
import time
from typing import Any
from statistics import mean, stdev
import logging
import torch
from tqdm import tqdm
import wandb

from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    BitsAndBytesConfig,
    HqqConfig,
)

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

def load_quant_config(args):
    quantization_config = None

    if args.quant_config == 'bnb':
        if args.quant_dtype_weights == '8':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=getattr(args, 'bnb_int8_threshold', 6.0),
            )
        elif args.quant_dtype_weights == '4':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                # if using voxtral this needs to be set to float32
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=getattr(args, 'bnb_4bit_use_double_quant', True),
                bnb_4bit_quant_type=getattr(args, 'bnb_4bit_quant_type', 'fp4'), # can be fp4 or nf4
            )
        else:
            raise NotImplementedError

    elif args.quant_config == 'hqq':
        quantization_config = HqqConfig(
            # officially tested for 1, 1.58, 2, 3, 4, 5, 6 or 8 bits
            nbits=float(args.quant_dtype_weights),
            group_size=getattr(args, 'quant_group_size', 64),
        )

    return quantization_config

def add_transcription_prompt_to_processor(processor, model_id, language='en', dialect=None):

    if 'Qwen2-Audio' in model_id:
        # https://huggingface.co/Qwen/Qwen2-Audio-7B
        # https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/evaluate_asr.py
        # prompt = '<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:'
        #prompt = f'<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and recognize the speech: <|en|>'
        prompt = f'<|audio_bos|><|AUDIO|><|audio_eos|>'
        processor.prompt_asr = prompt

    return processor


def load_model_and_processor(args):
    quantization_config = load_quant_config(args)

    if 'Qwen2-Audio' in args.model_id:
        cls = Qwen2AudioForConditionalGeneration

    model = cls.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation='sdpa',
        device_map='auto',
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor = add_transcription_prompt_to_processor(processor, args.model_id, args.force_asr_language)

    model_input_name = 'input_features'

    gen_kwargs = None

    if model.can_generate():
        # Set generation parameters
        gen_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'do_sample': False,  # Greedy decoding for deterministic transcription
            'num_beams': 1,  # Greedy search
        }

    if args.torch_compile:
        model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=True)
        if model.can_generate():
            # enable static k/v cache for autoregressive models
            model.generation_config.cache_implementation = "static"

    return model, processor, model_input_name, gen_kwargs


def build_qa_prompt(batch: dict[str, list[Any]], idx: int) -> str:

    # Initial prompts inspired from:
    #https://github.com/OpenBMB/UltraEval-Audio/blob/main/registry/prompt/choice.yaml
    # Tested: Qwen2-Audio 7B (8-bit quantization)
    qa_prompt = (
        f"{batch['question'][idx]}\n"
        f"A) {batch['mc_answer1'][idx]}\n"
        f"B) {batch['mc_answer2'][idx]}\n"
        f"C) {batch['mc_answer3'][idx]}\n"
        f"D) {batch['mc_answer4'][idx]}\n"
        f"Answer:"
    )
    return qa_prompt

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
        logger.info(f"Loading audio inputs")
        audios = batch["audio"]
        logger.debug(f"{type(audios[0])=}")
        batch["audio_length_s"] = [len(audio) / 16000 for audio in audios]
        minibatch_size = len(audios)
        start_time = time.time()

        pred_answer_nums: list[int] = []
        for idx in range(minibatch_size):
            qa_prompt = build_qa_prompt(batch, idx)
            prompt = processor.prompt_asr + qa_prompt
            logger.debug(f"prompt text: {prompt}")

            inputs = processor(
                text=prompt,
                audio=audios[idx],
                sampling_rate=16_000,
                return_tensors='pt',
                padding=True,
            )

            logger.debug(f"input_ids shape: {inputs.input_ids.shape}")
            logger.debug(f"input_features shape: {inputs[model_input_name].shape}")
            logger.debug(f"input_ids tokens: {processor.batch_decode(inputs.input_ids, skip_special_tokens=False)[0][:200]}")

            inputs = inputs.to(args.device)
            inputs[model_input_name] = inputs[model_input_name].to(torch.bfloat16)

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                logger.debug(f"outputs shape: {outputs.shape}")

            decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=False)[0]
            logger.debug(f"Raw Prediction: {decoded}")
            logger.debug(f"Reference {batch['correct_answer_num'][idx]}")
            pred = parse_answer_num(decoded)
            pred_answer_nums.append(pred)

        runtime = time.time() - start_time
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default='mistralai/Voxtral-Mini-3B-2507',
        help="Model identifier. Should be loadable with voxtral_asr",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        help="dataset_path from hugging face`",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default='tedlium',
        help="Config name e.g., 'eng_Latn'",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
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
        help="Number of samples to go batch in streaming, this is not batch inferencing.",
    )

    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
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
        "--max_audio_seconds",
        type=float,
        default=None,
        help="Maximum audio duration in seconds (filter out longer samples)."
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps to run before launching the timed runs.",
    )

    # Weights & Biases Logging
    parser.add_argument('--serial', type=int, default=0)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)

    # Logger Logging
    parser.add_argument('--debug', action='store_true', help='Enable Debug mode')

    # Data Related
    parser.add_argument('--target_sampling_rate', type=int, default=16_000)

    #parser.add_argument('--force_asr_language', type=str, default='en')
    parser.add_argument('--torch_compile', action='store_true')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead')
    parser.add_argument('--quant_config', type=str, default=None,
                        choices=['bnb', 'hqq'])
    parser.add_argument('--quant_dtype_weights', type=str,
                        choices=[
                            '4', '8', # for bnb
                            '1', '1.58', '2', '3', '4', '5', '6', '7', '8', # for hqq
                        ],
                        default=None)

    args = parser.parse_args()
    # parser.set_defaults(streaming=False)

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