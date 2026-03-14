import os
import time
import argparse
from statistics import mean, stdev

from omegaconf import OmegaConf
import yaml
import wandb
import evaluate
import torch
from tqdm import tqdm
from iso639 import Lang
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    BitsAndBytesConfig,
    HqqConfig,
)

from normalizer import data_utils, data_utils_v2

wer_metric = evaluate.load("wer")


def add_transcription_prompt_to_processor(processor, model_id, language='en', dialect=None):
    try:
        language_fn = Lang(language).name
    except:
        language_fn = language

    language_code = language

    if dialect is not None:
        try:
            language_fn = Lang(dialect).name
        except:
            language_fn = f'{language}-{dialect}'

        language_code = f'{language}-{language_code}'

    if 'Qwen2-Audio' in model_id:
        # https://huggingface.co/Qwen/Qwen2-Audio-7B
        # https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/evaluate_asr.py
        # prompt = '<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:'
        prompt = f'<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and recognize the speech: <|{language_code}|>'
        processor.prompt_asr = prompt

    elif 'Qwen2.5-Omni' in model_id:
        # https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb
        prompt = f'Transcribe the {language_fn} audio into text without any punctuation marks.'
        system_prompt='You are a speech recognition model.'
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
            {'role': 'user', 'content': [
                    {'type': 'audio', 'audio': 'audio_to_transcribe.wav'},
                    {'type': 'text', 'text': prompt},
                ]
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor.prompt_asr = text

    return processor

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


def load_model_and_processor(args):
    quantization_config = load_quant_config(args)

    # Load Qwen2Audio model
    if 'Qwen2.5-Omni' in args.model_id:
        cls = Qwen2_5OmniForConditionalGeneration
    elif 'Qwen2-Audio' in args.model_id:
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


def make_benchmark_fn(model, processor, normalizer, model_input_name, gen_kwargs, args):
    def benchmark(batch, min_new_tokens=None):
        # Load audio inputs
        # audios = [audio["array"] for audio in batch["audio"]]
        audios = [a['array'] for a in batch[args.audio_col_name]]
        batch["audio_length_s"] = [len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios]
        minibatch_size = len(audios)

        # START TIMING
        start_time = time.time()

        # 1. Pre-Processing
        # 1.1 Pad audios to max batch size if using torch compile to prevent re-compilations
        padding_size = None
        if minibatch_size != args.batch_size and args.torch_compile:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        if 'Qwen2.5-Omni' in args.model_id:
            inputs = processor(
                text=processor.prompt_asr,
                audio=audios,
                return_tensors='pt',
                padding=True,
            )
        elif 'Qwen2-Audio' in args.model_id:
            inputs = processor(
                text=[processor.prompt_asr] * len(audios),
                audio=audios,
                sampling_rate=16_000,
                return_tensors='pt',
                padding=True,
            )

        inputs = inputs.to(args.device)
        inputs[model_input_name] = inputs[model_input_name].to(torch.bfloat16)

        # min_new_tokens = None
        pred_ids = model.generate(**inputs, **gen_kwargs, min_new_tokens=min_new_tokens)

        # 3. Post-processing
        # 3.1 Strip padded ids from predictions
        if padding_size is not None:
            pred_ids = pred_ids[:-padding_size, ...]

        # 3.2 Convert token ids to text transcription
        # Decode predictions - skip the prompt tokens
        # Voxtral includes prompt tokens in output, so we slice from input_ids length
        pred_text = processor.batch_decode(
            pred_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # END TIMING
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # Normalize transcriptions (based on norm_english, norm_chinese, or norm_ml)
        batch["predictions"] = [normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]

        return batch

    return benchmark


def run_warmup(dataset, benchmark, args):
    dataset = iter(
        dataset.map(
            benchmark,
            batch_size=args.batch_size,
            batched=True,
            # use max new tokens for early compilation / preparation for worst case
            # this causes problems with mistral tokenizer
            fn_kwargs={'min_new_tokens': args.max_new_tokens},
        )
    )

    for _ in tqdm(dataset, desc='Warming up...'):
        pass

    return 0


def evaluate_dataset(dataset, benchmark, args):
    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=['audio'],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }

    # the map function happens in parallel and processes as a batch tensor/list
    # but this loop functions on a sample level 
    # so each of the elements in results are single values
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    return all_results


def count_params(model):
    return sum([p.numel() for p in model.parameters()])


def compute_and_log_metrics(all_results, model, args):
    # Write manifest results (WER and RTFX)
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
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)

    scores_dic = {
        'wer': wer,
        'rtfx': rtfx
    }

    no_params = round(count_params(model) / 1e6, 4)

    if torch.cuda.is_available():
        max_memory = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 4)
    else:
        max_memory = 0

    try:
        audio_length_s_std = stdev(all_results['audio_length_s'])
    except:
        audio_length_s_std = 0

    scores_dic.update({
        'max_memory': max_memory,
        'no_params': no_params,
        # also include dataset stats
        'num_samples': len(all_results['audio_length_s']),
        'audio_length_s_mean': mean(all_results['audio_length_s']),
        'audio_length_s_std': audio_length_s_std,
        'audio_length_s_min': min(all_results['audio_length_s']),
        'audio_length_s_max': max(all_results['audio_length_s']),
    })

    if args.wandb_project and args.wandb_entity:
        wandb.log(scores_dic)
        wandb.finish()

    return 0


def main():
    args = parse_args()

    model, processor, model_input_name, gen_kwargs = load_model_and_processor(args)

    dataset, normalizer = data_utils_v2.load_and_prepare_dataset(args, warmup=True)

    benchmark = make_benchmark_fn(model, processor, normalizer, model_input_name, gen_kwargs, args)

    if args.warmup_steps:
        run_warmup(dataset, benchmark, args)

    dataset, normalizer = data_utils_v2.load_and_prepare_dataset(args)

    results = evaluate_dataset(dataset, benchmark, args)
    compute_and_log_metrics(results, model, args)

    return 0


def load_config_from_yaml(yaml_path):
    """Load config from YAML file"""
    if isinstance(yaml_path, str):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    elif isinstance(yaml_path, list):
        base_cfg = OmegaConf.load(yaml_path[0])
        for cfg_path in yaml_path[1:]:
            yaml_cfg = OmegaConf.load(cfg_path)
            base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        return base_cfg


def merge_yaml_with_args(parser, yaml_config):
    """Merge YAML config with argparse, CLI args take precedence"""
    # Parse command line args
    args = parser.parse_args()
    
    # If no YAML provided, return CLI args
    if not yaml_config:
        return args
    
    # Convert args to dict
    args_dict = vars(args)
    
    # Update with YAML values only if not set via CLI
    defaults = vars(parser.parse_args([]))  # Get default values
    for key, value in yaml_config.items():
        # Only override if CLI arg is still at default value
        if key in args_dict and args_dict[key] == defaults.get(key):
            args_dict[key] = value
    
    return argparse.Namespace(**args_dict)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default='Qwen/Qwen2-Audio-7B',
        help="Model identifier. Should be loadable with qwen_asr",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hf-audio/esb-datasets-test-only-sorted",
        help="Dataset path. By default, it is `hf-audio/esb-datasets-test-only-sorted`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='tedlium',
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/hf-audio/esb-datasets-test-only-sorted`",
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
        help="Number of samples to go through each batch.",
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
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps to run before launching the timed runs.",
    )

    # new arguments
    # support for easy loading of dataset/models
    parser.add_argument('--config', type=str, default=None, nargs='+',
                       help='Path to YAML config file')

    # model related
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to JIT compile the forward pass of the model.",
    )
    parser.add_argument('--quant_config', type=str, default=None,
                        choices=['bnb', 'hqq'])
    parser.add_argument('--quant_dtype_weights', type=str, 
                        choices=[
                            '4', '8', # for bnb
                            '1', '1.58', '2', '3', '4', '5', '6', '7', '8', # for hqq
                        ],
                        default=None)
    # logging
    parser.add_argument('--serial', type=int, default=0)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)

    # data related
    parser.add_argument('--force_asr_language', type=str, default='en')
    parser.add_argument('--target_sampling_rate', type=int, default=16_000)
    parser.add_argument('--norm_english', action='store_true', help='english normalizer')
    parser.add_argument('--norm_chinese', action='store_true', help='chinese normalizer')

    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    # Load YAML if provided
    if args.config:
        yaml_config = load_config_from_yaml(args.config)
        args = merge_yaml_with_args(parser, yaml_config)

    if torch.cuda.is_available() and args.device != '-1':
        args.device = f'cuda:{args.device}'
    else:
        args.device = 'cpu'

    if 'Qwen2.5-Omni' in args.model_id:
        args.batch_size = 1

    if args.wandb_project and args.wandb_entity:
        q_weights = f'_w{args.quant_dtype_weights}' if args.quant_dtype_weights else ''
        q_config = f'_{args.quant_config}' if args.quant_config else ''
        q_all = f'{q_config}{q_weights}'

        ds = f'{args.dataset_path}_{args.dataset}_{args.split}'

        args.run_name = f'{args.model_id}{q_all}_{ds}_{args.serial}'
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
        wandb.run.name = args.run_name

    return args


if __name__ == "__main__":
    main()
