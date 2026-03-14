import argparse
import os
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2_5OmniForConditionalGeneration
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm
# from datasets import load_dataset, Audio
from iso639 import Lang

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


def main(args):
    # CONFIG_NAME = args.config_name
    # SPLIT_NAME = args.split

    # config = AutoConfig.from_pretrained(args.model_id)

    # Load Qwen2Audio model
    if 'Qwen2.5-Omni' in args.model_id:
        cls = Qwen2_5OmniForConditionalGeneration
    elif 'Qwen2-Audio' in args.model_id:
        cls = Qwen2AudioForConditionalGeneration
    model = cls.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=f"cuda:{args.device}" if args.device >= 0 else "cpu",
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    args.force_asr_language = 'en'
    processor = add_transcription_prompt_to_processor(processor, args.model_id, args.force_asr_language)

    # model_input_name = processor.model_input_names[0]

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

    # Load dataset using the HuggingFace dataset repository
    # print(f"Loading dataset: {args.dataset} with config: {CONFIG_NAME}")

    # dataset = load_dataset(
    #     args.dataset,
    #     CONFIG_NAME,
    #     split=SPLIT_NAME,
    #     streaming=args.streaming,
    #     token=True,
    # )

    # Resample to 16kHz
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # if args.max_eval_samples is not None and args.max_eval_samples > 0:
    #     print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
    #     if args.streaming:
    #         dataset = dataset.take(args.max_eval_samples)
    #     else:
    #         dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    def benchmark(batch, min_new_tokens=None):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios]
        minibatch_size = len(audios)

        # Load audio inputs
        # audios = [audio["array"] for audio in batch["audio"]]
        # minibatch_size = len(audios)
        
        # # Compute audio length in seconds (16kHz sampling rate)
        # batch["audio_length_s"] = [len(audio) / 16_000 for audio in audios]

        # START TIMING
        start_time = time.time()

        # INFERENCE
        # Qwen3-ASR expects audio as file paths or numpy arrays with sample rate
        # audio_inputs = [(audio, 16000) for audio in audios]

        # 1. Pre-Processing
        # 1.1 Pad audios to max batch size if using torch compile to prevent re-compilations
        padding_size = None
        if minibatch_size != args.batch_size and args.torch_compile:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        # inputs = processor(audios, sampling_rate=16_000, return_tensors="pt", device=args.device)
        if 'Qwen2.5-Omni' in args.model_id:
            inputs = processor(
                text=processor.prompt_asr,
                # text=[processor.prompt_asr] * len(audios),
                audio=audios,
                # sampling_rate=16_000,
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
        # inputs[model_input_name] = inputs[model_input_name].to(torch.bfloat16)
        inputs['input_features'] = inputs['input_features'].to(torch.bfloat16)

        # min_new_tokens = None
        pred_ids = model.generate(**inputs, **gen_kwargs, min_new_tokens=min_new_tokens)
        # results = model.generate(
        #     audio=audio_inputs,
        #     language=None,  # Auto-detect language
        # )

        # 3. Post-processing
        # 3.1 Strip padded ids from predictions
        if padding_size is not None:
            pred_ids = pred_ids[:-padding_size, ...]

        # 3.2 Convert token ids to text transcription
        # pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # Decode predictions - skip the prompt tokens
        # Voxtral includes prompt tokens in output, so we slice from input_ids length
        pred_text = processor.batch_decode(
            pred_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
            # add_special_tokens=False,
        )

        # Extract text predictions
        # pred_text = [r.text for r in results]

        # END TIMING
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # Normalize transcriptions with multilingual normalizer
        batch["predictions"] = [data_utils.ml_normalizer(pred) for pred in pred_text]
        # Get references from the dataset
        batch["references"] = batch["norm_text"]
        # batch["references"] = [data_utils.ml_normalizer(ref) for ref in batch["references"]]

        return batch

    if args.warmup_steps is not None:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True, fn_kwargs={"min_new_tokens": args.max_new_tokens}))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data(dataset)

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }

    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with qwen_asr",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'librispeech_asr` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice. The full list of dataset names "
        "can be found at `https://huggingface.co/datasets/esb/datasets`",
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
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
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
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
