import argparse
import os
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from normalizer import data_utils_v2, eval_utils
from datasets import Audio
from tqdm import tqdm
import gc
import base64
import io
import soundfile as sf






def main(args):

    # Load Voxtral model using transformers
    device = "cpu" if args.device == -1 else "mps"  #girish change
    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32, #girish mac change
        device_map=device,   #girish change
    )
    model.eval()

    def benchmark(batch):
        # Load audio inputs
        # INFERENCE
        # Process audio inputs for transcription
        conversations = []
        for audio_dict in batch["audio"]:
            audio = audio_dict["array"]
            sr = audio_dict["sampling_rate"]

            # convert audio → base64 WAV
            buffer = io.BytesIO()
            sf.write(buffer, audio, sr, format="WAV")
            audio_b64 = base64.b64encode(buffer.getvalue()).decode()

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": audio_b64},
                        },
                        {
                            "type": "text",
                            "text": "give correct answer to the question asked in this audio"
                        },
                    ],
                }
            ]
            conversations.append(conversation)

        inputs = processor.apply_chat_template(
        conversations,
        return_tensors="pt",
        padding=True
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30)

        decoded_outputs = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        normalizer = data_utils_v2.make_normalizer(english=True)

        # normalize transcriptions with English normalizer
        batch["predictions"] = [normalizer(pred) for pred in decoded_outputs]
        return batch

    if args.warmup_steps is not None:
        warmup_dataset = data_utils_v2.load_data(args.dataset_path, None, args.split, args.streaming)
        warmup_dataset = warmup_dataset.cast_column("audio", Audio(sampling_rate=16000))

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils_v2.load_data(args.dataset_path, None, args.split, args.streaming)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    all_results = {
        "predictions": [],
        "question" : [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])
    
    
    #print(f"all_results[predictions]: {all_results['predictions']}, " f"all_results[references]: {all_results['question']}")
    manifest_path = eval_utils.write_llm_as_judge_manifest(
        all_results["question"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))
    del dataset
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with transformers (e.g., 'mistralai/Voxtral-Mini-3B-2507')",
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
        default=8,
        help="Number of samples to go through each streamed batch.",
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
        default=500,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
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
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
