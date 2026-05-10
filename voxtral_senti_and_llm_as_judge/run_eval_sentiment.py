import argparse
import os
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import evaluate
from normalizer import data_utils_v2, eval_utils
import wandb
from tqdm import tqdm

import base64
import io
import soundfile as sf


#wer_metric = evaluate.load("wer")
accuracy_metric = evaluate.load("accuracy")  

def main(args):
   
    if args.wandb_project and args.wandb_entity:
        dataset_info = f"{args.dataset_path}_sentiment_{args.split}"
        args.run_name = f"{args.model_id}_{dataset_info}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )
        wandb.run.name = args.run_name

    device = "cpu" if args.device == -1 else "cuda" 
    print(f"Loading model: {args.model_id}")

     # Load Voxtral model using transformers
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32, #girish mac change
        device_map=device,  
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
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/voxtral/processing_voxtral.py
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
                            "text": "What is the sentiment of this audio? Reply with one word: positive, negative, or neutral."
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
            outputs = model.generate(**inputs, max_new_tokens=10)

        decoded_outputs = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )


        # normalize transcriptions with English normalizer
        batch["predictions"] = [normalizer(pred) for pred in decoded_outputs]
        batch["references"] = batch["norm_text"]
        return batch

    if args.warmup_steps is not None:
        warmup_dataset = data_utils_v2.load_data(args.dataset_path, None, args.split, args.streaming )
        warmup_dataset, normalizer = data_utils_v2.prepare_data(warmup_dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
        warmup_dataset = iter(warmup_dataset.map(benchmark, batch_size=args.batch_size, batched=True))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    dataset = data_utils_v2.load_data(args.dataset_path, None, args.split, args.streaming)
    dataset, normalizer = data_utils_v2.prepare_data(dataset)

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
        "references": [],
        "Utterance" : [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])
    
    label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2}
    pred_ids = [label2id[pred] for pred in all_results["predictions"]]
    ref_ids  = [label2id[ref] for ref in all_results["references"]]
    
    #print(f"all_results[predictions]: {all_results['predictions']}, " f"all_results[references]: {all_results['references']}, " f"pred_ids: {pred_ids}, ref_ids: {ref_ids}") 
    manifest_path = eval_utils.write_sentiment_manifest(
        all_results["references"],
        all_results["predictions"],
        all_results["Utterance"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    results = accuracy_metric.compute(predictions=pred_ids, references=ref_ids)
    if args.wandb_project and args.wandb_entity and wandb.run is not None:
        wandb.log(results)
        wandb.finish()

    print("ACCURACY: ", results)


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
