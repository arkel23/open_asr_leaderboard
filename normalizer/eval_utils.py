import os
import glob
import json
import wandb
import evaluate
from collections import defaultdict
from .llm_as_judge_eval_utils import run_llm_as_judge



def read_manifest(manifest_path: str):
    """
    Reads a manifest file (jsonl format) and returns a list of dictionaries containing samples.
    """
    data = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) > 0:
                datum = json.loads(line)
                data.append(datum)
    return data

def get_manifest_path( 
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    split: str):

    model_id = model_id.replace("/", "-")
    dataset_path = dataset_path.replace("/", "-")
    if dataset_name is None:
        dataset_name = dataset_path
    
    basedir = "./results/"
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    manifest_path = os.path.join(
        basedir, f"MODEL_{model_id}_DATASET_{dataset_path}_{dataset_name}_{split}.jsonl"
    )
    return manifest_path 


def write_sentiment_manifest(
    references: list,
    predicitons: list,
    user_text: list,
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
):
    """
    Writes a manifest file (jsonl format) and returns the path to the file.

    Args:
        references: Ground truth reference texts.
        predicitons: Model predicted transcriptions.
        user_text: audio transribed text by user.
        model_id: String identifier for the model.
        dataset_path: Path to the dataset.
        dataset_name: Name of the dataset.
        split: Dataset split name.

    Returns:
        Path to the manifest file.
    """

    if len(references) != len(predicitons):
        raise ValueError(
            f"The number of samples in `references` ({len(references)}) "
            f"must match `predicitons` ({len(predicitons)})."
        )

    if user_text is not None and len(user_text) != len(references):
        raise ValueError(
            f"The number of samples in `user_text` ({len(user_text)}) "
            f"must match `references` ({len(references)})."
        )
    
    basedir = "./results/"
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    

    manifest_path = get_manifest_path(model_id, dataset_path, dataset_name, split)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx, (usr_text, ref, pred ) in enumerate(
            zip(user_text, references, predicitons)
        ):
            datum = {
                "user_text": usr_text,
                "references": ref,
                "pred_text": pred,
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")
    return manifest_path

def write_llm_as_judge_manifest(
    question: list,
    predicitons: list,
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
):
    """
    Writes a manifest file (jsonl format) and returns the path to the file.

    Args:
        references: Ground truth reference texts.
        predicitons: Model predicted transcriptions.
        user_text: audio transribed text by user.
        model_id: String identifier for the model.
        dataset_path: Path to the dataset.
        dataset_name: Name of the dataset.
        split: Dataset split name.

    Returns:
        Path to the manifest file.
    """

    if len(question) != len(predicitons):
        raise ValueError(
            f"The number of samples in `question` ({len(question)}) "
            f"must match `predicitons` ({len(predicitons)})."
        )

    manifest_path = get_manifest_path(model_id, dataset_path, dataset_name, split)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx, (ques, pred ) in enumerate(
            zip(question,  predicitons)
        ):
            datum = {
                "question": ques,
                "prediction": pred,
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")
    return manifest_path

def write_llm_scores_manifest(
    scores: list,
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
):
    """
    Writes LLM-as-judge scores to a jsonl file.

    Args:
        scores: List of dicts with keys like question, prediction, score
    """

    manifest_path = get_manifest_path(model_id, "llm_as_judge", "final_scores", split)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in scores:
            try:
                f.write(f"{json.dumps(item, ensure_ascii=False)}\n")
            except Exception as e:
                print(f"Error writing item {item}:", e)

    return manifest_path

def write_manifest(
    references: list,
    transcriptions: list,
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
    audio_length: list = None,
    transcription_time: list = None,
):
    """
    Writes a manifest file (jsonl format) and returns the path to the file.

    Args:
        references: Ground truth reference texts.
        transcriptions: Model predicted transcriptions.
        model_id: String identifier for the model.
        dataset_path: Path to the dataset.
        dataset_name: Name of the dataset.
        split: Dataset split name.
        audio_length: Length of each audio sample in seconds.
        transcription_time: Transcription time of each sample in seconds.

    Returns:
        Path to the manifest file.
    """
    
    if len(references) != len(transcriptions):
        raise ValueError(
            f"The number of samples in `references` ({len(references)}) "
            f"must match `transcriptions` ({len(transcriptions)})."
        )

    if audio_length is not None and len(audio_length) != len(references):
        raise ValueError(
            f"The number of samples in `audio_length` ({len(audio_length)}) "
            f"must match `references` ({len(references)})."
        )
    if transcription_time is not None and len(transcription_time) != len(references):
        raise ValueError(
            f"The number of samples in `transcription_time` ({len(transcription_time)}) "
            f"must match `references` ({len(references)})."
        )

    audio_length = (
        audio_length if audio_length is not None else len(references) * [None]
    )
    transcription_time = (
        transcription_time
        if transcription_time is not None
        else len(references) * [None]
    )

    manifest_path = get_manifest_path(model_id, dataset_path, dataset_name, split)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx, (text, transcript, audio_length, transcription_time) in enumerate(
            zip(references, transcriptions, audio_length, transcription_time)
        ):
            datum = {
                "audio_filepath": f"sample_{idx}",  # dummy value for Speech Data Processor
                "duration": audio_length,
                "time": transcription_time,
                "text": text,
                "pred_text": transcript,
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")
    return manifest_path



def load_files (directory: str, model_id: str = None):
    """
    Scores all result files in a directory and returns a composite score over all evaluated datasets.

    Args:
        directory: Path to the result directory, containing one or more jsonl files.
        model_id: Optional, model name to filter out result files based on model name.

    Returns:
        Composite score over all evaluated datasets and a dictionary of all results.
    """
     # Strip trailing slash
    if directory.endswith(os.pathsep):
        directory = directory[:-1]

    # Find all result files in the directory
    result_files = list(glob.glob(f"{directory}/**/*.jsonl", recursive=True))
    result_files = list(sorted(result_files))

    # Filter files belonging to a specific model id
    if model_id is not None and model_id != "":
        print("Filtering models by id:", model_id)
        model_id = model_id.replace("/", "-")
        result_files = [fp for fp in result_files if model_id in fp]

    # Check if any result files were found
    if len(result_files) == 0:
        raise ValueError(f"No result files found in {directory}")
    
    return result_files



def parse_filepath(fp: str):
    # Utility function to parse the file path and extract model id, dataset path, dataset name and split
    model_index = fp.find("MODEL_")
    fp = fp[model_index:]
    ds_index = fp.find("DATASET_")
    model_id = fp[:ds_index].replace("MODEL_", "").rstrip("_")
    author_index = model_id.find("-")
    model_id = model_id[:author_index] + "/" + model_id[author_index + 1 :]

    ds_fp = fp[ds_index:]
    dataset_id = ds_fp.replace("DATASET_", "").rstrip(".jsonl")
    return model_id, dataset_id



def sentiment_score_results(directory: str, model_id: str = None):
    # Compute WER results per dataset, and RTFx over all datasets
    result_files = load_files(directory, model_id)
    results = {}
    accuracy_metric = evaluate.load("accuracy")
    
    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_id = parse_filepath(result_file)

        references = [datum["references"] for datum in manifest]
        predictions = [datum["pred_text"] for datum in manifest]

        label2id = {"negative": 0, "neutral": 1, "positive": 2}
        pred_ids = [label2id[pred] for pred in predictions]
        ref_ids  = [label2id[ref] for ref in references]

        accuracy = accuracy_metric.compute(predictions=pred_ids, references=ref_ids)
        

        result_key = f"{model_id_of_file} | {dataset_id}"
        results[result_key] = accuracy

    print("*" * 80)
    print("Results per dataset:")
    print("*" * 80)

    for k, v in results.items():
        metrics = f"{k}: accuracy = {v['accuracy']*100:0.2f} %"
        print(metrics)
    return metrics


def llm_as_judge_score_results(directory: str, model_id: str, model_name: str, wandb_entity: str = None, wandb_project: str = None):
    if wandb_entity and wandb_project:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
        )
        wandb.run.name = f"{model_id}_LLM_As_Judge_test"
    result_files = load_files(directory, model_id)
    results = {}
    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_id = parse_filepath(result_file)

        questions = [datum["question"] for datum in manifest]
        predictions = [datum["prediction"] for datum in manifest]


        all_scores = []
        all_scores = run_llm_as_judge(model_name, questions, predictions )
        #print(all_scores)
        _ = write_llm_scores_manifest( scores=all_scores, model_id=model_id_of_file, dataset_path=dataset_id, dataset_name="llm_judge", split="test")
        
        
        #all_scores = [float(s) for s in all_scores if s is not None]
        all_scores = [float(item["score"]) for item in all_scores if item["score"] is not None]

        average_llm_score = sum(all_scores) / len(all_scores) if all_scores else 0
        

        result_key = f"{model_id_of_file} | {dataset_id}"
        results[result_key] = {"average_llm_as_judge_score": average_llm_score}

    print("*" * 80)
    print("Results per dataset:")
    print("*" * 80)

    for k, v in results.items():
        #metrics = f"{k}: avg_llm_as_judge_score = {v['average_llm_as_judge_score']:0.2f}"
        metrics = {f"{k}/average_llm_as_judge_score": v["average_llm_as_judge_score"]}
        if wandb_entity and wandb_project:
            wandb.log(metrics)
        print(metrics)

    if wandb_entity and wandb_project:
        wandb.finish()

        
    return metrics


def score_results(directory: str, model_id: str = None):
    """
    Scores all result files in a directory and returns a composite score over all evaluated datasets.

    Args:
        directory: Path to the result directory, containing one or more jsonl files.
        model_id: Optional, model name to filter out result files based on model name.

    Returns:
        Composite score over all evaluated datasets and a dictionary of all results.
    """

    # Compute WER results per dataset, and RTFx over all datasets
    result_files = load_files(directory, model_id)
    results = {}
    wer_metric = evaluate.load("wer")
    
    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_id = parse_filepath(result_file)

        references = [datum["text"] for datum in manifest]
        predictions = [datum["pred_text"] for datum in manifest]

        time = [datum["time"] for datum in manifest]
        duration = [datum["duration"] for datum in manifest]
        compute_rtfx = all(time) and all(duration)

        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)

        if compute_rtfx:
            audio_length = sum(duration)
            inference_time = sum(time)
            rtfx = round(sum(duration) / sum(time), 4)
        else:
            audio_length = inference_time = rtfx = None

        result_key = f"{model_id_of_file} | {dataset_id}"
        results[result_key] = {"wer": wer, "audio_length": audio_length, "inference_time": inference_time, "rtfx": rtfx}

    print("*" * 80)
    print("Results per dataset:")
    print("*" * 80)

    for k, v in results.items():
        metrics = f"{k}: WER = {v['wer']:0.2f} %"
        if v["rtfx"] is not None:
            metrics += f", RTFx = {v['rtfx']:0.2f}"
        print(metrics)

    # composite WER should be computed over all datasets and with the same key
    composite_wer = defaultdict(float)
    composite_audio_length = defaultdict(float)
    composite_inference_time = defaultdict(float)
    count_entries = defaultdict(int)
    for k, v in results.items():
        key = k.split("|")[0].strip()
        composite_wer[key] += v["wer"]
        if v["rtfx"] is not None:
            composite_audio_length[key] += v["audio_length"]
            composite_inference_time[key] += v["inference_time"]
        else:
            composite_audio_length[key] = composite_inference_time[key] = None
        count_entries[key] += 1

    # normalize scores & print
    print()
    print("*" * 80)
    print("Composite Results:")
    print("*" * 80)
    for k, v in composite_wer.items():
        wer = v / count_entries[k]
        print(f"{k}: WER = {wer:0.2f} %")
    for k in composite_audio_length:
        if composite_audio_length[k] is not None:
            rtfx = composite_audio_length[k] / composite_inference_time[k]
            print(f"{k}: RTFx = {rtfx:0.2f}")
    print("*" * 80)
    return composite_wer, results
