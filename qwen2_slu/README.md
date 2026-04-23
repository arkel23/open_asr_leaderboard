# Qwen 2 Audio SLU Eval


**CLI**

```
python run_eval_ml_logging_version.py \
--model_id Qwen/Qwen2-Audio-7B \
--dataset_path WueNLP/belebele-fleurs \
--dataset eng_Latn \
--split test \
--batch_size 6 \
--max_new_tokens 10 \
--max_eval_samples 20 \
--max_audio_seconds 30 \
--warmup_steps 2 \
--quant_config bnb \
--quant_dtype_weights 8 \
--device 0 \
--debug

```

**CLI  W&B Logging**
```
python run_eval_ml_logging_version.py \
--model_id Qwen/Qwen2-Audio-7B \
--dataset_path WueNLP/belebele-fleurs \
--dataset eng_Latn \
--split test \
--batch_size 6 \
--max_new_tokens 10 \
--max_eval_samples 20 \
--max_audio_seconds 30 \
--warmup_steps 2 \
--quant_config bnb \
--quant_dtype_weights 8 \
--device 0 \
--debug \
--wandb_entity <entity name> \
--wandb_project <project name>

```