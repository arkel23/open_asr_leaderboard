#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
#MODEL_ID="openai/whisper-large"
MODEL_ID="openai/whisper-small"

#BATCH_SIZE=64
BATCH_SIZE=4
DEVICE_ID=0
DATASETS="McGill-NLP/african_celtic_dataset"

CONFIG_NAME="default"
SPLIT="dev"
LANGUAGE="english"

echo ""
echo "Running evaluation on $DATASETS"
echo "   Model: $MODEL_ID"
echo "   Dataset repo: $DATASETS"
echo "   Config: $CONFIG_NAME"
echo "   Split: $SPLIT"
echo "----------------------------------------"

python run_eval_ml.py \
    --model_id="$MODEL_ID" \
    --dataset="$DATASETS" \
    --config_name="$CONFIG_NAME" \
    --split="$SPLIT" \
    --language="$LANGUAGE" \
    --device="$DEVICE_ID" \
    --batch_size="$BATCH_SIZE" \
    --max_eval_samples=500 \
    --warmup_steps=3

echo ""
echo "========================================================"
echo "Evaluating results for $MODEL_ID"
echo "========================================================"

RUNDIR=`pwd`
cd ../normalizer
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
cd "$RUNDIR"