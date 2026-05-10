#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=(
    "mistralai/Voxtral-Mini-3B-2507"
)
BATCH_SIZE=3

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval_sentiment.py \
        --model_id=${MODEL_ID} \
        --dataset_path="TwinkStart/MELD" \
        --split="test" \
        --device=-1 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=12 \
        --wandb_entity="LisTAya" \
        --wandb_project="voxtral_sentiment" \

    # Evaluate results
    RUNDIR=`pwd` && \
    cd .. && \
    python -c "from normalizer import eval_utils; eval_utils.sentiment_score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
