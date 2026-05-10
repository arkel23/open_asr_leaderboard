#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=(
    "mistralai/Voxtral-Mini-3B-2507"
)
BATCH_SIZE=3

JUDGE_LLM_MODEL="cohere"

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval_llm.py \
        --model_id=${MODEL_ID} \
        --dataset_path="TwinkStart/speech-triavia-qa" \
        --split="test" \
        --device=-1 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=12 \

    # Evaluate results
    RUNDIR=`pwd` && \
    python -c "from normalizer import eval_utils; eval_utils.llm_as_judge_score_results('${RUNDIR}/results', '${MODEL_ID}', '${JUDGE_LLM_MODEL}', wandb_entity='LisTAya', wandb_project='voxtral_${JUDGE_LLM_MODEL}_model_as_judge')" && \
    cd $RUNDIR

done
