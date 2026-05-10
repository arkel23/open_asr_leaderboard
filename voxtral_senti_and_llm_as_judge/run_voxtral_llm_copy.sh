#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=(
    "mistralai/Voxtral-Mini-3B-2507"
)
JUDGE_LLM_MODEL="cohere"
# Evaluate results
RUNDIR=`pwd` && \
python -c "from normalizer import eval_utils; eval_utils.llm_as_judge_score_results('${RUNDIR}/results', '${MODEL_ID}', '${JUDGE_LLM_MODEL}', wandb_entity='LisTAya', wandb_project='voxtral_${JUDGE_LLM_MODEL}_model_as_judge')" && \
cd $RUNDIR

