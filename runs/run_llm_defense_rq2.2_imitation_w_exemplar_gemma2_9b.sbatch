#!/bin/sh
#SBATCH --job-name=run_llm_defense_rq2.2_imitation_w_exemplar_gemma2_9b
#SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL


MODEL="google/gemma-2-9b-it"
RQ="rq2.2_imitation_w_exemplar"
PROVIDER="local"
SLEEP_TIME=30
CORPORA=("ebg" "rj" "lcmc")

for corpus in "${CORPORA[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
        --corpus "$corpus" \
        --rq "$RQ" \
        --model "$MODEL" \
        --provider "$PROVIDER"

    if [ "$corpus" != "${CORPORA[-1]}" ]; then
        sleep $SLEEP_TIME
    fi
done
