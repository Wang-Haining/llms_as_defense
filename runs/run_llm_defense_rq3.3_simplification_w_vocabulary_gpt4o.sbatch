#!/bin/sh
#SBATCH --job-name=run_llm_defense_rq3.3_simplification_w_vocabulary_gpt4o
#SBATCH --account=group-jasonclark
#SBATCH --partition=unsafe
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL


module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
source ~/.bashrc
. .venv/bin/activate

MODEL="gpt-4o-2024-08-06"
RQ="rq3.3_simplification_w_vocabulary"
PROVIDER="openai"
SLEEP_TIME=30
CORPORA=("ebg" "rj")

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
