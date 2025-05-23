#!/bin/sh
#SBATCH --job-name=run_llm_defense_rq1.1_basic_paraphrase_ministral_8b_2410
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate

MODEL="mistralai/Ministral-8B-Instruct-2410"
RQ="rq1.1_basic_paraphrase"
PROVIDER="local"
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
