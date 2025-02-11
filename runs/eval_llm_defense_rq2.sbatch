#!/bin/sh
#SBATCH --job-name=run_llm_defense_rq2
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

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
source ~/.bashrc
. .venv/bin/activate

# define research questions
RQS=(
    "rq2.1_obfuscation"
    "rq2.2_imitation"
    "rq2.2_imitation_w_exemplar"
    "rq2.3_simplification"
)
SLEEP_TIME=30

# loop through RQs
for rq in "${RQS[@]}"; do
    echo "Running: rq=$rq"

    CUDA_VISIBLE_DEVICES=0 python -m eval_llm_defense --rq "$rq"

    # sleep between runs except for last iteration
    if [ "$rq" != "${RQS[-1]}" ]; then
        echo "Sleeping for $SLEEP_TIME seconds..."
        sleep $SLEEP_TIME
    fi
done
