#!/bin/sh
#SBATCH --job-name=estimate_llm_intervention_rq1.2_combined
#SBATCH --account=group-jasonclark
#SBATCH --partition=unsafe
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
source ~/.bashrc
. .venv/bin/activate

SLEEP_TIME=30

# process each corpus with sleep between runs
python -m estimate_llm_intervention \
    --rqs "rq1.2_backtranslation_via_de" "rq1.2_backtranslation_via_zh"

# sleep between runs except for last corpus
if [ "$corpus" != "${CORPORA[-1]}" ]; then
    echo "sleeping for $SLEEP_TIME seconds..."
    sleep $SLEEP_TIME
fi
