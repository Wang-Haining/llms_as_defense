#!/bin/sh
#SBATCH --job-name=estimate_llm_intervention_rq1
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

# define research questions (rq1)
RQS=("rq1.1_basic_paraphrase" "rq1.2_backtranslation_via_de" "rq1.2_backtranslation_via_zh")
CORPORA=("rj" "ebg")
SLEEP_TIME=30

# loop through corpora and research questions
for corpus in "${CORPORA[@]}"; do
    echo "running for corpus: $corpus"
    for rq in "${RQS[@]}"; do
        echo "  running: rqs=$rq"
        python -m estimate_llm_intervention --rqs "$rq" --corpus "$corpus"
        # sleep between runs except for last iteration of research questions for this corpus
        if [ "$rq" != "${RQS[-1]}" ]; then
            echo "  sleeping for $SLEEP_TIME seconds..."
            sleep $SLEEP_TIME
        fi
    done
done
