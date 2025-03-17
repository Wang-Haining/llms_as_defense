#!/bin/sh
#SBATCH --job-name=estimate_llm_intervention_rq3
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

# define research questions (rq3)
RQS=("rq3.1_obfuscation_w_persona" "rq3.2_imitation_w_500words" "rq3.2_imitation_w_1000words" "rq3.2_imitation_w_2500words" "rq3.3_simplification_w_vocabulary" "rq3.3_simplification_w_exemplar" "rq3.3_simplification_w_vocabulary_and_exemplar")
SLEEP_TIME=30

# loop through corpora and research questions
for rq in "${RQS[@]}"; do
    echo "  running: rqs=$rq"
    python -m estimate_llm_intervention --rqs "$rq"
    # sleep between runs except for last iteration of research questions for this corpus
    if [ "$rq" != "${RQS[-1]}" ]; then
        echo "  sleeping for $SLEEP_TIME seconds..."
        sleep $SLEEP_TIME
    fi
done
