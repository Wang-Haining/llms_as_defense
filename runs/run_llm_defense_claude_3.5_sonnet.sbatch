#!/bin/sh
#SBATCH --job-name=run_llm_defense_claude_3.5_sonnet
##SBATCH --account=group-jasonclark
#SBATCH --partition=unsafe
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL


module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
source ~/.bashrc
. .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus ebg \
--rq rq1.1_basic_paraphrase \
--model "claude-3-5-sonnet-20241022" \
--provider anthropic \
--temperature 0.7 \
--max_tokens 2048 \
--num_seeds 2 \
--debug
sleep 30
CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus rj \
--rq rq1.1_basic_paraphrase \
--model "claude-3-5-sonnet-20241022" \
--provider anthropic \
--temperature 0.7 \
--max_tokens 2048 \
--num_seeds 1 \
--debug
sleep 30
CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus lcmc \
--rq rq1.1_basic_paraphrase \
--model "claude-3-5-sonnet-20241022" \
--provider anthropic \
--temperature 0.7 \
--max_tokens 2048 \
--num_seeds 1 \
--debug

