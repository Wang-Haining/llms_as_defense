#!/bin/sh
#SBATCH --job-name=test_run_llm_defense_rq3.1
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL


module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus ebg \
--rq rq3.1_persona_playing \
--model "google/gemma-2-9b-it" \
--provider local \
--temperature 0.7 \
--max_tokens 4096 \
--num_seeds 2 \
--debug

sleep 30

CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus rj \
--rq rq3.1_persona_playing \
--model "meta-llama/Llama-3.1-8B-Instruct" \
--provider local \
--temperature 0.7 \
--max_tokens 4096 \
--num_seeds 1 \
--debug

sleep 30

CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus lcmc \
--rq rq3.1_persona_playing \
--model "mistralai/Ministral-8B-Instruct-2410" \
--provider local \
--temperature 0.7 \
--max_tokens 4096 \
--num_seeds 1 \
--debug

sleep 30

CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus lcmc \
--rq rq3.1_persona_playing \
--model "claude-3-5-sonnet-20241022" \
--provider anthropic \
--temperature 0.7 \
--max_tokens 4096 \
--num_seeds 1 \
--debug

sleep 30

CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus rj \
--rq rq3.1_persona_playing \
--model "gpt-4o-2024-08-06" \
--provider openai \
--temperature 0.7 \
--max_tokens 4096 \
--num_seeds 1 \
--debug
