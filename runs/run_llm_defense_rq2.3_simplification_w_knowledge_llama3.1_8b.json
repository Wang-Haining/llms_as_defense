#!/bin/sh
#SBATCH --job-name=run_llm_defense_rq2.3_simplification_w_knowledge_gemma2_9b
##SBATCH --account=group-jasonclark
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

CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus ebg \
--rq rq2.3_simplification_w_knowledge \
--model "meta-llama/Llama-3.1-8B-Instruct" \
--provider local \
--temperature 0.7 \
--max_tokens 2048 \
--num_seeds 5

sleep 30

CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus rj \
--rq rq2.3_simplification_w_knowledge \
--model "meta-llama/Llama-3.1-8B-Instruct" \
--provider local \
--temperature 0.7 \
--max_tokens 2048 \
--num_seeds 5

sleep 30

CUDA_VISIBLE_DEVICES=0 python -m run_llm_defense \
--corpus lcmc \
--rq rq2.3_simplification_w_knowledge \
--model "meta-llama/Llama-3.1-8B-Instruct" \
--provider local \
--temperature 0.7 \
--max_tokens 2048 \
--num_seeds 5
