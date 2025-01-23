#!/bin/sh
#SBATCH --job-name=run_llm_defense_rerun_ebg_rq1.2_backtranslation_via_zh
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
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
--rq rq1.2_backtranslation_via_zh \
--model "meta-llama/Llama-3.1-8B-Instruct" \
--provider local
