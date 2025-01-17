#!/bin/sh
#SBATCH --job-name=train_ml
##SBATCH --account=group-jasonclark
#SBATCH --partition=unsafe
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL


module load Python/3.10.8-GCCcore-12.2.0

. .venv/bin/activate

python -m train_ml --model logreg
python -m train_ml --model svm