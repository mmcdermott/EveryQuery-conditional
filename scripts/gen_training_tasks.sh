#!/bin/bash
#SBATCH --job-name=eq-gen-train
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen_train_%j.out
#SBATCH --mail-user=gbk2114@cumc.columbia.edu
#SBATCH --mail-type=END,FAIL
#
# Runs the full training-tasks sweep sequentially in one job:
#   input_shard=0..291 x task_shard=0..15 = 4672 workers
#
# Usage:
#   sbatch scripts/gen_training_tasks.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

set -a
source .env
set +a

mkdir -p logs

uv run EQ_generate_training_tasks -m \
    split=train \
    input_shard='range(0,292)' \
    task_shard='range(0,16)'
