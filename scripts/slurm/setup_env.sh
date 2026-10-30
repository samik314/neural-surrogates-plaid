#!/bin/bash
#SBATCH -J plaid_setup_env
#SBATCH --time=00:45:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Create the environment from requirements.lock (CPU job, ~11 min, ~8 GB). From the repo root:
#   mkdir -p logs && PLAID_ENV=$HOME/.conda/envs/plaid_env sbatch scripts/slurm/setup_env.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# module load miniforge3                        # edit for your cluster
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV="${PLAID_ENV:-plaid_env}"
if [[ "${ENV}" == */* ]]; then mamba create -y -p "${ENV}" python=3.12 pip; else mamba create -y -n "${ENV}" python=3.12 pip; fi
conda activate "${ENV}"
pip install --no-cache-dir -r requirements.lock
pip install --no-cache-dir -e . --no-deps
pip check
pytest -q -p no:cacheprovider
echo "done $(date)"
