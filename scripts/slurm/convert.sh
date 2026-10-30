#!/bin/bash
#SBATCH -J plaid_convert
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Download a dataset from Hugging Face and convert it to the local cache, then compare the cache
# with PLAID (integration tests). CPU job. From the repo root:
#   mkdir -p logs && DATASET=2D_MultiScHypEl sbatch scripts/slurm/convert.sh     # or Tensile2d
source scripts/slurm/_common.sh
: "${DATASET:?set DATASET=2D_MultiScHypEl or Tensile2d}"
python -u -m plaid_surrogates.data.convert --dataset "${DATASET}" --processes "${SLURM_CPUS_PER_TASK:-4}"
pytest -q -p no:cacheprovider tests/integration
echo "done $(date)"
