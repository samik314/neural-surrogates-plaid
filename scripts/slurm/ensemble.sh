#!/bin/bash
#SBATCH -J plaid_ensemble
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# Average the test predictions of several finished runs into one submission (CPU, a few minutes).
# From the repo root, after the three full retrains have finished:
#   RUNS="runs/*_geoligmls256_s[012]_full" CHECKPOINT=last sbatch scripts/slurm/ensemble.sh
source scripts/slurm/_common.sh
: "${RUNS:?set RUNS to the member run folders (a glob is fine)}" "${CHECKPOINT:=last}" "${SPLIT:=test}"
# shellcheck disable=SC2086  # RUNS is meant to expand
python -u -m plaid_surrogates.ensemble --runs ${RUNS} --split "${SPLIT}" --checkpoint "${CHECKPOINT}"
echo "done $(date)"
