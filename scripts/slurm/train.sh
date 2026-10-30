#!/bin/bash
#SBATCH -J plaid_train
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -p gpu --gres=gpu:1                    # edit for your cluster (partition, GPU type)
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# One training run, then its predictions. From the repo root (mkdir -p logs first):
#   CONFIG=configs/2dmultischypel_transolver_geoligmls256.yaml SEED=0 sbatch scripts/slurm/train.sh
#   CONFIG=... SEED=0 FOLD=2 sbatch scripts/slurm/train.sh        # 5-fold CV run on fold 2
#   CONFIG=... SEED=1 FULL=1 sbatch scripts/slurm/train.sh        # final retrain on all training data
# Dev and CV runs: dev round trip (predict --split dev must reproduce the training-time score) and
# test predictions from the best checkpoint. Full retrains: test predictions from the last
# checkpoint (they have no validation set). ~60 min for the final 2D config on an RTX A5000.
source scripts/slurm/_common.sh
: "${CONFIG:?set CONFIG=configs/<name>.yaml}" "${SEED:=0}"
ARGS=(--config "${CONFIG}" --seed "${SEED}")
[ -n "${FOLD:-}" ] && ARGS+=(--fold "${FOLD}")
[ -n "${FULL:-}" ] && ARGS+=(--full)
python -u -m plaid_surrogates.train "${ARGS[@]}"
RUNS="${PLAID_RUNS_DIR:-runs}"
TAG=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['tag'])" "${CONFIG}")
SUFFIX="_s${SEED}"; [ -n "${FOLD:-}" ] && SUFFIX="${SUFFIX}_f${FOLD}"; [ -n "${FULL:-}" ] && SUFFIX="${SUFFIX}_full"
RUN_DIR=$(ls -td "${RUNS}"/*_transolver_"${TAG}${SUFFIX}" | head -1)
echo "run folder: ${RUN_DIR}"
if [ -n "${FULL:-}" ]; then
    python -u -m plaid_surrogates.predict --run "${RUN_DIR}" --split test --checkpoint last
else
    python -u -m plaid_surrogates.predict --run "${RUN_DIR}" --split dev --checkpoint best
    python -u -m plaid_surrogates.predict --run "${RUN_DIR}" --split test --checkpoint best
fi
echo "done $(date)"
