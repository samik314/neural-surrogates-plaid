# Sourced by every job script in scripts/slurm/. Adapt the environment activation to your cluster.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"                     # submit from the repository root
if [ -f .env ]; then set -a; source .env; set +a; fi   # optional PLAID_DATA_DIR, PLAID_RUNS_DIR, HF_HOME
# --- environment: edit for your cluster (module names differ) --------------------------------
# module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${PLAID_ENV:-plaid_env}"
# ----------------------------------------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  commit $(git rev-parse --short HEAD 2>/dev/null)  start $(date)"
