#!/usr/bin/env bash
#SBATCH --job-name=fig5
#SBATCH --partition=gpu            # <-- set to your GPU partition name
#SBATCH --gres=gpu:1               # requests 1 GPU (see note in README; PyMOL
                                   #     ray tracing is CPU-only, GPU optional)
#SBATCH --cpus-per-task=8          # ray tracing scales with CPU cores
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=fig5_%j.out
#SBATCH --error=fig5_%j.err
# #SBATCH --account=your_account   # <-- uncomment/set if your cluster requires it

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# --- environment -----------------------------------------------------
# Option A: conda/mamba (recommended)
#   (create once:  mamba env create -f environment.yml)
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fig5

# xvfb (virtual display) if PyMOL needs a GL context on the compute node
module load xvfb 2>/dev/null || true

echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  gpu=${CUDA_VISIBLE_DEVICES:-none}"

# --- run -------------------------------------------------------------
bash run_fig5.sh

echo "Figure 5 build complete on $(hostname)."
