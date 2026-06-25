#!/usr/bin/env bash
#SBATCH --job-name=fmint_sde_repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=tutorials/fmint_sde_reproduction/logs/slurm-%A_%a.out
#SBATCH --account=hzyang-prj-aac

set -euo pipefail

# Example submissions:
#   sbatch tutorials/fmint_sde_reproduction/slurm_fmint_sde.sbatch exp1 all both
#   sbatch --gres=gpu:v100:1 tutorials/fmint_sde_reproduction/slurm_fmint_sde.sbatch exp1 all both
#   sbatch --array=1-7 tutorials/fmint_sde_reproduction/slurm_fmint_sde.sbatch
#
# The default GPU request is gpu:a100:1. Change the #SBATCH --gres line above,
# or override it at submit time with a GPU type available on your HPCC account.
#
# If submitted as an array with no first argument, array task 1 maps to exp1,
# task 2 maps to exp2, and so on.

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR/tutorials/fmint_sde_reproduction"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/reproduce_fmint_sde.sh" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
EXPERIMENT="${1:-}"
STAGE="${2:-all}"
MODE="${3:-both}"

if [[ -z "$EXPERIMENT" ]]; then
  TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
  EXPERIMENT="exp${TASK_ID}"
fi

if [[ -f "$SCRIPT_DIR/fmint_sde_reproduction.env" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/fmint_sde_reproduction.env"
fi

export GPU="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

bash "$SCRIPT_DIR/reproduce_fmint_sde.sh" \
  --experiment "$EXPERIMENT" \
  --stage "$STAGE" \
  --mode "$MODE"
