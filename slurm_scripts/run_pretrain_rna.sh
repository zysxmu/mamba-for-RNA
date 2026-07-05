#!/usr/bin/env bash
#SBATCH --job-name=rna_mamba
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

: "${RNA_TEXT_FILE:?Export RNA_TEXT_FILE as the absolute TXT dataset path}"
: "${RNA_FASTA_FILE:?Export RNA_FASTA_FILE as the absolute FASTA dataset path}"

NUM_DEVICES="${NUM_DEVICES:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_EPOCHS="${MAX_EPOCHS:-null}"
MAX_STEPS="${MAX_STEPS:-20000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-100}"
RUN_DIR="${RUN_DIR:-outputs/pretrain_rna/${SLURM_JOB_ID:-manual}}"

if [[ -z "${WARMUP_STEPS:-}" ]]; then
  WARMUP_STEPS=$(( MAX_STEPS / 5 ))
fi
if (( WARMUP_STEPS < 1 )); then
  WARMUP_STEPS=1
fi
if (( WARMUP_STEPS >= MAX_STEPS )); then
  WARMUP_STEPS=$(( MAX_STEPS - 1 ))
fi
DECAY_STEPS=$(( MAX_STEPS - WARMUP_STEPS ))

# Activate the environment before submitting, or export an activation command,
# for example: ENV_ACTIVATE='source ~/miniconda3/bin/activate caduceus_env'
if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  eval "${ENV_ACTIVATE}"
fi

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Lightning uses the Slurm-provided ranks: one task per GPU. NUM_DEVICES must
# match both --ntasks-per-node and --gres above.
srun --ntasks="${NUM_DEVICES}" --kill-on-bad-exit=1 python -m train \
  experiment=rna_pretrain \
  dataset.max_length="${MAX_LENGTH}" \
  dataset.max_length_val="${MAX_LENGTH}" \
  dataset.max_length_test="${MAX_LENGTH}" \
  dataset.batch_size="${PER_DEVICE_BATCH}" \
  dataset.batch_size_eval="${PER_DEVICE_BATCH}" \
  loader.num_workers="${NUM_WORKERS}" \
  train.global_batch_size="${GLOBAL_BATCH}" \
  train.ckpt=checkpoints/last.ckpt \
  trainer.devices="${NUM_DEVICES}" \
  trainer.num_nodes=1 \
  trainer.precision="${PRECISION}" \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  scheduler.t_initial="${DECAY_STEPS}" \
  scheduler.warmup_t="${WARMUP_STEPS}" \
  hydra.run.dir="${RUN_DIR}"
