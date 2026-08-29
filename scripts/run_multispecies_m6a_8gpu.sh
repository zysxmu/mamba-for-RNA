#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the final RNA MLM checkpoint on the formal six-species full-transcript
# nucleotide-level m6A dataset. Run inside an allocated eight-GPU node.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${MULTISPECIES_M6A_DATA_DIR:-$ROOT_DIR/data/processed/multispecies_m6a_full_transcript}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/multispecies_m6a_full_transcript}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_DEVICES="${NUM_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
PRECISION="${PRECISION:-bf16}"
GLOBAL_BATCH=$((NUM_DEVICES * BATCH_SIZE * GRAD_ACCUM))

if [[ -z "$PRETRAIN_CKPT" ]]; then
  echo "Set PRETRAIN_CKPT to the best MLM val_loss.ckpt." >&2
  exit 2
fi
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "Missing PRETRAIN_CKPT: $PRETRAIN_CKPT" >&2
  exit 2
fi
if [[ "$GLOBAL_BATCH" -ne 16 ]]; then
  echo "Expected global batch 16, got $GLOBAL_BATCH." >&2
  echo "devices=$NUM_DEVICES batch_size=$BATCH_SIZE accumulation=$GRAD_ACCUM" >&2
  exit 2
fi
if [[ "$FINETUNE_EPOCHS" -le 0 ]]; then
  echo "FINETUNE_EPOCHS must be positive." >&2
  exit 2
fi

for required in train.jsonl.gz val.jsonl.gz test.jsonl.gz stats.json; do
  if [[ ! -f "$DATA_DIR/$required" ]]; then
    echo "Missing prepared m6A data file: $DATA_DIR/$required" >&2
    exit 2
  fi
done

mkdir -p \
  "$RUN_DIR/checkpoints_best" \
  "$RUN_DIR/checkpoints_periodic" \
  "$RUN_DIR/checkpoints"

export CUDA_VISIBLE_DEVICES
export MULTISPECIES_M6A_DATA_DIR="$DATA_DIR"
cd "$ROOT_DIR"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status_paths=$(git status --porcelain | wc -l)"
  echo "data_dir=$DATA_DIR"
  echo "pretrain_checkpoint=$PRETRAIN_CKPT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "num_devices=$NUM_DEVICES"
  echo "batch_size_per_gpu=$BATCH_SIZE"
  echo "gradient_accumulation=$GRAD_ACCUM"
  echo "global_batch=$GLOBAL_BATCH"
  echo "finetune_epochs=$FINETUNE_EPOCHS"
  echo "precision=$PRECISION"
  python --version
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  fi
} > "$RUN_DIR/run_manifest.txt"

echo "Starting six-species full-transcript m6A fine-tuning:"
echo "  data: $DATA_DIR"
echo "  initialization: $PRETRAIN_CKPT"
echo "  global batch: $GLOBAL_BATCH"
echo "  epochs: $FINETUNE_EPOCHS"
echo "  precision: $PRECISION"
echo "  output: $RUN_DIR"

/usr/bin/time -v -o "$RUN_DIR/time.txt" \
  python -m train \
    experiment=multispecies_m6a_full_transcript \
    trainer.devices="$NUM_DEVICES" \
    trainer.accelerator=gpu \
    trainer.precision="$PRECISION" \
    trainer.max_epochs="$FINETUNE_EPOCHS" \
    trainer.accumulate_grad_batches="$GRAD_ACCUM" \
    model.config.residual_in_fp32=true \
    optimizer.lr=2e-5 \
    train.pretrained_model_path="$PRETRAIN_CKPT" \
    train.pretrained_model_strict_load=false \
    train.pretrained_model_state_hook._name_=load_matching_backbone \
    train.ckpt=null \
    train.test=false \
    dataset.data_dir="$DATA_DIR" \
    dataset.max_sequence_length=10240 \
    dataset.batch_size="$BATCH_SIZE" \
    dataset.batch_size_eval=1 \
    loader.num_workers="$NUM_WORKERS" \
    callbacks.model_checkpoint.dirpath="$RUN_DIR/checkpoints_best" \
    callbacks.model_checkpoint.filename=val_m6a_ap \
    callbacks.periodic_checkpoint.dirpath="$RUN_DIR/checkpoints_periodic" \
    callbacks.periodic_checkpoint.every_n_epochs=1 \
    callbacks.model_checkpoint_every_n_steps.dirpath="$RUN_DIR/checkpoints" \
    callbacks.model_checkpoint_every_n_steps.save_top_k=0 \
    callbacks.model_checkpoint_every_n_steps.save_last=false \
    wandb=null \
    hydra.run.dir="$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/console.log"
