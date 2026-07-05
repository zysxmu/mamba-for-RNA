# RNA-Mamba

RNA-Mamba is a memory-augmented bidirectional Mamba model for RNA masked
language modeling. It uses the Caduceus BiMamba backbone and adds two
RNA-specific components:

1. **Bidirectional Consistent Writing (BCW)** aligns forward and backward
   states at the same RNA position and learns a feature-wise directional
   fusion.
2. **Cross-layer multi-slot memory** compresses the fused states into global
   and regional slots that later layers read through gated cross-attention.

The implementation is intended for reproducible pretraining, ablation studies,
and multi-GPU Slurm runs.

## Architecture

```text
RNA tokens [B, L]
  -> BiMamba layer
     -> aligned forward states  [B, L, d_model]
     -> aligned backward states [B, L, d_model]
  -> read memory written by earlier layers
  -> BCW directional fusion     [B, L, d_sum]
  -> global + regional pooling  [B, S, d_mem]
  -> FIFO cross-layer bank      [B, M, d_mem]
  -> later token-to-memory cross-attention
  -> same-position MLM head
```

Each layer reads before it writes, so a layer cannot read its own newly created
slots. Memory exists only inside one forward pass and is never shared between
independent RNA samples or batches. Padding is excluded from fusion statistics,
slot pooling, and attention.

## Default pretraining configuration

The formal configuration is
[`configs/experiment/rna_pretrain.yaml`](configs/experiment/rna_pretrain.yaml):

- 12 BiMamba layers, `d_model=768`;
- sequence length 1024;
- A/U/C/G/N character tokenizer;
- 15% same-position MLM;
- BCW with `d_sum=256`;
- one global and eight valid-length-aware regional slots per write layer;
- weighted pooling and `d_mem=128`;
- write stride 4, read stride 2, and a 64-slot FIFO bank;
- four-head shared memory reader with an independent gate per read layer;
- BF16, AdamW, gradient clipping, and cosine decay with warmup.

The reader gates start conservatively at `-4.0` before the sigmoid. This keeps
the pretrained backbone path dominant at initialization while still allowing
memory gradients to flow.

## Dataset

Provide one TXT corpus and one FASTA corpus:

```text
data/
├── data-random_15K_sequences.txt
└── rnacentral_small_ATCG_only.fasta
```

- TXT: one sequence per line; when commas are present, only the first field is
  used.
- FASTA: standard multiline records are supported.
- Input is uppercased and `T` is normalized to `U`.
- The current mixed-RNA corpus loader retains canonical `A/U/C/G` sequences;
  the tokenizer also contains `N` for masking and compatible inputs.
- The split is deterministic: 80% train, 10% validation, and 10% test.
- Training masking is dynamic; validation and test masking are deterministic.

Data files are excluded from Git. On a cluster, use absolute paths visible from
every compute node.

## Installation

CUDA training requires Linux, a Linux cluster, or WSL2.

- Python 3.10
- PyTorch 2.2.0 with CUDA 12.1
- `causal-conv1d==1.2.0.post2`
- `mamba-ssm==1.2.2`

```bash
bash setup_linux_env.sh
source .venv/bin/activate
```

`requirements-core.txt` contains the dependencies used by the RNA pretraining
path. `requirements.txt` retains the original full environment export.

## Local smoke test

```bash
source .venv/bin/activate
bash run_local_smoke.sh
```

This runs a small end-to-end GPU job with training, validation, test,
checkpoint saving, and checkpoint reload. The default output directory is
`outputs/local-smoke/`.

Useful overrides:

```bash
PRECISION=bf16 MAX_STEPS=20 RUN_TEST=false \
RUN_DIR=outputs/local-smoke-20 bash run_local_smoke.sh
```

## Ablations

All variants use the same training path and can be selected with Hydra
overrides.

```bash
# BiMamba baseline
python -m train experiment=rna_pretrain model.config.use_memory=false

# Single-direction memory writer
python -m train experiment=rna_pretrain \
  model.config.memory_writer_mode=single

# Multi-slot memory without learned bidirectional writing
python -m train experiment=rna_pretrain \
  model.config.memory_writer_mode=average

# Full BCW + multi-slot memory (formal default)
python -m train experiment=rna_pretrain \
  model.config.memory_writer_mode=bcw

# One global slot only
python -m train experiment=rna_pretrain \
  model.config.memory_num_global_slots=1 \
  model.config.memory_num_local_slots=0

# Mean pooling without a learned write score
python -m train experiment=rna_pretrain \
  model.config.memory_pooling=mean \
  model.config.memory_use_write_score=false
```

`scalar_gate`, `max` pooling, source-embedding switches, reader sharing, and
read/write strides are also configurable in
[`configs/model/caduceus.yaml`](configs/model/caduceus.yaml).

BCW is the writer used by the memory path; a nominal "BCW-only" setting with no
memory consumer would not affect the MLM loss and is therefore not presented
as a valid ablation.

## Slurm training

The provided script launches one Slurm task per GPU. PyTorch Lightning uses the
Slurm-provided global and local ranks to form the eight-process DDP job:

```bash
export RNA_TEXT_FILE=/shared/data/data-random_15K_sequences.txt
export RNA_FASTA_FILE=/shared/data/rnacentral_small_ATCG_only.fasta
export ENV_ACTIVATE='source ~/miniconda3/bin/activate rna-mamba'

export NUM_DEVICES=8
export PER_DEVICE_BATCH=16
export GLOBAL_BATCH=256
export MAX_STEPS=20000
export MAX_EPOCHS=null
export RUN_DIR=/shared/outputs/rna-mamba

sbatch slurm_scripts/run_pretrain_rna.sh
```

Run a short cluster smoke test first:

```bash
export MAX_STEPS=20
export RUN_DIR=/shared/outputs/rna-mamba-smoke
sbatch slurm_scripts/run_pretrain_rna.sh
```

Confirm that every GPU is active, validation completes, memory use remains
stable, and `checkpoints/last.ckpt` is written before submitting the full run.

The accumulation factor is:

```text
ceil(global_batch / (nodes * devices_per_node * per_device_batch))
```

## Checkpoints

Training produces:

```text
checkpoints_best/       best validation-loss checkpoint
checkpoints_periodic/   periodic recovery checkpoints
checkpoints/last.ckpt   latest complete training state
resolved_config.yaml    resolved Hydra configuration
run_metadata.json       command, Git state, seed, batch size, and parameter counts
runtime_metrics.json    step time, throughput, and peak allocated GPU memory
```

`train.ckpt` performs an exact Lightning resume, including model, optimizer,
scheduler, and global step. `train.pretrained_model_path` is a separate
compatible warm-start path: matching backbone tensors are loaded and a
structured missing/unexpected/shape-mismatch report is written for new memory
parameters.

With PyTorch Lightning 1.8, a checkpoint taken in the middle of a shuffled
epoch does not guarantee restoration of the exact dataloader cursor. Model and
optimizer state are restored, but a few samples may be repeated or skipped.
Use an epoch-boundary checkpoint when exact sample order is required.

## Tests

```bash
pytest -q \
  caduceus/tests/test_writer_only.py \
  caduceus/tests/test_memory_bank_reader.py \
  caduceus/tests/test_model_memory_smoke.py \
  caduceus/tests/test_memory_backward.py \
  caduceus/tests/test_memory_read_before_write.py \
  caduceus/tests/test_memory_stride_counts.py \
  caduceus/tests/test_memory_eval_isolation.py \
  caduceus/tests/test_mixed_rna_dataset.py \
  caduceus/tests/test_mlm_alignment.py
```

The suite checks alignment, padding invariance, empty slots, FIFO capacity,
read-before-write ordering, memory-specific gradients, forward isolation, and
same-position MLM labels.

## Current validation

The formal 50.3M-parameter model has completed forward and backward passes on
an RTX 4060 Laptop GPU with BF16, sequence length 1024, and batch size 8.
Observed peak allocated memory was approximately 4.26 GiB. A two-step
real-corpus smoke run completed training and validation without NaN, Inf, or
OOM. These measurements validate the execution path; they are not biological
benchmark results.

True multi-GPU DDP must still be verified with the 20-step smoke job on the
target cluster because the local machine exposes only one GPU.

## Project structure

```text
caduceus/memory/        BCW writer, multi-slot summarizer, bank, and reader
caduceus/               model configuration and BiMamba integration
configs/                Hydra model, data, pipeline, and experiment settings
src/dataloaders/        mixed-RNA loading and MLM collation
src/callbacks/          runtime and validation logging
slurm_scripts/          cluster submission scripts
tests/fixtures/         synthetic smoke-test data
train.py                training, resume, warm-start, and metadata entry point
```

## Lineage

The backbone is based on Caduceus:

```bibtex
@article{schiff2024caduceus,
  title={Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling},
  author={Schiff et al.},
  year={2024}
}
```

The cross-layer memory design is inspired by the memory-pattern perspective in
*MemMamba: Rethinking Memory Patterns in State Space Models* and is reworked
here for aligned bidirectional RNA states and multi-slot layer memory.
