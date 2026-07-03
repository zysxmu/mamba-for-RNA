# Caduceus-RNA

Memory-augmented bidirectional Mamba for RNA masked language modeling (MLM).
The project pretrains on a mixed corpus of coding RNA in TXT format and
non-coding RNA in FASTA format.

## Status

The complete training pipeline has been validated on an RTX 4060 Laptop GPU,
including:

- RNA loading, normalization, and deterministic data splitting;
- dynamic MLM masking;
- bidirectional Mamba and memory cross-attention;
- forward and backward passes with optimizer updates;
- validation, testing, checkpoint saving, and checkpoint restoration.

A one-epoch reference run on the current local corpus used `d_model=768`,
12 layers, sequence length 1024, and a global batch size of 256:

| Metric | Result |
| --- | ---: |
| Training loss | 1.690 |
| Validation loss | 1.280 |
| Validation perplexity | 3.610 |

These values confirm that the implementation trains and the loss decreases.
They are not intended as final biological benchmarks.

## Requirements

CUDA training requires Linux, either natively, on a Linux cluster, or through
WSL2.

- Python 3.10
- PyTorch 2.2.0 with CUDA 12.1
- `causal-conv1d==1.2.0.post2`
- `mamba-ssm==1.2.2`
- An NVIDIA GPU

Create the recommended environment with:

```bash
bash setup_linux_env.sh
source .venv/bin/activate
```

`requirements-core.txt` contains the minimal dependency set used by the current
RNA training path. `requirements.txt` is the original full environment export
and is retained for reference. The CUDA version reported by the NVIDIA driver
may be newer than the CUDA 12.1 runtime bundled with PyTorch.

## Dataset

The training corpus consists of:

```text
data/
├── data-random_15K_sequences.txt
└── rnacentral_small_ATCG_only.fasta
```

- TXT input: each line starts with an RNA sequence. If a line contains commas,
  only the first field is used.
- FASTA input: standard multiline FASTA records are supported.
- Normalization: sequences are uppercased and `T` is converted to `U`. Records
  containing characters outside `A/U/C/G` are discarded.
- Split: a fixed seed produces 80% training, 10% validation, and 10% test data.
- Masking: training masks are generated dynamically. Validation and test masks
  are deterministic so checkpoints can be compared consistently.

Data files are excluded by `.gitignore` and are not stored in this repository.
Use absolute paths on storage that is accessible from every compute node.

## Quick smoke test

The repository includes a small synthetic corpus for testing the complete GPU
pipeline:

```bash
source .venv/bin/activate
bash run_local_smoke.sh
```

The smoke test runs three optimizer steps and covers training, validation,
testing, and checkpointing. Outputs are written to `outputs/local-smoke/`.
This is a pipeline check, not a biological experiment.

Run the core test suite with:

```bash
pytest -q \
  caduceus/tests/test_writer_only.py \
  caduceus/tests/test_model_memory_smoke.py \
  caduceus/tests/test_memory_backward.py \
  caduceus/tests/test_memory_read_before_write.py \
  caduceus/tests/test_memory_stride_counts.py \
  caduceus/tests/test_memory_eval_isolation.py \
  caduceus/tests/test_mixed_rna_dataset.py
```

## Training on a Slurm cluster

`slurm_scripts/run_pretrain_rna.sh` is the prepared entry point for a
single-node, eight-GPU Slurm job. Cluster-specific partition names, GPU resource
names, and environment modules may require changes to the `#SBATCH` header.

```bash
export RNA_TEXT_FILE=/absolute/path/data-random_15K_sequences.txt
export RNA_FASTA_FILE=/absolute/path/rnacentral_small_ATCG_only.fasta
export ENV_ACTIVATE='source ~/miniconda3/bin/activate caduceus_env'

export NUM_DEVICES=8
export PER_DEVICE_BATCH=16
export GLOBAL_BATCH=256
export MAX_LENGTH=1024
export MAX_EPOCHS=null
export MAX_STEPS=20000
export NUM_WORKERS=4
export RUN_DIR=/absolute/path/to/output

sbatch slurm_scripts/run_pretrain_rna.sh
```

The effective gradient accumulation factor is:

```text
ceil(GLOBAL_BATCH / (NUM_NODES * NUM_DEVICES * PER_DEVICE_BATCH))
```

Always begin with a short cluster smoke run:

```bash
export MAX_EPOCHS=null
export MAX_STEPS=20
export RUN_DIR=/absolute/path/to/smoke-output
sbatch slurm_scripts/run_pretrain_rna.sh
```

Before submitting the full job, verify the Slurm log, GPU utilization, host
memory usage, validation loss, and checkpoint creation.

### Epoch and step limits

PyTorch Lightning stops when either `max_epochs` or `max_steps` is reached
first. With approximately 29,600 valid sequences and a global batch size of
256, one epoch contains roughly 93 optimizer steps:

- 50 epochs correspond to approximately 4,650 optimizer steps.
- 20,000 optimizer steps correspond to approximately 216 epochs.

Therefore, 50 epochs and 20,000 steps cannot both be treated as the final
training target. For a 20,000-step run, set `MAX_EPOCHS=null`. For a 50-epoch
run, adjust the cosine schedule and warmup to the resulting total step count;
a 4,000-step warmup would otherwise cover almost the entire run.

## Default model configuration

- Character-level RNA tokenizer;
- 15% same-position MLM;
- `d_model=768` and 12 layers;
- sequence length 1024;
- bidirectional Mamba with additive forward/reverse fusion;
- tied forward/reverse input and output projections;
- hierarchical memory sidecar;
- AdamW with learning rate `8e-5` and weight decay `0.01`;
- FP16 mixed precision and gradient clipping at `0.5`.

Memory is isolated between independent batches by default. Within one forward
pass, a layer may read entries written by earlier layers but never its own newly
written summary. Padding tokens are excluded from memory pooling.

The current memory cross-attention is not reverse-complement equivariant.
Consequently, the supported combinations are:

```text
use_memory=true  -> rcps=false
rcps=true        -> use_memory=false
```

The model rejects configurations that enable both `use_memory=true` and
`rcps=true`.

## Checkpoints and resuming

Training outputs include:

```text
checkpoints_best/       # Best validation-loss checkpoint
checkpoints_periodic/   # Periodic checkpoints
checkpoints/last.ckpt   # Most recent training state
```

The Slurm script uses:

```text
train.ckpt=checkpoints/last.ckpt
```

Training resumes when this file exists and starts from scratch otherwise. Set
`RUN_DIR` to persistent shared storage so checkpoints survive job termination.

## Repository structure

```text
caduceus/               # Caduceus/Mamba model and memory modules
configs/                # Hydra configuration
src/dataloaders/        # RNA datasets and data modules
src/tasks/              # MLM loss and evaluation metrics
slurm_scripts/          # Cluster submission scripts
tests/fixtures/         # Small synthetic test corpus
train.py                # Training entry point
```

## Lineage

This project extends the original Caduceus implementation for mixed-RNA
pretraining and memory-augmented sequence modeling:

```bibtex
@article{schiff2024caduceus,
  title={Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling},
  author={Schiff et al.},
  year={2024}
}
```
