# RNA-Mamba

RNA-Mamba is a bidirectional Mamba model for same-position RNA masked language
modeling. This implementation keeps the original project structure from
commit `bea4cda9a88e4e45490cfabc425e7dd32be29483` and applies only correctness
fixes plus a lightweight cross-layer memory path.

## Model

Each layer contains the original weight-tied BiMamba backbone. At configured
write layers, the lightweight Bidirectional Consistent Writer (BCW):

1. aligns forward and backward states by RNA position;
2. mean-pools valid tokens before any learned projection;
3. applies a shared projection to both directions;
4. builds directional relation features;
5. learns a feature-wise gate and writes one compressed memory vector.

Later layers read only entries written by earlier layers. The reader averages
the small memory bank once per sample and projects the result into the hidden
dimension. It does not run token-to-memory multi-head attention, so its learned
matrix cost does not grow with sequence length.

Default memory settings:

```yaml
use_memory: true
memory_d_sum: 64
memory_d_mem: 64
memory_write_stride: 6
memory_read_stride: 2
memory_persist_across_batches: false
```

Memory is local to one forward pass and is never shared across independent RNA
samples or batches.

## Correctness fixes relative to `bea4cda`

- MLM logits and labels remain aligned at the same sequence positions.
- PAD and EOS tokens are excluded from MLM selection.
- Random MLM replacements are restricted to RNA base tokens.
- Validation and test masking are deterministic.
- Forward and backward summaries use the same valid-token mask.
- A layer reads memory before writing its own state.
- The writer runs only on configured write layers.
- Memory is not duplicated in both detached and differentiable banks.
- Evaluation batches cannot leak memory into one another.
- Memory parameters receive gradients from the MLM objective.
- Input embeddings are created only once and attention masks reach the model.

## Data

Provide one TXT corpus and one FASTA corpus:

```text
data/
  data-random_15K_sequences.txt
  rnacentral_small_ATCG_only.fasta
```

Sequences are uppercased, `T` is converted to `U`, and records containing
characters outside `A/U/C/G` are discarded.

## Environment

The tested cluster environment uses Python 3.10, PyTorch 2.2, CUDA 12.x,
`causal-conv1d==1.2.0.post2`, and `mamba-ssm==1.2.2`.

```bash
bash setup_linux_env.sh
```

## Pretraining

Single GPU 100-epoch pretraining example:

```bash
RUN_DIR=/absolute/path/runs/rna100_lightweight
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 python -m train \
  experiment=hg38/hg38 \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  dataset.dataset_name=mixed_rna \
  +dataset.text_file=/absolute/path/data-random_15K_sequences.txt \
  +dataset.rna_fasta_file=/absolute/path/rnacentral_small_ATCG_only.fasta \
  dataset.tokenizer_name=char \
  +dataset.kmer=1 \
  +dataset.frame=0 \
  dataset.max_length=1024 \
  dataset.batch_size=16 \
  dataset.batch_size_eval=64 \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  loader.num_workers=12 \
  +loader.persistent_workers=true \
  +loader.prefetch_factor=4 \
  model=caduceus \
  model.config.d_model=768 \
  model.config.n_layer=12 \
  model.config.vocab_size=12 \
  model.config.bidirectional=true \
  model.config.bidirectional_strategy=add \
  model.config.bidirectional_weight_tie=true \
  model.config.rcps=false \
  model.config.use_memory=true \
  model.config.memory_d_sum=64 \
  model.config.memory_d_mem=64 \
  model.config.memory_write_stride=6 \
  model.config.memory_read_stride=2 \
  model.config.memory_persist_across_batches=false \
  model.config.pad_token_id=4 \
  optimizer.lr=8e-5 \
  optimizer.weight_decay=0.01 \
  'optimizer.betas=[0.9,0.98]' \
  trainer.precision=16 \
  trainer.gradient_clip_val=0.5 \
  trainer.max_epochs=100 \
  trainer.max_steps=20000 \
  trainer.limit_val_batches=1.0 \
  +trainer.val_check_interval=1.0 \
  trainer.num_sanity_val_steps=0 \
  scheduler._name_=cosine_warmup_timm \
  scheduler.t_initial=30000 \
  scheduler.warmup_t=4000 \
  scheduler.lr_min=2e-5 \
  scheduler.warmup_lr_init=1e-6 \
  train.ckpt=null \
  train.test=false \
  train.monitor=val/loss \
  train.mode=min \
  wandb=null \
  callbacks.model_checkpoint.monitor=val/loss \
  callbacks.model_checkpoint.mode=min \
  callbacks.model_checkpoint.dirpath="$RUN_DIR/checkpoints_best" \
  callbacks.model_checkpoint.filename=val_loss \
  callbacks.periodic_checkpoint.dirpath="$RUN_DIR/checkpoints_periodic" \
  callbacks.model_checkpoint_every_n_steps.dirpath="$RUN_DIR/checkpoints" \
  hydra.run.dir="$RUN_DIR/output"
```

For two GPUs, use `CUDA_VISIBLE_DEVICES=0,1` and `trainer.devices=2`.

## 100-epoch training report

The following run was completed with commit `8e38ae4`.

| Item | Setting |
| --- | --- |
| GPU | 1 x NVIDIA A100-PCIE-40GB |
| Input sequences | 29,621 RNA sequences |
| TXT sequences | 14,991 |
| RNAcentral FASTA sequences | 14,630 |
| Maximum sequence length | 1024 nt |
| Tokenizer | Character-level tokenizer |
| MLM probability | 0.15 |
| Batch size | 16 |
| Evaluation batch size | 64 |
| Epochs | 100 |
| Global steps | 9,300 |
| Precision | FP16 |
| Optimizer | AdamW |
| Learning rate | 8e-5 |
| Weight decay | 0.01 |
| Adam betas | [0.9, 0.98] |
| Gradient clipping | 0.5 |
| Scheduler | Cosine warmup |
| Warmup steps | 4,000 |
| DataLoader workers | 12 |
| Model dimension | 768 |
| Number of layers | 12 |
| Vocabulary size | 12 |
| Bidirectional Mamba | Enabled |
| Bidirectional strategy | Add |
| Bidirectional weight tying | Enabled |
| RCPS | Disabled |
| Lightweight memory | Enabled |
| Memory summary dimension | 64 |
| Memory slot dimension | 64 |
| Memory read stride | 2 |
| Memory write stride | 6 |
| Memory persistence across batches | Disabled |

Training loss decreased from 1.29 at epoch 0 to 0.84 at epoch 99. The lowest
displayed training loss during the run was 0.792 at epoch 79.

## Tests

Run tests from the repository root so the local package is importable:

```bash
python -m pytest -q caduceus/tests
```

The suite checks MLM alignment, deterministic masking, memory isolation,
read-before-write semantics, stride counts, padding-aware BCW, memory
gradients, and model smoke behavior.

## Reproducible environment setup

The tested training environment used Linux, Python 3.10, PyTorch 2.2.x, CUDA 12.x compatible drivers, `mamba-ssm==1.2.2`, and `causal-conv1d==1.2.0.post2`.

```bash
conda create -n rna-mamba python=3.10 -y
conda activate rna-mamba

python -m pip install --upgrade pip setuptools wheel

bash setup_linux_env.sh
If downloading mamba-ssm from GitHub is slow, copy the prebuilt wheel to the server and install it manually:
python -m pip install /path/to/mamba_ssm-1.2.2+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
Verify the environment from the repository root:
python - <<'PY'
import torch
import mamba_ssm

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mamba_ssm:", mamba_ssm.__version__ if hasattr(mamba_ssm, "__version__") else "installed")
PY

python -m pytest -q caduceus/tests
