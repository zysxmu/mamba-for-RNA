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

Single GPU:

```bash
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
  loader.num_workers=4 \
  model=caduceus \
  model.config.d_model=768 \
  model.config.n_layer=12 \
  model.config.vocab_size=12 \
  model.config.bidirectional=true \
  model.config.bidirectional_strategy=add \
  model.config.bidirectional_weight_tie=true \
  model.config.rcps=false \
  model.config.pad_token_id=4 \
  optimizer.lr=8e-5 \
  optimizer.weight_decay=0.01 \
  'optimizer.betas=[0.9,0.98]' \
  trainer.precision=16 \
  trainer.gradient_clip_val=0.5 \
  trainer.max_epochs=50 \
  trainer.max_steps=20000 \
  trainer.limit_val_batches=1.0 \
  +trainer.val_check_interval=1.0 \
  trainer.num_sanity_val_steps=0 \
  scheduler._name_=cosine_warmup_timm \
  scheduler.t_initial=30000 \
  scheduler.warmup_t=4000 \
  scheduler.lr_min=2e-5 \
  scheduler.warmup_lr_init=1e-6 \
  wandb=null
```

For two GPUs, use `CUDA_VISIBLE_DEVICES=0,1` and `trainer.devices=2`.

## Tests

Run tests from the repository root so the local package is importable:

```bash
python -m pytest -q caduceus/tests
```

The suite checks MLM alignment, deterministic masking, memory isolation,
read-before-write semantics, stride counts, padding-aware BCW, memory
gradients, and model smoke behavior.
