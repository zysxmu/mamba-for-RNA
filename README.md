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

## Human m6A sliding-window fine-tuning

The human m6A task is a downstream fine-tuning task. It does not replace the
mixed-RNA pretraining corpus. The preparation script maps each observed m6A
context to one unique CDS position, removes repeated calls at the same
transcript position, splits by gene before windowing, and converts `T` to `U`.

Prepare the two supplied ZIP archives without extracting their multi-gigabyte
members:

```bash
python scripts/prepare_human_m6a.py \
  --cds-source /path/to/data-20260731T165030Z-1-001.zip \
  --sites-source /path/to/data-20260731T165030Z-1-002.zip \
  --output-dir data/processed/human_m6a \
  --window-length 128 \
  --stride 64
```

The output contains `train.jsonl`, `val.jsonl`, `test.jsonl`, and an auditable
`stats.json`. Sequences are stored once per transcript; overlapping windows are
created lazily by the dataloader. The primary target is the number of observed,
uniquely mapped m6A sites in each window. Unlabelled adenosines are not claimed
to be experimentally verified negatives.

For the supplied archives, the audit found 10,240 transcripts and 293,367 m6A
rows. A total of 266,514 sites (90.85%) mapped uniquely with a centred 41-nt or
21-nt context. The default gene-level split produced 220,649 training, 27,881
validation, and 26,978 test windows; approximately 79% contain at least one
observed m6A site.

Run a 500-window smoke fine-tune from the best MLM checkpoint:

```bash
python -m train \
  experiment=human_m6a_window \
  train.pretrained_model_path=/path/to/checkpoints_best/val_loss.ckpt \
  train.pretrained_model_state_hook.freeze_backbone=true \
  dataset.max_train_windows=500 \
  dataset.max_val_windows=100 \
  dataset.max_test_windows=100 \
  trainer.max_epochs=3 \
  wandb=null \
  hydra.run.dir=outputs/human-m6a-smoke
```

After the smoke run and `stats.json` review, remove the three
`max_*_windows` overrides and set
`train.pretrained_model_state_hook.freeze_backbone=false` for complete
fine-tuning. On an 8 GB RTX 4060, start with the configured batch size of 8 and
reduce it to 4 if necessary. Increase `loader.num_workers` only on Linux after
the first successful run.

The supplied data converged after one complete full-parameter epoch. The best
checkpoint produced train/validation/test MAE values of 0.845/0.840/0.853 and
test MSE of 1.386 on 26,978 windows from 999 held-out genes. A constant
two-sites-per-window baseline has test MAE 1.255 and MSE 2.531. Later epochs
overfit, so checkpoint selection must remain based on validation loss.

Evaluate a complete fine-tuned checkpoint without entering the fit loop:

```bash
python -m train \
  experiment=human_m6a_window \
  dataset.data_dir=/absolute/path/data/processed/human_m6a \
  dataset.batch_size_eval=128 \
  loader.num_workers=8 \
  train.pretrained_model_path=null \
  train.ckpt=/absolute/path/checkpoints/val/loss.ckpt \
  train.eval_only=true \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  trainer.precision=16 \
  wandb=null \
  hydra.run.dir=outputs/human-m6a-test
```

`train.pretrained_model_state_hook` is a warm-start transformation only. Exact
resume and test loading restore the complete checkpoint, including the
fine-tuned regression head.

### Sparse multi-site recovery

The count model predicts how many observed m6A sites occur in each 128-nt
window. Dense overlapping windows define an implicit measurement system
`y = Wp`, where columns of `W` correspond only to adenosines and `p` contains
non-negative sparse site scores. The recovery script solves a box-constrained
non-negative L1 problem without materializing `W`.

First verify identifiability using exact validation-window counts. This is an
oracle geometry check, not a model result:

```bash
python scripts/reconstruct_m6a_sites.py \
  --data-dir data/processed/human_m6a \
  --split val \
  --mode oracle \
  --window-length 128 \
  --stride 1 \
  --l1-penalty 0.01 \
  --output-dir outputs/m6a-cs-oracle-val
```

On the supplied validation split, dense stride-1 exact counts recover 99.99%
of observed sites at `K = number of observed sites`, establishing that the
window geometry is identifiable. It does not establish performance under
noisy model-predicted counts.

Next tune sparse recovery on validation predictions only. Start with a bounded
50-transcript run before processing the full validation split:

```bash
python scripts/reconstruct_m6a_sites.py \
  --data-dir data/processed/human_m6a \
  --split val \
  --mode model \
  --checkpoint /absolute/path/checkpoints/val/loss.ckpt \
  --window-length 128 \
  --stride 1 \
  --l1-penalty 0.1 \
  --max-transcripts 50 \
  --batch-size 128 \
  --device cuda \
  --output-dir outputs/m6a-cs-model-val
```

Choose stride, L1 penalty, and stopping rules using validation only. The script
refuses to read `test.jsonl` unless `--allow-test` is supplied after those
choices are frozen. Because unlabelled adenosines are unknown rather than
verified negatives, site metrics are explicitly reported as recovery of
observed calls, not biological specificity.

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

## Nucleotide-level human m6A fine-tuning

The full-mRNA task uses `transcript_sequence` as input and the equally long
`m6a_nt_mask` as its target. A labelled adenosine is `1`, an unmethylated
adenosine is `0`, and non-A bases are excluded from the loss. Preparation now
verifies the strict 0-based, half-open contract
`transcript_sequence == 5'UTR + CDS + 3'UTR`, including CDS start/end and
sequence-derived start/stop-codon flags. Splits are made by gene rather than by
window or transcript isoform.

Prepare and audit the three source tables:

```bash
cd /home/zys/mamba-for-RNA
conda activate rna-mamba

SOURCE_DIR=/home/zys/gpt
M6A_NT_DATA_DIR=/home/zys/mamba-for-RNA/data/processed/human_m6a_nucleotide
mkdir -p "$M6A_NT_DATA_DIR"

python scripts/prepare_m6a_nucleotide.py \
  --transcript-master "$SOURCE_DIR/transcript_master.csv.gz" \
  --mask-table "$SOURCE_DIR/m6a_nt_mask_full_mrna.csv.gz" \
  --exon-map "$SOURCE_DIR/exon_coordinate_map.csv.gz" \
  --output-dir "$M6A_NT_DATA_DIR" \
  | tee "$M6A_NT_DATA_DIR/prepare.log"

cat "$M6A_NT_DATA_DIR/stats.json"
```

The processed files retain all records and region boundaries. Recommended
full-transcript training requires both `mrna_coordinate_system_reliable=true`
and `cds_boundary_reliable=true`. The legacy 1024-nt window experiment remains
available as `experiment=human_m6a_nucleotide` for controlled comparisons.

### Complete-transcript long-context training

`experiment=human_m6a_full_transcript` treats one complete
`5'UTR + CDS + 3'UTR` sequence as one item. It does not split a transcript by
region or silently truncate it. The default 10,240-nt cap retains approximately
98% of coordinate-reliable labelled transcripts; excluded longer transcripts
are counted in the startup audit. Batches use dynamic right padding, and the
padding-aware BiMamba reverse path reverses only each sample's valid prefix.

First benchmark 100 optimizer steps on the A100 80GB GPU:

```bash
export M6A_NT_DATA_DIR=/home/zys/mamba-for-RNA/data/processed/human_m6a_nucleotide
NT_CKPT=/home/zys/mamba-for-RNA/runs/human_m6a_nucleotide_full_retry/checkpoints/val/m6a_average_precision.ckpt
RUN_DIR=/home/zys/mamba-for-RNA/runs/human_m6a_full_10240_benchmark
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v -o "$RUN_DIR/time.txt" \
python -m train \
  experiment=human_m6a_full_transcript \
  train.pretrained_model_path="$NT_CKPT" \
  train.pretrained_model_strict_load=true \
  dataset.max_sequence_length=10240 \
  dataset.max_train_transcripts=512 \
  dataset.max_val_transcripts=64 \
  dataset.max_test_transcripts=64 \
  dataset.batch_size=1 \
  dataset.batch_size_eval=1 \
  loader.num_workers=8 \
  trainer.max_epochs=1 \
  +trainer.max_steps=100 \
  trainer.accumulate_grad_batches=1 \
  trainer.limit_val_batches=0 \
  train.test=false \
  wandb=null \
  hydra.run.dir="$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/console.log"
```

If the benchmark is stable, run the complete job. This preserves the existing
nucleotide-level classifier and fine-tunes all model parameters on complete
transcripts:

```bash
export M6A_NT_DATA_DIR=/home/zys/mamba-for-RNA/data/processed/human_m6a_nucleotide
NT_CKPT=/home/zys/mamba-for-RNA/runs/human_m6a_nucleotide_full_retry/checkpoints/val/m6a_average_precision.ckpt
RUN_DIR=/home/zys/mamba-for-RNA/runs/human_m6a_full_10240
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v -o "$RUN_DIR/time.txt" \
python -m train \
  experiment=human_m6a_full_transcript \
  train.pretrained_model_path="$NT_CKPT" \
  train.pretrained_model_strict_load=true \
  dataset.max_sequence_length=10240 \
  dataset.batch_size=1 \
  dataset.batch_size_eval=1 \
  loader.num_workers=8 \
  trainer.max_epochs=5 \
  trainer.accumulate_grad_batches=16 \
  wandb=null \
  hydra.run.dir="$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/console.log"
```

Do not set `train.ckpt` to the old window checkpoint: that is an exact resume
and would also restore its optimizer and epoch state. Use
`train.pretrained_model_path` as shown above for a new long-context run.

### Legacy 1024-nt window baseline

Run a one-epoch integration smoke test before the full job. The warm-start
hook loads only shape-compatible backbone tensors and deliberately leaves the
new nucleotide classifier randomly initialized.

```bash
export M6A_NT_DATA_DIR=/home/zys/mamba-for-RNA/data/processed/human_m6a_nucleotide
PRETRAINED_CKPT=/home/zys/mamba-for-RNA/runs/human_m6a_full_finetune/checkpoints/val/loss.ckpt
RUN_DIR=/home/zys/mamba-for-RNA/runs/human_m6a_nucleotide_smoke

CUDA_VISIBLE_DEVICES=0 python -m train \
  experiment=human_m6a_nucleotide \
  train.pretrained_model_path="$PRETRAINED_CKPT" \
  dataset.max_train_windows=64 \
  dataset.max_val_windows=32 \
  dataset.max_test_windows=32 \
  dataset.batch_size=2 \
  dataset.batch_size_eval=4 \
  loader.num_workers=4 \
  trainer.max_epochs=1 \
  trainer.accumulate_grad_batches=1 \
  wandb=null \
  hydra.run.dir="$RUN_DIR"
```

After the smoke test succeeds, launch the complete single-GPU fine-tuning run:

```bash
export M6A_NT_DATA_DIR=/home/zys/mamba-for-RNA/data/processed/human_m6a_nucleotide
PRETRAINED_CKPT=/home/zys/mamba-for-RNA/runs/human_m6a_full_finetune/checkpoints/val/loss.ckpt
RUN_DIR=/home/zys/mamba-for-RNA/runs/human_m6a_nucleotide_full
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v -o "$RUN_DIR/time.txt" \
python -m train \
  experiment=human_m6a_nucleotide \
  train.pretrained_model_path="$PRETRAINED_CKPT" \
  dataset.batch_size=8 \
  dataset.batch_size_eval=16 \
  loader.num_workers=8 \
  trainer.max_epochs=10 \
  wandb=null \
  hydra.run.dir="$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/console.log"
```

Checkpoint selection uses validation average precision, which is more
informative than raw accuracy for the approximately 4.3% positive A sites.
The run also reports AUROC, precision, recall, F1, accuracy, and the observed
positive rate on A candidates. The ranking metrics use a distributed
fixed-memory histogram, so evaluation does not retain millions of site scores
in GPU memory.

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
python -m pytest -q \
  caduceus/tests/test_padding_aware_bimamba.py \
  tests/test_human_m6a_window.py \
  tests/test_m6a_nucleotide.py \
  tests/test_m6a_sparse.py \
  tests/test_checkpoint_and_collate_contracts.py
```

The suite checks MLM alignment, deterministic masking, memory isolation,
read-before-write semantics, stride counts, padding-aware BCW, memory
gradients, and model smoke behavior.

## Project structure

```text
caduceus/memory/writer.py  lightweight bidirectional memory writer
caduceus/               model configuration and lightweight memory integration
configs/                Hydra model, data, pipeline, and experiment settings
src/dataloaders/        mixed-RNA and human-m6A loading and collation
src/m6a_sparse.py       implicit window system and non-negative sparse solver
scripts/                m6A preparation and sparse site-recovery entry points
scripts/prepare_human_m6a.py  human m6A mapping and gene-level splitting
scripts/prepare_m6a_nucleotide.py  full-mRNA mask validation and gene splitting
src/callbacks/          runtime and validation logging
slurm_scripts/          cluster submission scripts
tests/fixtures/         synthetic smoke-test data
train.py                training, exact resume/evaluation, and warm-start entry point
```

## Reproducible environment setup

The tested training environment used Linux, Python 3.10, PyTorch 2.2.x, CUDA
12.x compatible drivers, `mamba-ssm==1.2.2`, and
`causal-conv1d==1.2.0.post2`.

```bash
conda create -n rna-mamba python=3.10 -y
conda activate rna-mamba

python -m pip install --upgrade pip setuptools wheel

bash setup_linux_env.sh
```

If downloading `mamba-ssm` from GitHub is slow, copy the prebuilt wheel to the
server and install it manually:

```bash
python -m pip install /path/to/mamba_ssm-1.2.2+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Verify the environment from the repository root:

```bash
python - <<'PY'
import torch
import mamba_ssm

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mamba_ssm:", mamba_ssm.__version__ if hasattr(mamba_ssm, "__version__") else "installed")
PY

python -m pytest -q caduceus/tests
```
