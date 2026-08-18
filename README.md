# RNA-Mamba

RNA-Mamba is a bidirectional Mamba model for same-position RNA masked language
modeling. This implementation keeps the original project structure from
commit `bea4cda9a88e4e45490cfabc425e7dd32be29483` and applies only correctness
fixes plus a lightweight cross-layer memory path.

## Current 6M-sequence pretraining plan

The current production target is **3,000,000 RNAcentral non-coding RNA
records plus 3,000,000 coding-RNA records**, with complete sequences up to
10,240 nt. The delivery archive contains fewer than 3M independent full-length
mRNAs, so the coding half is assembled transparently from primary full-mRNA
and prokaryotic coding records, then filled to 3M with explicitly labelled
eukaryotic CDS views. m6A labels are excluded from pretraining and reserved
for later fine-tuning.

The million-scale loader uses memory-mapped sequence and offset files; it does
not copy six million Python strings into every DDP process. The preparation
script, data audit, exact epoch-to-step calculation, eight-GPU launcher,
resource estimate, and resume procedure are documented in
[`docs/PRETRAINING_6M.md`](docs/PRETRAINING_6M.md).

Minimal entry points:

```bash
python scripts/prepare_pretraining_6m.py \
  --bundle /path/to/生信.zip \
  --output-dir /path/to/data/processed/rna_pretraining_6m \
  --temp-dir /path/to/fast-temporary-storage

export RNA_PRETRAIN_INDEXED_DIR=/path/to/data/processed/rna_pretraining_6m
export RUN_DIR=/path/to/runs/rna_pretraining_6m
export PRETRAIN_EPOCHS=2
bash scripts/run_pretrain_6m_8gpu.sh
```

With the default 98/1/1 split and global batch 16, one full corpus pass is
approximately 367,500 optimizer steps and the initial two-epoch plan is about
735,000 steps. The launcher reads the exact training count from
`manifest.json` and calculates these values automatically.

## Legacy validated small-corpus workflow (8 GPUs)

This older workflow records the already validated small-corpus experiments and
m6A results. Use the 6M recipe above for the next pretraining run. It runs the
following stages in order:

1. 50,000 optimizer steps of 10,240-nt masked-language-model (MLM)
   pretraining;
2. one epoch of full-model, nucleotide-level m6A fine-tuning on complete
   `5'UTR + CDS + 3'UTR` transcripts;
3. validation-threshold calibration and evaluation on the gene-disjoint test
   set.

The launcher is [`scripts/run_formal_8gpu.sh`](scripts/run_formal_8gpu.sh).
The expanded configuration rationale, measured results, checkpoint semantics,
and resource estimates are in
[`docs/FORMAL_8GPU_TRAINING.md`](docs/FORMAL_8GPU_TRAINING.md). Commands later
in this README are retained for single-GPU development and ablation; they do
not supersede this production recipe.

### 1. Cluster requirements

The validated software stack is Linux, Python 3.10, PyTorch 2.2.0, CUDA 12.x,
`causal-conv1d==1.2.0.post2`, and `mamba-ssm==1.2.2`. The formal configuration
assumes one node with eight NVIDIA GPUs. Eight A100 GPUs are recommended.

Before installation, check the allocation and filesystem:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h "$HOME" /path/to/runs
```

Reserve at least 30 GB of free disk space for prepared data, logs, the best and
last checkpoints, and periodic recovery checkpoints. Each full checkpoint is
approximately 579 MB. Do not launch the job on a cluster login node; run it
inside an interactive GPU allocation or a scheduler job.

### 2. Clone and identify the code

Configure GitHub SSH access for the cluster account, then clone the repository:

```bash
cd /path/to/workspace
git clone git@github.com:zysxmu/mamba-for-RNA.git
cd mamba-for-RNA

git branch --show-current
git log -1 --oneline
git status --short
```

Formal training should use a clean `main` worktree. The launcher records the
exact Git commit and dirty-worktree count in `run_manifest.txt`, so results can
always be traced back to their code.

### 3. Create the environment

The following Conda installation matches the tested binary wheels:

```bash
conda create -n rna-mamba python=3.10 -y
conda activate rna-mamba

python -m pip install --upgrade "pip<27" "setuptools<70" wheel packaging ninja
python -m pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-core.txt

python -m pip install \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2%2Bcu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
python -m pip install \
  "https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2%2Bcu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

If GitHub downloads are slow, download the two wheels elsewhere, transfer them
to the server, and replace the two URLs with their absolute local paths. The
same installation is automated by `setup_linux_env.sh` when a system
`python3.10` executable is available; that script creates `.venv` rather than
a Conda environment.

Verify CUDA kernels and the repository before allocating eight GPUs:

```bash
python - <<'PY'
import causal_conv1d
import mamba_ssm
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
print("causal-conv1d:", getattr(causal_conv1d, "__version__", "installed"))
print("mamba-ssm:", getattr(mamba_ssm, "__version__", "installed"))
PY

python -m pytest -q \
  caduceus/tests \
  tests/test_full_transcript_mlm.py \
  tests/test_m6a_nucleotide.py \
  tests/test_m6a_full_transcript_evaluation.py \
  tests/test_checkpoint_and_collate_contracts.py
```

Do not continue if package import, CUDA availability, or tests fail.

### 4. Place and prepare the data

The production workflow requires the following four source files:

```text
/path/to/source_tables/
  transcript_master.csv.gz
  m6a_nt_mask_full_mrna.csv.gz
  exon_coordinate_map.csv.gz

/path/to/rnacentral_small_ATCG_only.fasta
```

`transcript_master.csv.gz` supplies reference transcript sequences and
`5'UTR/CDS/3'UTR` coordinates. `m6a_nt_mask_full_mrna.csv.gz` supplies one m6A
label per nucleotide: `1` is methylated, `0` is unmethylated, and non-A
positions are excluded from the m6A loss. `exon_coordinate_map.csv.gz` audits
genome-to-transcript coordinates. These files do not contain SNP alleles or
genotypes; SNP-aware training is a later extension and is not required for the
current run.

Prepare gene-disjoint compressed splits:

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

SOURCE_DIR=/path/to/source_tables
M6A_DATA=/path/to/processed/human_m6a_full_transcript
mkdir -p "$M6A_DATA"

python scripts/prepare_m6a_nucleotide.py \
  --transcript-master "$SOURCE_DIR/transcript_master.csv.gz" \
  --mask-table "$SOURCE_DIR/m6a_nt_mask_full_mrna.csv.gz" \
  --exon-map "$SOURCE_DIR/exon_coordinate_map.csv.gz" \
  --output-dir "$M6A_DATA" \
  2>&1 | tee "$M6A_DATA/prepare.log"
```

Audit the contract before training:

```bash
python - "$M6A_DATA/stats.json" <<'PY'
import json
import sys

path = sys.argv[1]
stats = json.load(open(path, encoding="utf-8"))
assert stats["schema_version"] == 2
assert stats["sequence_contract"]["composition"] == \
    "transcript_sequence = 5'UTR + CDS + 3'UTR"
assert stats["splitting"]["leaking_genes"] == 0

for split in ("train", "val", "test"):
    row = stats["splits"][split]
    assert row["full_transcript_training_eligible"] > 0
    print(
        split,
        "eligible_transcripts=", row["full_transcript_training_eligible"],
        "candidate_A=", row["candidate_adenosines"],
        "positive_m6a=", row["positive_m6a"],
    )
print("data audit: PASS")
PY

ls -lh \
  "$M6A_DATA/train.jsonl.gz" \
  "$M6A_DATA/val.jsonl.gz" \
  "$M6A_DATA/test.jsonl.gz" \
  "$M6A_DATA/stats.json"
```

With the supplied tables, the 10,240-nt cap retains 39,349 training, 4,805
validation, and 4,723 test transcripts after reliability and length checks.
The split contains no shared genes.

### 5. Run the complete workflow on eight GPUs

The default command starts long-context pretraining from random model weights,
then automatically fine-tunes and evaluates its best validation-loss
checkpoint:

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_DEVICES=8
export BATCH_SIZE=1
export GRAD_ACCUM=2
export NUM_WORKERS=8

export M6A_DATA=/path/to/processed/human_m6a_full_transcript
export RNA_FASTA_FILE=/path/to/rnacentral_small_ATCG_only.fasta
export RUN_ROOT=/path/to/runs/rna_mamba_formal_8gpu

unset PRETRAIN_INIT_CKPT
unset PRETRAIN_RESUME_CKPT

bash scripts/run_formal_8gpu.sh all
```

The launcher checks the global batch before starting:

```text
8 GPUs x 1 sequence/GPU x 2 accumulation steps = global batch 16
```

It then uses 50,000 MLM optimizer steps, approximately 15.7 epochs on the
validated corpus, and one m6A fine-tuning epoch. Do not change accumulation to
16 on eight GPUs: that would silently increase the global batch from 16 to
128 and make the experiment incomparable.

The from-scratch run is the independent model-training experiment. If the
established 1,024-nt MLM checkpoint is available, long-context continuation
currently gives the best downstream AP. Start that as a new experiment with:

```bash
export RUN_ROOT=/path/to/runs/rna_mamba_formal_8gpu_continued
export PRETRAIN_INIT_CKPT=/absolute/path/to/rna100_lightweight/checkpoints_best/val_loss.ckpt
unset PRETRAIN_RESUME_CKPT

bash scripts/run_formal_8gpu.sh all
```

`PRETRAIN_INIT_CKPT` starts a new optimizer and scheduler from model weights.
It is not an interrupted-job resume.

### 6. Slurm example

Scheduler directives differ by cluster. The template below requests one
eight-GPU node; replace the partition, account, environment path, and data
paths with site-specific values:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=rna_mamba
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/runs/rna_mamba_formal_8gpu/slurm-%j.out
#SBATCH --error=/path/to/runs/rna_mamba_formal_8gpu/slurm-%j.err
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT

set -euo pipefail
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate rna-mamba

cd /path/to/mamba-for-RNA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_DEVICES=8
export BATCH_SIZE=1
export GRAD_ACCUM=2
export NUM_WORKERS=8
export M6A_DATA=/path/to/processed/human_m6a_full_transcript
export RNA_FASTA_FILE=/path/to/rnacentral_small_ATCG_only.fasta
export RUN_ROOT=/path/to/runs/rna_mamba_formal_8gpu

bash scripts/run_formal_8gpu.sh all
```

Submit and monitor it with the commands supported by the cluster, for example:

```bash
sbatch run_rna_mamba.slurm
squeue -u "$USER"
tail -f /path/to/runs/rna_mamba_formal_8gpu/pretrain/console.log
watch -n 5 nvidia-smi
```

### 7. Interrupted-run recovery

For an exact pretraining resume, use the last checkpoint. This restores model
weights, optimizer, scheduler, epoch, and global step:

```bash
export RUN_ROOT=/path/to/runs/rna_mamba_formal_8gpu
unset PRETRAIN_INIT_CKPT
export PRETRAIN_RESUME_CKPT="$RUN_ROOT/pretrain/checkpoints/last.ckpt"

bash scripts/run_formal_8gpu.sh pretrain
```

After pretraining is complete, stages can be launched independently:

```bash
export PRETRAIN_CKPT="$RUN_ROOT/pretrain/checkpoints_best/val_loss.ckpt"
bash scripts/run_formal_8gpu.sh finetune

export FINETUNE_CKPT="$RUN_ROOT/finetune/checkpoints_best/val_m6a_ap.ckpt"
bash scripts/run_formal_8gpu.sh evaluate
```

Never use `PRETRAIN_INIT_CKPT` and `PRETRAIN_RESUME_CKPT` together. Never use a
fine-tuning checkpoint as `PRETRAIN_INIT_CKPT`.

### 8. Successful-run checklist

Retain the following files when the workflow finishes:

```text
RUN_ROOT/run_manifest.txt
RUN_ROOT/pretrain/checkpoints_best/val_loss.ckpt
RUN_ROOT/pretrain/checkpoints/last.ckpt
RUN_ROOT/pretrain/console.log
RUN_ROOT/pretrain/time.txt
RUN_ROOT/finetune/checkpoints_best/val_m6a_ap.ckpt
RUN_ROOT/finetune/console.log
RUN_ROOT/finetune/time.txt
RUN_ROOT/finetune/calibrated_evaluation/m6a_calibrated_evaluation.json
RUN_ROOT/finetune/calibrated_evaluation/rna_mamba_m6a_calibrated_evaluation.png
RUN_ROOT/finetune/calibrated_evaluation/rna_mamba_m6a_calibrated_evaluation.pdf
RUN_ROOT/finetune/calibrated_evaluation/rna_mamba_m6a_calibrated_evaluation.svg
```

The final report should include the Git commit, GPU count and type, wall time,
best pretraining step and validation loss, test AP/AUROC/F1, and the threshold
selected on validation. AP is the primary downstream metric because only
approximately 4.37% of candidate adenosines are positive.

The current reference results on the same gene-disjoint test set are:

| Initialization before m6A fine-tuning | Test AP | AUROC | F1 |
| --- | ---: | ---: | ---: |
| Original full-mRNA baseline | 0.6924 | 0.9839 | 0.6864 |
| Scratch long-context MLM, 20k steps | 0.7143 | 0.9854 | 0.6999 |
| Scratch long-context MLM, 50k steps | 0.7179 | 0.9854 | 0.6960 |
| Continued long-context MLM | **0.7259** | **0.9856** | 0.6970 |

Scratch validation loss was best at step 47,851. Extending the same run from
50,000 to 70,000 optimizer steps did not improve the best checkpoint, so the
formal stopping limit remains 50,000 steps with best-checkpoint selection.

### 9. Common launch failures

- **`No module named pytest`**: the wrong Conda environment is active. Run
  `conda activate rna-mamba` and verify `which python`.
- **`No module named mamba_ssm`**: install the Python 3.10, PyTorch 2.2 wheel
  shown above; do not use a wheel built for another Python or PyTorch ABI.
- **Global-batch error**: keep `NUM_DEVICES=8`, `BATCH_SIZE=1`, and
  `GRAD_ACCUM=2`, or deliberately redesign the optimizer schedule.
- **Missing prepared data**: rerun the preparation and audit commands; the
  launcher requires all three `.jsonl.gz` splits and `stats.json`.
- **CUDA out of memory**: first confirm that per-GPU batch size is one and that
  no unrelated process occupies the GPUs. Do not silently crop transcripts.
- **Disk full or stalled checkpoint writes**: inspect `df -h` and retain the
  best and last checkpoints before removing superseded periodic checkpoints.
- **Training stopped with no Python process in `nvidia-smi`**: inspect the end
  of `console.log`, the Slurm `.err` file, scheduler state, and `time.txt`
  before restarting.

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

### Complete-transcript long-context MLM

`experiment=rna_long_pretrain` is the formal long-context pretraining path.
One sample is one complete RNA sequence, never a silent crop. The mRNA source
uses the gene-level train/validation/test files produced by
`prepare_m6a_nucleotide.py`; methylation labels are not read by the MLM task.
An optional RNAcentral FASTA source supplies ncRNA and is assigned to
reproducible 80/10/10 splits. Sequences longer than the configured 10,240-nt
cap, coordinate-unreliable mRNAs, PAD positions, and special tokens are
reported or excluded from the MLM loss.

Run a bounded single-GPU integration test before spending a cluster allocation:

```bash
export M6A_FULL_DATA_DIR=/home/zys/mamba-for-RNA/data/processed/human_m6a_full_transcript
export RNA_FASTA_FILE=/home/zys/mamba-for-RNA/data/rnacentral_small_ATCG_only.fasta
MLM_CKPT=/home/zys/mamba-for-RNA/runs/rna100_lightweight/checkpoints_best/val_loss.ckpt
RUN_DIR=/home/zys/mamba-for-RNA/runs/rna_long_10240_benchmark
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v -o "$RUN_DIR/time.txt" \
python -m train \
  experiment=rna_long_pretrain \
  trainer.devices=1 \
  train.pretrained_model_path="$MLM_CKPT" \
  dataset.max_train_sequences=100 \
  dataset.max_val_sequences=16 \
  dataset.max_test_sequences=16 \
  trainer.max_epochs=1 \
  trainer.max_steps=-1 \
  trainer.accumulate_grad_batches=1 \
  train.test=false \
  callbacks.model_checkpoint.save_top_k=0 \
  callbacks.periodic_checkpoint.save_top_k=0 \
  callbacks.model_checkpoint_every_n_steps.save_top_k=0 \
  callbacks.model_checkpoint_every_n_steps.save_last=false \
  wandb=null \
  hydra.run.dir="$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/console.log"
```

The validated formal limit is 50,000 optimizer steps at global batch 16,
which is about 15.7 epochs on the current corpus. The best scratch checkpoint
occurred at step 47,851; extending the run to 70,000 steps did not improve
validation loss. For an 8-GPU node, use one sequence per GPU and two gradient
accumulation steps. The exact command, checkpoint layout, resume rules, and
resource estimate are in the formal 8-GPU guide linked above.

After long-context MLM converges, use its best `val_loss.ckpt` as
`train.pretrained_model_path` for `experiment=human_m6a_full_transcript`.
That second stage replaces the MLM head with the nucleotide m6A classifier and
fine-tunes the complete model on labelled adenosines.

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

If the benchmark is stable, initialize from the best long-context MLM
`val_loss.ckpt` and fine-tune all model parameters for one epoch. Across the
validated runs, the first epoch was consistently the best validation-AP
checkpoint and later epochs overfit. On eight GPUs, keep global batch 16 with
`dataset.batch_size=1` and `trainer.accumulate_grad_batches=2`. Use
`bash scripts/run_formal_8gpu.sh finetune` for the exact launch command.

Do not set `train.ckpt` to the old window checkpoint: that is an exact resume
and would also restore its optimizer and epoch state. Use
`train.pretrained_model_path` as shown above for a new long-context run.

### Leakage-safe threshold calibration and stratified evaluation

The full-transcript evaluator first chooses one decision threshold using the
validation split only. It then freezes that threshold before evaluating the
gene-disjoint test split. This avoids tuning the classifier on test labels.
In addition to overall AP and AUROC, it reports thresholded precision, recall,
F1, balanced accuracy, and MCC, plus separate test results for 5'UTR, CDS,
3'UTR, and transcript-length bins. Metrics are accumulated with bounded-memory
histograms rather than retaining every adenosine score in RAM.

The command below can run on GPU 1 while a pretraining job occupies GPU 0:

```bash
cd /home/zys/mamba-for-RNA
conda activate rna-mamba

BEST_CKPT=/home/zys/mamba-for-RNA/runs/human_m6a_full_10240_from_long/checkpoints_best/val_m6a_ap.ckpt
M6A_DATA=/home/zys/mamba-for-RNA/data/processed/human_m6a_full_transcript
EVAL_DIR=/home/zys/mamba-for-RNA/runs/human_m6a_full_10240_from_long/calibrated_evaluation
mkdir -p "$EVAL_DIR"

CUDA_VISIBLE_DEVICES=1 /usr/bin/time -v -o "$EVAL_DIR/time.txt" \
python scripts/evaluate_m6a_full_transcript.py \
  --checkpoint "$BEST_CKPT" \
  --data-dir "$M6A_DATA" \
  --output-dir "$EVAL_DIR" \
  --batch-size 1 \
  --num-workers 8 \
  --device cuda \
  --bins 4096 \
  --threshold-objective f1 \
  2>&1 | tee "$EVAL_DIR/console.log"
```

The primary machine-readable result is
`m6a_calibrated_evaluation.json`. The directory also contains a combined CSV,
validation-threshold search data, PR/ROC source data, and publication-ready
PNG, SVG, and PDF figures. If Matplotlib is unavailable, install the pinned
version with `python -m pip install matplotlib==3.7.4`; all metric tables are
still written before plotting.

### Current complete-transcript results

The same gene-disjoint test set was used for every row. Thresholds were chosen
on validation only and frozen before testing. AP is the primary metric because
positive m6A calls make up approximately 4.37% of candidate adenosines.

| Initialization | Test AP | AUROC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original full-mRNA baseline | 0.6924 | 0.9839 | 0.6242 | 0.7623 | 0.6864 |
| Scratch long-context MLM, 20k steps | 0.7143 | 0.9854 | 0.6346 | 0.7801 | 0.6999 |
| Scratch long-context MLM, 50k steps | 0.7179 | 0.9854 | 0.6364 | 0.7679 | 0.6960 |
| Continued long-context MLM | **0.7259** | **0.9856** | 0.6351 | 0.7724 | 0.6970 |

The 50k scratch model improved AP for transcripts longer than 8,192 nt from
0.6535 to 0.6775 relative to the 20k scratch model. Continued pretraining is
the best current initialization by overall AP; the scratch 50k run remains the
clean from-scratch ablation.

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
  tests/test_m6a_full_transcript_evaluation.py \
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
scripts/evaluate_m6a_full_transcript.py  validation-calibrated full-mRNA evaluation
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
