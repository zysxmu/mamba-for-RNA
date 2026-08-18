# Six-million-sequence RNA-Mamba pretraining

This document is the authoritative recipe for the next pretraining run. The
scope is pretraining only: approximately three million non-coding RNA records
and three million coding-RNA records. The m6A tables in the delivery archive
are deliberately reserved for later fine-tuning.

## 1. Delivered data and the final corpus contract

The source archive is `生信.zip`. Its relevant pretraining contents are:

| Source | Delivered content | Role in the 6M corpus |
| --- | --- | --- |
| RNAcentral selection | exactly 3,000,000 records selected from 40,712,942 active RNAcentral records; seed 2357; lengths 18--10,240 nt | 3M non-coding RNA records |
| Eukaryote archives | 45 species and 1,532,392 protein-coding full-mRNA transcript records, with matching CDS/UTR FASTA files | primary coding records; CDS can supply secondary views |
| Prokaryote archive | 51 species and 149,969 coding records, primarily CDS | primary coding records |
| Human transcript master | 211,446 full-transcript rows with `5'UTR + CDS + 3'UTR` sequences and coordinates | primary coding records |
| Human/mouse m6A tables and m6A archive | nucleotide modification labels | **not used in pretraining**; retained for fine-tuning |

The archive therefore does not contain three million distinct full-length mRNA
records. The preparation code uses this explicit policy instead of duplicating
records or silently calling CDS sequences full mRNA:

1. keep valid unique eukaryotic full mRNA, human full mRNA, and prokaryotic CDS
   as primary coding records;
2. if those records do not reach 3M after length/alphabet checks, draw a seeded
   sample of unique eukaryotic CDS sequences as `eukaryote_cds_view` records;
3. stop at exactly 3M coding records;
4. combine them with the delivered 3M RNAcentral records;
5. write the exact source type of every record to compressed metadata.

This distinction must remain in the methods section: the coding half is a
three-million-record coding-sequence corpus, not three million independent
full-length mRNAs.

All sequences are upper-cased, `T` is converted to `U`, the accepted alphabet
is `A/U/C/G/N`, and lengths outside 18--10,240 nt are excluded. Sequences are
never truncated. Primary coding records and CDS-view records are
content-deduplicated within their respective representation pools. A CDS view
may legitimately have the same sequence as its full-mRNA view, but both are
labelled as such and are assigned to the same split. The already curated
RNAcentral 3M record selection is preserved at accession level.

## 2. Why the indexed format is required

The older data loader stored every RNA sequence as a Python string. With six
million records and eight DDP processes, that would duplicate the corpus in
host memory eight times. The new preparation script writes:

```text
rna_pretraining_6m/
  manifest.json
  train.sequences.bin
  train.offsets.u64
  train.sources.u8
  train.records.tsv.gz
  val.*
  test.*
```

Each worker memory-maps sequence bytes and offsets on demand. The loader keeps
random access for distributed sampling but does not load all sequence strings
into RAM. The metadata files preserve source class, source type, species,
record identifier, and length for every example.

## 3. Prepare the data once

Use a fast local or parallel filesystem with at least 40 GB of temporary free
space and at least 30 GB for the final prepared data. Actual usage depends on
the sequence-length distribution. Do not prepare the archive independently in
multiple distributed ranks.

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

BUNDLE=/path/to/生信.zip
DATA_DIR=/path/to/data/processed/rna_pretraining_6m
TEMP_DIR=/path/to/fast-temporary-storage

python scripts/prepare_pretraining_6m.py \
  --bundle "$BUNDLE" \
  --output-dir "$DATA_DIR" \
  --temp-dir "$TEMP_DIR" \
  2>&1 | tee /path/to/prepare_pretraining_6m.log
```

Preparation is transactional. Files are first created under
`rna_pretraining_6m.building` and moved into place only after every count and
file-size check succeeds. Pass `--overwrite` only when intentionally replacing
an existing prepared corpus.

Audit the result before requesting GPUs:

```bash
python - "$DATA_DIR/manifest.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["schema_version"] == 1
assert d["totals"]["records"] == 6_000_000
assert d["totals"]["source_class_counts"] == {
    "ncRNA": 3_000_000,
    "coding": 3_000_000,
}
assert d["corpus_contract"]["m6a_used"] is False
print(json.dumps(d["corpus_contract"], indent=2))
print(json.dumps(d["splits"], indent=2))
print("6M pretraining data audit: PASS")
PY
```

The default split is stable 98%/1%/1% assignment. Full-mRNA and CDS views of
the same species/transcript identifier are always placed in the same split.
It produces approximately
5.88M training records and 60k records in each evaluation split. The exact
counts are recorded in `manifest.json`; training never relies on an estimated
count.

## 4. Preflight test

Install the environment described in the main README, then run:

```bash
python -m pytest -q \
  caduceus/tests \
  tests/test_full_transcript_mlm.py \
  tests/test_indexed_rna_pretraining.py \
  tests/test_checkpoint_and_collate_contracts.py
```

Run a short one-GPU smoke job before allocating an eight-GPU node:

```bash
export RNA_PRETRAIN_INDEXED_DIR="$DATA_DIR"

CUDA_VISIBLE_DEVICES=0 python -m train \
  experiment=rna_6m_pretrain \
  trainer.devices=1 \
  trainer.max_steps=20 \
  trainer.accumulate_grad_batches=1 \
  dataset.max_train_sequences=100 \
  dataset.max_val_sequences=16 \
  dataset.max_test_sequences=16 \
  callbacks.model_checkpoint.save_top_k=0 \
  callbacks.periodic_checkpoint.save_top_k=0 \
  callbacks.model_checkpoint_every_n_steps.save_top_k=0 \
  train.test=false \
  wandb=null \
  hydra.run.dir=runs/rna_6m_smoke
```

## 5. Formal eight-GPU launch

The formal default is a from-scratch 12-layer, 768-dimensional RNA-Mamba with
about 49M trainable parameters, character-level MLM, 15% masking, a 10,240-nt
maximum sequence length, FP16, and the current lightweight BCW/cross-layer
memory configuration. m6A labels are not inputs or targets in this stage.

| Item | Formal setting |
| --- | --- |
| Backbone | 12-layer, 768-dimensional weight-tied BiMamba |
| Parameters | approximately 49M trainable parameters |
| Lightweight memory | enabled; write stride 6, read stride 2 |
| Memory dimensions | summary 64, memory 64, 4 heads, maximum 32 slots |
| Cross-batch state | disabled (`memory_persist_across_batches=false`) |
| Objective | same-position character MLM; 15% eligible bases selected |
| Maximum length | 10,240 nt; dynamic right padding; no truncation |
| Precision | FP16 |
| Optimizer | AdamW, LR `8e-5`, weight decay `0.01`, betas `[0.9, 0.98]` |
| Scheduler | cosine decay over the exact run, 20k warmup steps, minimum LR `2e-5` |
| Default global batch | 16 sequences (`8 GPUs x 1 x accumulation 2`) |
| Initial stopping plan | 2 complete training-corpus passes; best validation-loss checkpoint |

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

export RNA_PRETRAIN_INDEXED_DIR=/path/to/data/processed/rna_pretraining_6m
export RUN_DIR=/path/to/runs/rna_pretraining_6m
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_DEVICES=8
export BATCH_SIZE=1
export GRAD_ACCUM=2
export NUM_WORKERS=4
export PRETRAIN_EPOCHS=2

bash scripts/run_pretrain_6m_8gpu.sh
```

The launcher calculates the optimizer-step target from the actual manifest:

```text
global batch = GPU count x per-GPU batch x gradient accumulation
steps per epoch = ceil(training records / global batch)
max steps = steps per epoch x PRETRAIN_EPOCHS
```

For an approximately 5.88M-record training split and global batch 16, this is
about 367,500 optimizer steps per epoch and 735,000 steps for two epochs. The
old 50,000-step setting is only about 0.14 epoch on this corpus and is not a
complete training pass.

Two epochs are the initial compute plan, not a claim that convergence is
guaranteed. Select the best `val/loss` checkpoint. Add a third epoch only if
the second-epoch validation loss is still improving materially. Do not choose
the last checkpoint merely because it is later.

## 6. Time and resource planning

The teacher's earlier eight-H200 run on the much smaller mixed corpus took
about three hours for 50,000 optimizer steps. A purely linear extrapolation is
approximately 44 hours for 735,000 steps. That is a planning estimate, not a
measurement of the new corpus: the new length distribution, storage bandwidth,
padding, and validation set can change throughput.

Use the first 2,000--5,000 steps of the formal job to record measured seconds
per optimizer step, then estimate:

```text
remaining wall time = remaining optimizer steps x measured seconds per step
```

The launcher writes the Git commit, GPU inventory, exact record counts, global
batch, epoch-derived step target, scheduler length, and resume choice to
`run_manifest.txt`. `/usr/bin/time` writes measured total runtime to
`time.txt`.

Recommended minimum allocation for preparation and training:

- one node with 8 H200 GPUs;
- 128 GB host RAM (the indexed loader itself needs far less, but preparation
  uses hash sets and the operating system benefits from filesystem cache);
- at least 100 GB free run/data space for prepared data, logs, and recovery
  checkpoints;
- local NVMe or a high-throughput parallel filesystem.

## 7. Resume and required outputs

To resume an interrupted run exactly, restore model, optimizer, scheduler and
step state from the last checkpoint:

```bash
export RESUME_CKPT=/path/to/runs/rna_pretraining_6m/checkpoints/last.ckpt
bash scripts/run_pretrain_6m_8gpu.sh
```

Keep these outputs:

```text
data/processed/rna_pretraining_6m/manifest.json
runs/rna_pretraining_6m/run_manifest.txt
runs/rna_pretraining_6m/console.log
runs/rna_pretraining_6m/time.txt
runs/rna_pretraining_6m/checkpoints_best/val_loss.ckpt
runs/rna_pretraining_6m/checkpoints/last.ckpt
```

Fine-tuning is intentionally a separate later decision. At that point the m6A
tables can be processed and the best pretraining checkpoint loaded into the
nucleotide-level classification model.
