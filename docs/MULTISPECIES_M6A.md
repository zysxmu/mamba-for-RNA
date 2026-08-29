# Six-species full-transcript m6A fine-tuning

This is the authoritative data contract for the downstream m6A task. It is
separate from the 3M non-coding + approximately 2M coding RNA masked-language
model pretraining corpus. m6A labels are never mixed into MLM pretraining.

## Dataset contract

The formal task uses the six species in `m6A_modification_dataset.zip`.
`scripts/prepare_multispecies_m6a.py` joins every nucleotide mask to the
corresponding complete mRNA in the eukaryotic sequence archives without
extracting those large ZIP files. Human and mouse remain supported as optional
extensions, but are not included in the default formal six-species build.

| Canonical species | Masked transcripts | Strictly reliable complete mRNAs | Positive m6A sites |
| --- | ---: | ---: | ---: |
| `pan_troglodytes` | 13,112 | 13,112 | 124,595 |
| `arabidopsis_thaliana` | 18,628 | 18,628 | 87,211 |
| `saccharomyces_cerevisiae` | 5,633 | 5,633 | 73,056 |
| `macaca_mulatta` | 12,733 | 12,733 | 116,621 |
| `sus_scrofa` | 25,965 | 25,965 | 324,208 |
| `rattus_norvegicus` | 13,405 | 13,325 | 135,400 |
| **Formal six-species total** | **89,476** | **89,396** | **861,091** |

Optional human and mouse sources contain another 71,277 and 48,352 labelled
transcripts, respectively. They must be requested explicitly with `--species`
and should be reported as a separate eight-species experiment.

One sample is one complete transcript:

```text
complete mRNA = 5'UTR + CDS + 3'UTR
```

Coordinates are zero-based, half-open transcript coordinates. `cds_start` is
the index of the first CDS nucleotide in that complete mRNA. A mask value of
`1` is a methylated adenosine, `0` is an unmethylated adenosine, and non-A
positions are excluded from both loss and metrics.

The split key is a connected component of `species + version-stripped gene_id`
and the exact sequence hash. Thus isoforms from one gene cannot leak between
train, validation, and test, and completely identical input sequences are kept
in the same split even when they come from different genes or species. Source
`FAIL` rows and unreliable CDS boundaries remain visible in the audit but are
excluded by the training configuration. T is converted to U. Other IUPAC
ambiguity symbols are converted to N and counted; only nine ambiguous
nucleotides were found in the complete delivered dataset.
The final audit found 89,423 exact-sequence groups; 51 groups (104 records)
were duplicated and were deliberately kept within one split.

The six-species archive contains only transcripts with at least one observed
m6A site. Consequently it contains positive and unmodified A positions within
selected transcripts, but no transcript-level all-negative controls. This
source-selection fact is recorded per species in `stats.json` and must be
disclosed when reporting results.

## Prepare the data

First create the canonical source tree as documented in
[`PRETRAINING_5M.md`](PRETRAINING_5M.md), then run:

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

export RNA_MAMBA_DATA_ROOT=/path/to/rna_mamba_data
export MULTISPECIES_M6A_DATA_DIR="$RNA_MAMBA_DATA_ROOT/processed/multispecies_m6a_full_transcript"

python scripts/prepare_multispecies_m6a.py \
  --data-root "$RNA_MAMBA_DATA_ROOT" \
  --output-dir "$MULTISPECIES_M6A_DATA_DIR" \
  2>&1 | tee "$RNA_MAMBA_DATA_ROOT/processed/prepare_multispecies_m6a.log"
```

The script writes only five compact artifacts:

```text
multispecies_m6a_full_transcript/
  train.jsonl.gz
  val.jsonl.gz
  test.jsonl.gz
  stats.json
  README.md
```

The raw archives remain unchanged. Preparation is atomic: a failed audit does
not leave a partly built training directory.

## Verified 80/10/10 split

At the formal 10,240-nt cap, longer records are counted and excluded without
truncation. The full delivered archive produced the following verified model
splits:

| Split | All labelled transcripts | Strictly reliable and length <=10,240 nt | Positive m6A used by the model |
| --- | ---: | ---: | ---: |
| Train | 71,568 | 70,891 | 668,173 |
| Validation | 8,939 | 8,851 | 83,197 |
| Test | 8,969 | 8,871 | 83,431 |

The model-ready training split contains 55,490,725 candidate adenosines and
668,173 positive m6A sites, with a measured negative-to-positive ratio of
82.0484. Exact counts are also written to `stats.json`, and the data module
calculates the weight again from the records it actually loads.

Audit before launching training:

```bash
python - "$MULTISPECIES_M6A_DATA_DIR/stats.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["schema_version"] == 3
expected = {
    "pan_troglodytes",
    "arabidopsis_thaliana",
    "saccharomyces_cerevisiae",
    "macaca_mulatta",
    "sus_scrofa",
    "rattus_norvegicus",
}
assert set(d["species"]) == expected
assert d["splitting"]["leaking_genes"] == 0
assert d["splitting"]["leaking_exact_sequences"] == 0
for split in ("train", "val", "test"):
    row = d["combined_splits"][split]
    assert row["model_10240_training_transcripts"] > 0
    print(
        split,
        "transcripts=", row["model_10240_training_transcripts"],
        "positive_m6a=", row["model_10240_positive_m6a"],
    )
print("six-species m6A audit: PASS")
PY
```

To create a separate eight-species extension, explicitly add human and mouse:

```bash
python scripts/prepare_multispecies_m6a.py \
  --data-root "$RNA_MAMBA_DATA_ROOT" \
  --output-dir "$RNA_MAMBA_DATA_ROOT/processed/eight_species_m6a_full_transcript" \
  --species \
    pan_troglodytes arabidopsis_thaliana saccharomyces_cerevisiae \
    macaca_mulatta sus_scrofa rattus_norvegicus \
    homo_sapiens mus_musculus
```

## Fine-tune from the final MLM checkpoint

The task is full-model fine-tuning, not a new pretraining run. Start from the
best validation-loss checkpoint produced by RNA MLM pretraining:

The formal eight-GPU launcher has no user-specific filesystem paths. Set the
three paths for the current cluster and run only the fine-tuning stage:

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

export MULTISPECIES_M6A_DATA_DIR=/path/to/rna_mamba_data/processed/multispecies_m6a_full_transcript
export PRETRAIN_CKPT=/path/to/pretraining/checkpoints_best/val_loss.ckpt
export RUN_DIR=/path/to/runs/multispecies_m6a_full_transcript

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_DEVICES=8
export BATCH_SIZE=1
export GRAD_ACCUM=2
export NUM_WORKERS=8
export FINETUNE_EPOCHS=2
export PRECISION=bf16

bash scripts/run_multispecies_m6a_8gpu.sh
```

The launcher enforces an effective global batch of 16, checks the checkpoint
and all four processed-data files before allocating model memory, records the
exact Git commit and GPU inventory, and writes the best validation-AP model to
`$RUN_DIR/checkpoints_best/val_m6a_ap.ckpt`. The formal configuration uses
BF16, `residual_in_fp32=true`, gradient clipping, and full-model fine-tuning.
Start with two epochs and report the checkpoint with maximum
`val/m6a_average_precision`; previous small-corpus runs showed that later
epochs can overfit.
