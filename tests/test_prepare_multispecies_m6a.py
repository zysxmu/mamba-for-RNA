import csv
import gzip
import io
import json
import zipfile
from pathlib import Path

from scripts.prepare_multispecies_m6a import SourceSpec, prepare_multispecies_dataset


REGION_FIELDS = [
    "transcript_id",
    "gene_id",
    "transcript_length",
    "utr5_start",
    "utr5_end",
    "cds_start",
    "cds_end",
    "utr3_start",
    "utr3_end",
    "utr5_length",
    "cds_length",
    "utr3_length",
    "utr5_sequence",
    "cds_sequence",
    "utr3_sequence",
]


def _csv_bytes(fieldnames, rows, delimiter=","):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_gzip(path: Path, fieldnames, rows):
    with gzip.open(path, "wb") as handle:
        handle.write(_csv_bytes(fieldnames, rows))


def _base_transcript(transcript_id: str):
    return {
        "transcript_id": transcript_id,
        "gene_id": "SHARED_GENE.1",
        "transcript_length": 9,
        "utr5_start": 0,
        "utr5_end": 2,
        "cds_start": 2,
        "cds_end": 8,
        "utr3_start": 8,
        "utr3_end": 9,
        "utr5_length": 2,
        "cds_length": 6,
        "utr3_length": 1,
        "utr5_sequence": "AA",
        "cds_sequence": "ATGTAA",
        "utr3_sequence": "A",
    }


def test_prepare_combines_direct_and_archived_species(tmp_path: Path):
    direct_master = tmp_path / "direct_master.csv.gz"
    direct_mask = tmp_path / "direct_mask.csv.gz"
    direct_row = {
        **_base_transcript("TX_DIRECT.1"),
        "transcript_sequence": "AAATGTAAA",
        "cds_boundary_reliable": "True",
        "mrna_coordinate_system_reliable": "True",
    }
    _write_gzip(
        direct_master,
        REGION_FIELDS
        + [
            "transcript_sequence",
            "cds_boundary_reliable",
            "mrna_coordinate_system_reliable",
        ],
        [direct_row],
    )
    _write_gzip(
        direct_mask,
        [
            "transcript_id",
            "transcript_length",
            "m6a_nt_mask",
            "cds_boundary_reliable",
            "mrna_coordinate_system_reliable",
        ],
        [
            {
                "transcript_id": "TX_DIRECT.1",
                "transcript_length": 9,
                "m6a_nt_mask": "1,0,0,0,0,0,0,0,1",
                "cds_boundary_reliable": "True",
                "mrna_coordinate_system_reliable": "True",
            }
        ],
    )

    transcript_zip = tmp_path / "transcripts.zip"
    archived_row = {
        **_base_transcript("TX_ARCHIVED.1"),
        "mrna_sequence": "AAATGTAAA",
        "qc_status": "PASS",
        "qc_flags": "",
        "cds_match": "True",
    }
    transcript_fields = REGION_FIELDS + ["mrna_sequence", "qc_status", "qc_flags", "cds_match"]
    with zipfile.ZipFile(transcript_zip, "w") as archive:
        archive.writestr(
            "processed/species_b/transcripts.tsv",
            _csv_bytes(transcript_fields, [archived_row], delimiter="\t"),
        )
    mask_zip = tmp_path / "masks.zip"
    archived_mask = _csv_bytes(
        ["transcript_id", "transcript_length", "m6a_nt_mask", "qc_status", "qc_flags"],
        [
            {
                "transcript_id": "TX_ARCHIVED.1",
                "transcript_length": 9,
                "m6a_nt_mask": "0,1,0,0,0,0,0,0,0",
                "qc_status": "PASS",
                "qc_flags": "",
            }
        ],
    )
    with zipfile.ZipFile(mask_zip, "w") as archive:
        archive.writestr("species_b/mask.csv.gz", gzip.compress(archived_mask))

    sources = [
        SourceSpec(
            species="species_a",
            scientific_name="Species a",
            taxid=1,
            transcript_path=direct_master,
            mask_path=direct_mask,
        ),
        SourceSpec(
            species="species_b",
            scientific_name="Species b",
            taxid=2,
            transcript_zip=transcript_zip,
            transcript_member="processed/species_b/transcripts.tsv",
            mask_zip=mask_zip,
            mask_member="species_b/mask.csv.gz",
            positive_transcripts_only=True,
        ),
    ]
    output = tmp_path / "processed"
    report = prepare_multispecies_dataset(
        tmp_path,
        output,
        train_fraction=0.5,
        val_fraction=0.0,
        sources=sources,
    )

    assert report["schema_version"] == 3
    assert sum(report["combined_splits"][split]["transcripts"] for split in ("train", "test")) == 2
    assert sum(report["combined_splits"][split]["genes"] for split in ("train", "test")) == 2
    assert sum(report["combined_splits"][split]["positive_m6a"] for split in ("train", "test")) == 3
    assert report["splitting"]["leaking_genes"] == 0
    assert report["splitting"]["leaking_exact_sequences"] == 0
    assert report["species"]["species_b"]["source_selection"].startswith(
        "transcripts_with_at_least_one"
    )

    records_by_split = {}
    for split in ("train", "val", "test"):
        with gzip.open(output / f"{split}.jsonl.gz", "rt", encoding="utf-8") as handle:
            records_by_split[split] = [json.loads(line) for line in handle]
    populated = [split for split, records in records_by_split.items() if records]
    assert len(populated) == 1, "identical sequences must be assigned to one split"
    records = records_by_split[populated[0]]
    assert {record["species"] for record in records} == {"species_a", "species_b"}
    assert all(record["sequence"] == "AAAUGUAAA" for record in records)
    assert all(record["cds_start"] == 2 and record["cds_end"] == 8 for record in records)
    assert (output / "stats.json").is_file()
    assert (output / "README.md").is_file()
