import csv
import gzip
import importlib.util
import json
from pathlib import Path

import torch

from scripts.prepare_m6a_nucleotide import prepare_dataset


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dataloaders"
    / "datasets"
    / "human_m6a_nucleotide_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("human_m6a_nucleotide_dataset", DATASET_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DATASET_MODULE)
HumanM6ANucleotideDataset = DATASET_MODULE.HumanM6ANucleotideDataset
window_ownership = DATASET_MODULE.window_ownership


class DummyTokenizer:
    ids = {"A": 7, "C": 8, "G": 9, "U": 10, "N": 11}
    pad_token_id = 4
    unk_token_id = 6

    def get_vocab(self):
        return self.ids


def _write_gzip_csv(path: Path, fieldnames, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_window_ownership_partitions_each_position_once():
    layout = window_ownership(sequence_length=11, window_length=6, stride=3)
    assert [(item[0], item[1]) for item in layout] == [(0, 6), (3, 9), (5, 11)]
    owned = []
    for _, _, loss_start, loss_end in layout:
        owned.extend(range(loss_start, loss_end))
    assert owned == list(range(11))


def test_prepare_and_load_nucleotide_labels(tmp_path: Path):
    master = tmp_path / "transcript_master.csv.gz"
    masks = tmp_path / "m6a_nt_mask_full_mrna.csv.gz"
    exons = tmp_path / "exon_coordinate_map.csv.gz"

    _write_gzip_csv(
        master,
        ["transcript_id", "gene_id", "transcript_length", "transcript_sequence"],
        [
            {
                "transcript_id": "ENST1.1",
                "gene_id": "ENSG1.1",
                "transcript_length": 6,
                "transcript_sequence": "AATCGA",
            },
            {
                "transcript_id": "ENST2.1",
                "gene_id": "ENSG2.1",
                "transcript_length": 3,
                "transcript_sequence": "AAA",
            },
        ],
    )
    _write_gzip_csv(
        masks,
        [
            "transcript_id",
            "transcript_length",
            "m6a_nt_mask",
            "cds_boundary_reliable",
            "mrna_coordinate_system_reliable",
        ],
        [
            {
                "transcript_id": "ENST1.1",
                "transcript_length": 6,
                "m6a_nt_mask": "1,0,0,0,0,1",
                "cds_boundary_reliable": "True",
                "mrna_coordinate_system_reliable": "True",
            },
            {
                "transcript_id": "ENST2.1",
                "transcript_length": 3,
                "m6a_nt_mask": "0,1,0",
                "cds_boundary_reliable": "False",
                "mrna_coordinate_system_reliable": "False",
            },
        ],
    )
    _write_gzip_csv(
        exons,
        [
            "transcript_id",
            "genomic_start_1based",
            "genomic_end_1based",
            "tx_start_0based",
            "tx_end_0based",
            "exon_length",
        ],
        [
            {
                "transcript_id": "ENST1.1",
                "genomic_start_1based": 100,
                "genomic_end_1based": 105,
                "tx_start_0based": 0,
                "tx_end_0based": 6,
                "exon_length": 6,
            },
            {
                "transcript_id": "ENST2.1",
                "genomic_start_1based": 200,
                "genomic_end_1based": 202,
                "tx_start_0based": 0,
                "tx_end_0based": 3,
                "exon_length": 3,
            },
        ],
    )

    output_dir = tmp_path / "processed"
    report = prepare_dataset(
        transcript_master=master,
        mask_table=masks,
        exon_map=exons,
        output_dir=output_dir,
        train_fraction=1.0,
        val_fraction=0.0,
    )
    assert report["splits"]["train"]["transcripts"] == 2
    assert report["splits"]["train"]["candidate_adenosines"] == 6
    assert report["splits"]["train"]["positive_m6a"] == 3
    assert report["exon_map_audit"]["transcripts"] == 2

    dataset = HumanM6ANucleotideDataset(
        output_dir / "train.jsonl.gz",
        tokenizer=DummyTokenizer(),
        window_length=4,
        stride=2,
        require_mrna_coordinate_reliable=True,
    )
    assert len(dataset.records) == 1
    assert len(dataset) == 2
    assert dataset.candidate_count == 3
    assert dataset.positive_count == 2
    assert dataset.negative_count == 1

    labels = []
    for index in range(len(dataset)):
        input_ids, target = dataset[index]
        assert input_ids.shape == (4,)
        assert target.shape == (4,)
        labels.extend(target[target != DATASET_MODULE.IGNORE_INDEX].tolist())
    assert labels == [1.0, 0.0, 1.0]

    with gzip.open(output_dir / "train.jsonl.gz", "rt", encoding="utf-8") as handle:
        first = json.loads(next(handle))
    assert "T" not in first["sequence"]
    assert first["m6a_positions"] == [0, 5]
