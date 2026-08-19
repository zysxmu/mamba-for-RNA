from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import io
import json
import pickle
import struct
import sys
import zipfile
from pathlib import Path

import torch

from scripts.prepare_pretraining_5m import prepare_corpus

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dataloaders"
    / "datasets"
    / "indexed_rna_mlm_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("indexed_rna_mlm_dataset", DATASET_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = DATASET_MODULE
SPEC.loader.exec_module(DATASET_MODULE)
IndexedRNAMLMDataset = DATASET_MODULE.IndexedRNAMLMDataset


class DummyTokenizer:
    ids = {"A": 7, "C": 8, "G": 9, "U": 10, "N": 11}
    pad_token_id = 4
    mask_token_id = 3
    unk_token_id = 6

    def get_vocab(self):
        return self.ids


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def _fasta(records: list[tuple[str, str]]) -> bytes:
    return "".join(f">{identifier}\n{sequence}\n" for identifier, sequence in records).encode()


def _synthetic_bundle(path, *, include_mouse: bool = True) -> bytes:
    euk = _zip_bytes(
        {
            "euk/Species_A/mrna.fa": _fasta(
                [("e1", "ATGCAAAA"), ("e2", "TTTTCCCC"), ("e3", "GGGGAAAA")]
            ),
            "euk/Species_A/cds.fa": _fasta(
                [("e1", "ATGC"), ("e2", "TTCC"), ("e3", "GGAA"), ("e4", "ACACAC")]
            ),
        }
    )
    prok = _zip_bytes(
        {"prok/Species_B/cds.fa": _fasta([("p1", "CGCGCGCG")])}
    )
    human_text = io.StringIO()
    writer = csv.DictWriter(human_text, fieldnames=["transcript_id", "transcript_sequence"])
    writer.writeheader()
    writer.writerow({"transcript_id": "h1", "transcript_sequence": "AATTCCGG"})
    mouse_text = io.StringIO()
    writer = csv.DictWriter(mouse_text, fieldnames=["transcript_id", "transcript_sequence"])
    writer.writeheader()
    writer.writerow({"transcript_id": "m1", "transcript_sequence": "ACGUACGU"})
    writer.writerow({"transcript_id": "m2", "transcript_sequence": "UUUUGGGG"})

    ncrna = _fasta(
        [("n1", "AAAAUUUU"), ("n2", "CCCCGGGG"), ("n3", "AUCGAUCG"), ("n4", "NNNNAAAA")]
    )
    mouse_compressed = gzip.compress(mouse_text.getvalue().encode())
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("eukaryote_mRNA_dataset_part1.zip", euk)
        outer.writestr("prokaryote_mRNA_dataset.zip", prok)
        outer.writestr("human_transcript_master.csv.gz", gzip.compress(human_text.getvalue().encode()))
        if include_mouse:
            outer.writestr("mouse_transcript_master.csv.gz", mouse_compressed)
        outer.writestr("rnacentral_active_mRNA100_priority_3M.fasta.gz", gzip.compress(ncrna))
        # These labels are intentionally present but must not enter pretraining.
        outer.writestr("human_m6a_nt_mask_full_mrna.csv.gz", gzip.compress(b"mask\n"))
    return mouse_compressed


def test_prepare_and_read_indexed_five_million_style_corpus(tmp_path):
    bundle = tmp_path / "bundle.zip"
    output = tmp_path / "prepared"
    _synthetic_bundle(bundle)
    args = argparse.Namespace(
        bundle=str(bundle),
        output_dir=str(output),
        temp_dir=str(tmp_path),
        target_ncrna=4,
        target_coding=9,
        min_length=4,
        max_length=32,
        train_fraction=0.6,
        val_fraction=0.2,
        seed=2357,
        overwrite=False,
    )

    manifest = prepare_corpus(args)
    assert manifest["totals"]["records"] == 11
    assert manifest["totals"]["source_class_counts"] == {"ncRNA": 4, "coding": 7}
    assert manifest["corpus_contract"]["coding_primary_records"] == 7
    assert manifest["corpus_contract"]["coding_cds_view_fillers"] == 0
    assert manifest["corpus_contract"]["coding_shortfall_from_requested_target"] == 2
    assert manifest["corpus_contract"]["m6a_used"] is False
    assert sum(item["records"] for item in manifest["splits"].values()) == 11
    assert sum(
        item["source_type_counts"].get("mouse_full_mrna", 0)
        for item in manifest["splits"].values()
    ) == 2

    transcript_splits: dict[tuple[str, str], set[str]] = {}
    for split, info in manifest["splits"].items():
        with gzip.open(output / info["files"]["metadata"], "rt", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["source_class"] == "coding":
                    transcript_splits.setdefault(
                        (row["species"], row["record_id"]), set()
                    ).add(split)
    assert all(len(splits) == 1 for splits in transcript_splits.values())

    for split, info in manifest["splits"].items():
        offsets = (output / info["files"]["offsets"]).read_bytes()
        assert len(offsets) == (info["records"] + 1) * 8
        last_offset = struct.unpack("<Q", offsets[-8:])[0]
        assert last_offset == (output / info["files"]["sequences"]).stat().st_size
        assert (output / info["files"]["sources"]).stat().st_size == info["records"]

    nonempty_split = next(
        split for split, info in manifest["splits"].items() if info["records"]
    )
    tokenizer = DummyTokenizer()
    dataset = IndexedRNAMLMDataset(
        output,
        tokenizer,
        split=nonempty_split,
        mlm_probability=0.5,
        deterministic_mlm=True,
        seed=7,
    )
    inputs_a, labels_a = dataset[0]
    inputs_b, labels_b = dataset[0]
    assert torch.equal(inputs_a, inputs_b)
    assert torch.equal(labels_a, labels_b)
    assert (labels_a != tokenizer.pad_token_id).any()

    restored = pickle.loads(pickle.dumps(dataset))
    inputs_c, labels_c = restored[0]
    assert torch.equal(inputs_a, inputs_c)
    assert torch.equal(labels_a, labels_c)


def test_indexed_manifest_records_m6a_exclusion(tmp_path):
    bundle = tmp_path / "bundle.zip"
    output = tmp_path / "prepared"
    _synthetic_bundle(bundle)
    args = argparse.Namespace(
        bundle=str(bundle),
        output_dir=str(output),
        temp_dir=str(tmp_path),
        target_ncrna=4,
        target_coding=5,
        min_length=4,
        max_length=32,
        train_fraction=0.7,
        val_fraction=0.15,
        seed=1,
        overwrite=False,
    )
    prepare_corpus(args)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_contract"]["m6a_used"] is False
    assert all("m6a" not in key.lower() for key in manifest["audit"])


def test_prepare_accepts_external_mouse_master(tmp_path):
    bundle = tmp_path / "bundle.zip"
    mouse_master = tmp_path / "mouse_transcript_master.csv.gz"
    mouse_master.write_bytes(_synthetic_bundle(bundle, include_mouse=False))
    output = tmp_path / "prepared"
    args = argparse.Namespace(
        bundle=str(bundle),
        mouse_transcript_master=str(mouse_master),
        output_dir=str(output),
        temp_dir=str(tmp_path),
        target_ncrna=4,
        target_coding=7,
        min_length=4,
        max_length=32,
        train_fraction=0.7,
        val_fraction=0.15,
        seed=1,
        overwrite=False,
    )
    manifest = prepare_corpus(args)
    assert manifest["totals"]["source_class_counts"]["coding"] == 7
    assert manifest["input_mouse_transcript_master"] == str(mouse_master.resolve())
    assert sum(
        item["source_type_counts"].get("mouse_full_mrna", 0)
        for item in manifest["splits"].values()
    ) == 2
