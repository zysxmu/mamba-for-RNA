"""Build one audited, training-ready multi-species m6A dataset.

The formal task uses the six species delivered in the m6A ZIP archive. Human
and mouse tables are also supported as optional extensions. The six-species
labels are split between the eukaryotic mRNA ZIP archives and the m6A ZIP
archive. This script joins those sources without
extracting the large archives, validates complete 5'UTR+CDS+3'UTR sequences,
checks that every positive call is an adenosine, and writes one portable set of
gene-disjoint train/validation/test JSONL files.

The output is consumed directly by the full-transcript m6A data loader.  Each
record retains its species and transcript-region metadata so evaluation can be
stratified by species and by 5'UTR/CDS/3'UTR.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, NamedTuple, Optional, Sequence, TextIO


SCHEMA_VERSION = 3
SPLITS = ("train", "val", "test")
RNA_ALPHABET = frozenset("ACGUN")
FORMAL_SIX_SPECIES = (
    "pan_troglodytes",
    "arabidopsis_thaliana",
    "saccharomyces_cerevisiae",
    "macaca_mulatta",
    "sus_scrofa",
    "rattus_norvegicus",
)


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _parse_bool(value: object, field: str, transcript_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{transcript_id}: {field} must be True or False, got {value!r}")


def _rna_sequence(value: object) -> str:
    sequence = str(value).strip().upper().replace("T", "U")
    return "".join(base if base in RNA_ALPHABET else "N" for base in sequence)


def _ambiguous_base_count(value: object) -> int:
    sequence = str(value).strip().upper().replace("T", "U")
    return sum(base not in RNA_ALPHABET for base in sequence)


def _stable_gene_split(
    species: str,
    gene_id: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> str:
    normalized_gene = gene_id.strip().split(".", 1)[0]
    key = f"{species}:{normalized_gene}"
    return _stable_key_split(key, seed, train_fraction, val_fraction)


def _stable_key_split(
    key: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


class DisjointSet:
    """Union gene groups that share an exactly identical input sequence."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@contextmanager
def _open_path_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with path.open("rt", encoding="utf-8", newline="") as handle:
            yield handle


@contextmanager
def _open_zip_text(zip_path: Path, member: str, *, gzipped: bool) -> Iterator[TextIO]:
    with zipfile.ZipFile(zip_path) as archive:
        try:
            binary = archive.open(member)
        except KeyError as exc:
            raise FileNotFoundError(f"Missing {member!r} in {zip_path}") from exc
        with binary:
            if gzipped:
                with gzip.open(binary, "rt", encoding="utf-8", newline="") as handle:
                    yield handle
            else:
                with io.TextIOWrapper(binary, encoding="utf-8", newline="") as handle:
                    yield handle


@dataclass(frozen=True)
class SourceSpec:
    species: str
    scientific_name: str
    taxid: Optional[int]
    transcript_path: Optional[Path] = None
    transcript_zip: Optional[Path] = None
    transcript_member: Optional[str] = None
    mask_path: Optional[Path] = None
    mask_zip: Optional[Path] = None
    mask_member: Optional[str] = None
    exon_map: Optional[Path] = None
    positive_transcripts_only: bool = False

    @contextmanager
    def open_transcripts(self) -> Iterator[TextIO]:
        if self.transcript_path is not None:
            with _open_path_text(self.transcript_path) as handle:
                yield handle
            return
        if self.transcript_zip is None or self.transcript_member is None:
            raise ValueError(f"{self.species}: transcript source is incomplete")
        with _open_zip_text(self.transcript_zip, self.transcript_member, gzipped=False) as handle:
            yield handle

    @contextmanager
    def open_masks(self) -> Iterator[TextIO]:
        if self.mask_path is not None:
            with _open_path_text(self.mask_path) as handle:
                yield handle
            return
        if self.mask_zip is None or self.mask_member is None:
            raise ValueError(f"{self.species}: mask source is incomplete")
        with _open_zip_text(self.mask_zip, self.mask_member, gzipped=True) as handle:
            yield handle

    def portable_sources(self, data_root: Path) -> Mapping[str, Optional[str]]:
        def relative(path: Optional[Path]) -> Optional[str]:
            if path is None:
                return None
            try:
                return path.resolve().relative_to(data_root.resolve()).as_posix()
            except ValueError:
                return str(path.resolve())

        return {
            "transcript_path": relative(self.transcript_path),
            "transcript_zip": relative(self.transcript_zip),
            "transcript_member": self.transcript_member,
            "mask_path": relative(self.mask_path),
            "mask_zip": relative(self.mask_zip),
            "mask_member": self.mask_member,
            "exon_map": relative(self.exon_map),
        }


class MaskRecord(NamedTuple):
    length: int
    positive_positions: tuple[int, ...]
    cds_boundary_reliable: Optional[bool]
    mrna_coordinate_system_reliable: Optional[bool]
    qc_status: str
    qc_flags: str


def _canonical_sources(data_root: Path) -> list[SourceSpec]:
    data_root = data_root.resolve()
    euk = data_root / "pretraining" / "coding_rna" / "eukaryote"
    m6a_zip = (
        data_root
        / "finetuning"
        / "m6a"
        / "multispecies"
        / "m6A_modification_dataset.zip"
    )
    direct = [
        SourceSpec(
            species="homo_sapiens",
            scientific_name="Homo sapiens",
            taxid=9606,
            transcript_path=data_root
            / "pretraining/coding_rna/human/human_transcript_master.csv.gz",
            mask_path=data_root
            / "finetuning/m6a/human/human_m6a_nt_mask_full_mrna.csv.gz",
            exon_map=data_root
            / "finetuning/m6a/human/human_exon_coordinate_map.csv.gz",
        ),
        SourceSpec(
            species="mus_musculus",
            scientific_name="Mus musculus",
            taxid=10090,
            transcript_path=data_root
            / "pretraining/coding_rna/mouse/mouse_transcript_master.csv.gz",
            mask_path=data_root
            / "finetuning/m6a/mouse/mouse_m6a_nt_mask_full_mrna.csv.gz",
            exon_map=data_root
            / "finetuning/m6a/mouse/mouse_exon_coordinate_map.csv.gz",
        ),
    ]
    archive_species = [
        ("pan_troglodytes", "Pan troglodytes", 9598, 1),
        ("arabidopsis_thaliana", "Arabidopsis thaliana", 3702, 1),
        ("saccharomyces_cerevisiae", "Saccharomyces cerevisiae", 4932, 1),
        ("macaca_mulatta", "Macaca mulatta", 9544, 2),
        ("sus_scrofa", "Sus scrofa", 9823, 3),
        ("rattus_norvegicus", "Rattus norvegicus", 10116, 4),
    ]
    return direct + [
        SourceSpec(
            species=species,
            scientific_name=scientific_name,
            taxid=taxid,
            transcript_zip=euk / f"eukaryote_mRNA_dataset_part{part}.zip",
            transcript_member=f"processed/{species}/transcripts.tsv",
            mask_zip=m6a_zip,
            mask_member=f"{species}/m6a_nt_mask_full_mrna.csv.gz",
            positive_transcripts_only=True,
        )
        for species, scientific_name, taxid, part in archive_species
    ]


def _check_sources(sources: Sequence[SourceSpec]) -> None:
    missing = []
    for source in sources:
        for path in (
            source.transcript_path,
            source.transcript_zip,
            source.mask_path,
            source.mask_zip,
            source.exon_map,
        ):
            if path is not None and not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(sorted(set(missing))))


def _read_masks(source: SourceSpec) -> tuple[Dict[str, MaskRecord], Counter]:
    _raise_csv_field_limit()
    records: Dict[str, MaskRecord] = {}
    audit: Counter = Counter()
    with source.open_masks() as handle:
        reader = csv.DictReader(handle)
        required = {"transcript_id", "transcript_length", "m6a_nt_mask"}
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError(
                f"{source.species}: mask table is missing {sorted(required - fields)}"
            )
        has_explicit_reliability = {
            "cds_boundary_reliable",
            "mrna_coordinate_system_reliable",
        }.issubset(fields)
        for row_number, row in enumerate(reader, start=2):
            transcript_id = row["transcript_id"].strip()
            if not transcript_id:
                raise ValueError(f"{source.species}: mask row {row_number} has no transcript_id")
            if transcript_id in records:
                raise ValueError(f"{source.species}: duplicate mask for {transcript_id}")
            length = int(row["transcript_length"])
            tokens = row["m6a_nt_mask"].split(",")
            if len(tokens) != length:
                raise ValueError(
                    f"{source.species}/{transcript_id}: mask length {len(tokens)} != {length}"
                )
            invalid = set(tokens) - {"0", "1"}
            if invalid:
                raise ValueError(
                    f"{source.species}/{transcript_id}: invalid mask values {sorted(invalid)}"
                )
            positives = tuple(index for index, value in enumerate(tokens) if value == "1")
            if not positives:
                audit["transcripts_with_zero_positive"] += 1
            qc_status = str(row.get("qc_status", "PASS")).strip().upper() or "PASS"
            qc_flags = str(row.get("qc_flags", "")).strip()
            records[transcript_id] = MaskRecord(
                length=length,
                positive_positions=positives,
                cds_boundary_reliable=(
                    _parse_bool(row["cds_boundary_reliable"], "cds_boundary_reliable", transcript_id)
                    if has_explicit_reliability
                    else None
                ),
                mrna_coordinate_system_reliable=(
                    _parse_bool(
                        row["mrna_coordinate_system_reliable"],
                        "mrna_coordinate_system_reliable",
                        transcript_id,
                    )
                    if has_explicit_reliability
                    else None
                ),
                qc_status=qc_status,
                qc_flags=qc_flags,
            )
            audit["mask_rows"] += 1
            audit[f"mask_qc_{qc_status.lower()}"] += 1
    if not records:
        raise ValueError(f"{source.species}: mask table is empty")
    return records, audit


def _normalise_transcript(
    source: SourceSpec,
    row: Mapping[str, str],
    mask: MaskRecord,
) -> tuple[Mapping[str, object], bool, bool]:
    transcript_id = row["transcript_id"].strip()
    gene_id = row["gene_id"].strip()
    if not gene_id:
        raise ValueError(f"{source.species}/{transcript_id}: empty gene_id")
    sequence_field = "transcript_sequence" if "transcript_sequence" in row else "mrna_sequence"
    ambiguous_nucleotides_replaced = _ambiguous_base_count(row[sequence_field])
    sequence = _rna_sequence(row[sequence_field])
    declared_length = int(row["transcript_length"])
    if len(sequence) != declared_length or declared_length != mask.length:
        raise ValueError(
            f"{source.species}/{transcript_id}: sequence, table, and mask lengths disagree"
        )
    if any(sequence[position] != "A" for position in mask.positive_positions):
        raise ValueError(
            f"{source.species}/{transcript_id}: positive m6A mask falls on a non-A base"
        )

    coordinates = {
        field: int(row[field])
        for field in (
            "utr5_start",
            "utr5_end",
            "cds_start",
            "cds_end",
            "utr3_start",
            "utr3_end",
        )
    }
    lengths = {
        field: int(row[field]) for field in ("utr5_length", "cds_length", "utr3_length")
    }
    boundaries = [
        coordinates["utr5_start"],
        coordinates["utr5_end"],
        coordinates["cds_start"],
        coordinates["cds_end"],
        coordinates["utr3_start"],
        coordinates["utr3_end"],
    ]
    if not (
        boundaries[0] == 0
        and boundaries[1] == boundaries[2]
        and boundaries[3] == boundaries[4]
        and boundaries[5] == declared_length
        and boundaries == sorted(boundaries)
    ):
        raise ValueError(
            f"{source.species}/{transcript_id}: invalid full-mRNA region boundaries"
        )
    expected_lengths = {
        "utr5_length": coordinates["utr5_end"] - coordinates["utr5_start"],
        "cds_length": coordinates["cds_end"] - coordinates["cds_start"],
        "utr3_length": coordinates["utr3_end"] - coordinates["utr3_start"],
    }
    if lengths != expected_lengths:
        raise ValueError(f"{source.species}/{transcript_id}: region lengths disagree")
    region_sequences = {
        "utr5": _rna_sequence(row["utr5_sequence"]),
        "cds": _rna_sequence(row["cds_sequence"]),
        "utr3": _rna_sequence(row["utr3_sequence"]),
    }
    if sequence != region_sequences["utr5"] + region_sequences["cds"] + region_sequences["utr3"]:
        raise ValueError(
            f"{source.species}/{transcript_id}: full mRNA != 5'UTR + CDS + 3'UTR"
        )

    if mask.mrna_coordinate_system_reliable is not None:
        master_mrna_reliable = _parse_bool(
            row["mrna_coordinate_system_reliable"],
            "mrna_coordinate_system_reliable",
            transcript_id,
        )
        master_cds_reliable = _parse_bool(
            row["cds_boundary_reliable"], "cds_boundary_reliable", transcript_id
        )
        if master_mrna_reliable != mask.mrna_coordinate_system_reliable:
            raise ValueError(
                f"{source.species}/{transcript_id}: mRNA reliability differs between sources"
            )
        if master_cds_reliable != mask.cds_boundary_reliable:
            raise ValueError(
                f"{source.species}/{transcript_id}: CDS reliability differs between sources"
            )
        mrna_reliable = master_mrna_reliable
        cds_reliable = master_cds_reliable
        qc_status = "PASS" if mrna_reliable and cds_reliable else "WARN"
        qc_flags = ""
    else:
        transcript_qc = str(row.get("qc_status", mask.qc_status)).strip().upper() or "PASS"
        if transcript_qc != mask.qc_status:
            raise ValueError(
                f"{source.species}/{transcript_id}: QC status differs between sources"
            )
        qc_status = transcript_qc
        qc_flags = str(row.get("qc_flags", mask.qc_flags)).strip()
        mrna_reliable = qc_status != "FAIL"
        cds_reliable = mrna_reliable and _parse_bool(
            row.get("cds_match", "False"), "cds_match", transcript_id
        )

    cds = region_sequences["cds"]
    record = {
        "species": source.species,
        "scientific_name": source.scientific_name,
        "taxid": source.taxid,
        "transcript_id": transcript_id,
        "gene_id": gene_id,
        "sequence": sequence,
        "m6a_positions": mask.positive_positions,
        "mrna_coordinate_system_reliable": mrna_reliable,
        "cds_boundary_reliable": cds_reliable,
        "source_qc_status": qc_status,
        "source_qc_flags": qc_flags,
        "ambiguous_nucleotides_replaced_with_N": ambiguous_nucleotides_replaced,
        **coordinates,
        **lengths,
        "cds_starts_with_atg": cds.startswith("AUG"),
        "cds_ends_with_stop_codon": cds.endswith(("UAA", "UAG", "UGA")),
        "cds_length_multiple_of_3": len(cds) % 3 == 0,
    }
    return record, mrna_reliable, cds_reliable


def _update_stats(stats: Counter, record: Mapping[str, object]) -> None:
    sequence = str(record["sequence"])
    positions = tuple(record["m6a_positions"])
    stats["transcripts"] += 1
    stats["nucleotides"] += len(sequence)
    stats["max_transcript_length"] = max(stats["max_transcript_length"], len(sequence))
    stats["candidate_adenosines"] += sequence.count("A")
    stats["positive_m6a"] += len(positions)
    stats["negative_adenosines"] += sequence.count("A") - len(positions)
    stats["ambiguous_nucleotides_replaced_with_N"] += int(
        record.get("ambiguous_nucleotides_replaced_with_N", 0)
    )
    stats["mrna_coordinate_system_reliable"] += int(
        bool(record["mrna_coordinate_system_reliable"])
    )
    stats["cds_boundary_reliable"] += int(bool(record["cds_boundary_reliable"]))
    stats["full_transcript_training_eligible"] += int(
        bool(record["mrna_coordinate_system_reliable"])
        and bool(record["cds_boundary_reliable"])
    )
    reliable = bool(record["mrna_coordinate_system_reliable"]) and bool(
        record["cds_boundary_reliable"]
    )
    if len(sequence) <= 10240:
        stats["length_le_10240"] += 1
        if reliable:
            stats["model_10240_training_transcripts"] += 1
            stats["model_10240_nucleotides"] += len(sequence)
            stats["model_10240_candidate_adenosines"] += sequence.count("A")
            stats["model_10240_positive_m6a"] += len(positions)
            stats["model_10240_negative_adenosines"] += sequence.count("A") - len(positions)
    else:
        stats["overlength_gt_10240"] += 1
    for region in ("utr5", "cds", "utr3"):
        start = int(record[f"{region}_start"])
        end = int(record[f"{region}_end"])
        stats[f"{region}_nucleotides"] += end - start
        stats[f"{region}_candidate_adenosines"] += sequence[start:end].count("A")
        stats[f"{region}_positive_m6a"] += bisect.bisect_left(
            positions, end
        ) - bisect.bisect_left(positions, start)


def _finalise_stats(stats: Counter, genes: set[str]) -> Mapping[str, object]:
    result = dict(stats)
    # Keep the report schema stable even when a caller deliberately requests a
    # zero-sized split (for example, a unit test with ``val_fraction=0``).
    # Consumers should be able to read ``transcripts`` and the other counters
    # without special-casing an empty Counter.
    for key in (
        "transcripts",
        "nucleotides",
        "max_transcript_length",
        "candidate_adenosines",
        "positive_m6a",
        "negative_adenosines",
        "ambiguous_nucleotides_replaced_with_N",
        "mrna_coordinate_system_reliable",
        "cds_boundary_reliable",
        "full_transcript_training_eligible",
        "length_le_10240",
        "model_10240_training_transcripts",
        "model_10240_nucleotides",
        "model_10240_candidate_adenosines",
        "model_10240_positive_m6a",
        "model_10240_negative_adenosines",
        "overlength_gt_10240",
        "utr5_nucleotides",
        "utr5_candidate_adenosines",
        "utr5_positive_m6a",
        "cds_nucleotides",
        "cds_candidate_adenosines",
        "cds_positive_m6a",
        "utr3_nucleotides",
        "utr3_candidate_adenosines",
        "utr3_positive_m6a",
    ):
        result.setdefault(key, 0)
    result["genes"] = len(genes)
    candidates = stats["candidate_adenosines"]
    positives = stats["positive_m6a"]
    result["positive_fraction_among_adenosines"] = positives / candidates if candidates else 0.0
    result["recommended_pos_weight"] = (
        stats["negative_adenosines"] / positives if positives else 1.0
    )
    model_positives = stats["model_10240_positive_m6a"]
    result["model_10240_recommended_pos_weight"] = (
        stats["model_10240_negative_adenosines"] / model_positives
        if model_positives
        else 1.0
    )
    return result


def _validate_exon_map(path: Path, lengths: Mapping[str, int]) -> Mapping[str, int]:
    stats: Counter = Counter()
    previous_end: Dict[str, int] = {}
    seen = set()
    with _open_path_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {
            "transcript_id",
            "genomic_start_1based",
            "genomic_end_1based",
            "tx_start_0based",
            "tx_end_0based",
            "exon_length",
        }
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError(f"Exon map is missing {sorted(required - fields)}")
        for row in reader:
            transcript_id = row["transcript_id"].strip()
            if transcript_id not in lengths:
                continue
            tx_start = int(row["tx_start_0based"])
            tx_end = int(row["tx_end_0based"])
            exon_length = int(row["exon_length"])
            genomic_length = int(row["genomic_end_1based"]) - int(
                row["genomic_start_1based"]
            ) + 1
            if tx_end - tx_start != exon_length or genomic_length != exon_length:
                raise ValueError(f"{transcript_id}: inconsistent exon length")
            if tx_start != previous_end.get(transcript_id, 0):
                raise ValueError(f"{transcript_id}: non-contiguous transcript exon map")
            previous_end[transcript_id] = tx_end
            seen.add(transcript_id)
            stats["rows"] += 1
    missing = set(lengths) - seen
    if missing:
        raise ValueError(f"Exon map is missing {len(missing)} labelled transcripts")
    bad = [key for key, value in lengths.items() if previous_end.get(key) != value]
    if bad:
        raise ValueError(f"Exon map does not cover {len(bad)} complete transcripts")
    stats["transcripts"] = len(seen)
    return dict(stats)


def prepare_multispecies_dataset(
    data_root: Path,
    output_dir: Path,
    *,
    seed: int = 2357,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    selected_species: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[SourceSpec]] = None,
) -> Mapping[str, object]:
    """Validate all selected species and write one combined dataset."""

    if train_fraction < 0 or val_fraction < 0 or train_fraction + val_fraction > 1:
        raise ValueError("Invalid split fractions")
    data_root = Path(data_root).resolve()
    source_list = list(sources if sources is not None else _canonical_sources(data_root))
    if selected_species is None and sources is None:
        selected_species = FORMAL_SIX_SPECIES
    if selected_species:
        selected = set(selected_species)
        source_list = [source for source in source_list if source.species in selected]
        missing_species = selected - {source.species for source in source_list}
        if missing_species:
            raise ValueError(f"Unknown species: {sorted(missing_species)}")
    if not source_list:
        raise ValueError("No species selected")
    _check_sources(source_list)

    output_dir = Path(output_dir).resolve()
    build_dir = output_dir.with_name(output_dir.name + ".building")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    spool_path = build_dir / "validated_records.jsonl"
    components = DisjointSet()
    gene_nodes = set()
    species_reports = {}

    try:
        with spool_path.open("w", encoding="utf-8", newline="\n") as spool:
            for source_index, source in enumerate(source_list, start=1):
                print(
                    f"[{source_index}/{len(source_list)}] {source.species}: reading masks",
                    flush=True,
                )
                masks, source_audit = _read_masks(source)
                mask_lengths = {key: value.length for key, value in masks.items()}
                matched = set()
                with source.open_transcripts() as handle:
                    delimiter = "\t" if source.transcript_member is not None else ","
                    reader = csv.DictReader(handle, delimiter=delimiter)
                    fields = set(reader.fieldnames or [])
                    required = {
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
                    }
                    if not required.issubset(fields) or not (
                        {"transcript_sequence", "mrna_sequence"} & fields
                    ):
                        raise ValueError(
                            f"{source.species}: transcript table has an unsupported schema"
                        )
                    for row in reader:
                        transcript_id = row["transcript_id"].strip()
                        mask = masks.get(transcript_id)
                        if mask is None:
                            continue
                        if transcript_id in matched:
                            raise ValueError(
                                f"{source.species}: duplicate transcript row for {transcript_id}"
                            )
                        record, _, _ = _normalise_transcript(source, row, mask)
                        gene_id = str(record["gene_id"]).strip().split(".", 1)[0]
                        gene_key = f"{source.species}:{gene_id}"
                        gene_node = f"gene:{gene_key}"
                        sequence_node = "sequence:" + hashlib.sha256(
                            str(record["sequence"]).encode("utf-8")
                        ).hexdigest()
                        components.union(gene_node, sequence_node)
                        gene_nodes.add(gene_node)
                        spool.write(json.dumps(record, separators=(",", ":")) + "\n")
                        matched.add(transcript_id)
                missing = set(masks) - matched
                if missing:
                    raise ValueError(
                        f"{source.species}: transcript source is missing {len(missing)} masks"
                    )
                exon_audit = (
                    _validate_exon_map(source.exon_map, mask_lengths)
                    if source.exon_map is not None
                    else None
                )
                species_reports[source.species] = {
                    "scientific_name": source.scientific_name,
                    "taxid": source.taxid,
                    "source_selection": (
                        "transcripts_with_at_least_one_observed_m6a"
                        if source.positive_transcripts_only
                        else "all_rows_in_provided_mask_table"
                    ),
                    "sources": source.portable_sources(data_root),
                    "mask_audit": dict(source_audit),
                    "exon_map_audit": exon_audit,
                }
                print(
                    f"[{source.species}] matched={len(matched):,} "
                    f"positive={sum(item.positive_positions != () for item in masks.values()):,}",
                    flush=True,
                )
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise

    component_gene_keys: Dict[str, str] = {}
    for gene_node in gene_nodes:
        root = components.find(gene_node)
        gene_key = gene_node.removeprefix("gene:")
        current = component_gene_keys.get(root)
        if current is None or gene_key < current:
            component_gene_keys[root] = gene_key

    handles = {
        split: gzip.open(
            build_dir / f"{split}.jsonl.gz",
            "wt",
            encoding="utf-8",
            newline="\n",
            compresslevel=6,
        )
        for split in SPLITS
    }
    total_stats = {split: Counter() for split in SPLITS}
    total_genes = {split: set() for split in SPLITS}
    species_stats = {
        species: {split: Counter() for split in SPLITS} for species in species_reports
    }
    species_genes = {
        species: {split: set() for split in SPLITS} for species in species_reports
    }
    all_gene_splits = defaultdict(set)
    sequence_splits = defaultdict(set)
    sequence_record_counts = Counter()
    try:
        with spool_path.open("r", encoding="utf-8") as spool:
            for line in spool:
                record = json.loads(line)
                species = str(record["species"])
                gene_id = str(record["gene_id"]).strip().split(".", 1)[0]
                gene_key = f"{species}:{gene_id}"
                root = components.find(f"gene:{gene_key}")
                split_key = component_gene_keys[root]
                split = _stable_key_split(split_key, seed, train_fraction, val_fraction)
                sequence_hash = hashlib.sha256(
                    str(record["sequence"]).encode("utf-8")
                ).hexdigest()
                sequence_record_counts[sequence_hash] += 1

                handles[split].write(json.dumps(record, separators=(",", ":")) + "\n")
                _update_stats(total_stats[split], record)
                _update_stats(species_stats[species][split], record)
                total_genes[split].add(gene_key)
                species_genes[species][split].add(gene_key)
                all_gene_splits[gene_key].add(split)
                sequence_splits[sequence_hash].add(split)
    except Exception:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    else:
        for handle in handles.values():
            handle.close()
        spool_path.unlink()

    leaking_genes = [key for key, splits in all_gene_splits.items() if len(splits) > 1]
    leaking_sequences = [key for key, splits in sequence_splits.items() if len(splits) > 1]
    if leaking_genes or leaking_sequences:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise RuntimeError(
            "Split leakage detected: "
            f"genes={len(leaking_genes)} exact_sequences={len(leaking_sequences)}"
        )

    for species in species_reports:
        species_reports[species]["splits"] = {
            split: _finalise_stats(
                species_stats[species][split], species_genes[species][split]
            )
            for split in SPLITS
        }

    report: Mapping[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task": "multi-species nucleotide-level m6A classification at adenosines",
        "sequence_contract": {
            "sample": "one complete mRNA before optional model-length filtering",
            "composition": "mRNA = 5'UTR + CDS + 3'UTR",
            "coordinate_system": "0-based half-open transcript coordinates",
            "cds_start_definition": "index of the first CDS nucleotide in the complete mRNA",
        },
        "label_semantics": {
            "1": "methylated adenosine in the supplied nucleotide mask",
            "0": "unmethylated adenosine in the supplied nucleotide mask",
            "non_A": "excluded from loss and metrics",
        },
        "alphabet_normalisation": {
            "output_alphabet": "A/C/G/U/N",
            "T": "converted to U",
            "other_IUPAC_or_unknown_symbols": "converted to N and counted in split statistics",
        },
        "splitting": {
            "unit": (
                "connected components of species-qualified version-stripped gene_id "
                "and exact sequence hash"
            ),
            "seed": seed,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": 1.0 - train_fraction - val_fraction,
            "leaking_genes": 0,
            "leaking_exact_sequences": 0,
            "exact_sequence_groups": len(sequence_record_counts),
            "duplicate_exact_sequence_groups": sum(
                count > 1 for count in sequence_record_counts.values()
            ),
            "records_in_duplicate_exact_sequence_groups": sum(
                count for count in sequence_record_counts.values() if count > 1
            ),
        },
        "species": species_reports,
        "combined_splits": {
            split: _finalise_stats(total_stats[split], total_genes[split]) for split in SPLITS
        },
        "training_filter": {
            "require_mrna_coordinate_reliable": True,
            "require_cds_boundary_reliable": True,
            "max_sequence_length": 10240,
            "overlength_policy": "exclude_without_truncation",
            "note": "The provided full-transcript training configuration enforces these filters.",
        },
    }
    with (build_dir / "stats.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (build_dir / "README.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "# Multi-species full-transcript m6A dataset\n\n"
            "This directory is generated by `scripts/prepare_multispecies_m6a.py`. "
            "Each JSONL record contains one complete 5'UTR+CDS+3'UTR mRNA and "
            "the positions of methylated adenosines. The three splits are disjoint "
            "by species-qualified gene ID and exact input-sequence hash. See "
            "`stats.json` for source-level audits and exact counts.\n"
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(build_dir, output_dir)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--species",
        nargs="+",
        help=(
            "Canonical species names; default: the formal six-species m6A archive. "
            "Pass homo_sapiens and/or mus_musculus explicitly to include them."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = prepare_multispecies_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        selected_species=args.species,
    )
    summary = {
        "output_dir": str(args.output_dir.resolve()),
        "species": len(report["species"]),
        "combined_splits": report["combined_splits"],
    }
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
