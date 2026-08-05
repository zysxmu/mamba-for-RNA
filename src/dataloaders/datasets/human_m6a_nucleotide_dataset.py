"""Overlapping full-mRNA windows for nucleotide-level m6A classification."""

from __future__ import annotations

import bisect
import gzip
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

IGNORE_INDEX = -100.0


def window_starts(sequence_length: int, window_length: int, stride: int) -> List[int]:
    """Generate starts while guaranteeing that one window covers the tail."""

    if sequence_length <= 0:
        return []
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    if sequence_length <= window_length:
        return [0]
    starts = list(range(0, sequence_length - window_length + 1, stride))
    tail_start = sequence_length - window_length
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def window_ownership(
    sequence_length: int,
    window_length: int,
    stride: int,
) -> List[Tuple[int, int, int, int]]:
    """Return windows plus a non-overlapping loss interval for each window.

    Overlap supplies context, while midpoint boundaries make every nucleotide
    contribute to the training loss exactly once.
    """

    if stride > window_length:
        raise ValueError("stride must not exceed window_length")
    starts = window_starts(sequence_length, window_length, stride)
    if not starts:
        return []
    ends = [min(start + window_length, sequence_length) for start in starts]
    centers = [(start + end) / 2.0 for start, end in zip(starts, ends)]
    boundaries = [0]
    boundaries.extend(int((centers[index - 1] + centers[index]) / 2.0) for index in range(1, len(starts)))
    boundaries.append(sequence_length)

    layout = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        loss_start = boundaries[index]
        loss_end = boundaries[index + 1]
        if not (start <= loss_start <= loss_end <= end):
            raise RuntimeError("Window overlap is insufficient to define unique loss ownership")
        layout.append((start, end, loss_start, loss_end))
    return layout


class HumanM6ANucleotideDataset(torch.utils.data.Dataset):
    """Lazily construct fixed-length token windows and A-only binary labels."""

    def __init__(
        self,
        jsonl_file,
        tokenizer,
        window_length: int = 1024,
        stride: int = 512,
        require_mrna_coordinate_reliable: bool = True,
        max_transcripts: Optional[int] = None,
        max_windows: Optional[int] = None,
    ):
        self.jsonl_file = Path(jsonl_file)
        self.tokenizer = tokenizer
        self.window_length = int(window_length)
        self.max_length = self.window_length
        self.stride = int(stride)
        self.require_mrna_coordinate_reliable = bool(require_mrna_coordinate_reliable)
        if self.window_length <= 0 or self.stride <= 0 or self.stride > self.window_length:
            raise ValueError("Require 0 < stride <= window_length")

        self.records: List[Mapping[str, object]] = []
        opener = gzip.open if self.jsonl_file.suffix == ".gz" else open
        with opener(self.jsonl_file, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate_record(record, line_number)
                if self.require_mrna_coordinate_reliable and not bool(
                    record["mrna_coordinate_system_reliable"]
                ):
                    continue
                self.records.append(record)
                if max_transcripts is not None and len(self.records) >= int(max_transcripts):
                    break

        self.window_index: List[Tuple[int, int, int, int, int]] = []
        for record_index, record in enumerate(self.records):
            sequence = str(record["sequence"])
            for start, end, loss_start, loss_end in window_ownership(
                len(sequence), self.window_length, self.stride
            ):
                if "A" in sequence[loss_start:loss_end]:
                    self.window_index.append((record_index, start, end, loss_start, loss_end))
        if max_windows is not None:
            if int(max_windows) <= 0:
                raise ValueError("max_windows must be positive")
            self.window_index = self.window_index[: int(max_windows)]

        self.candidate_count, self.positive_count = self._label_counts()
        self.negative_count = self.candidate_count - self.positive_count
        self.positive_weight = self.negative_count / max(1, self.positive_count)

        vocab = tokenizer.get_vocab()
        self._byte_to_token = np.full(256, int(tokenizer.unk_token_id), dtype=np.int64)
        for base in "ACGUN":
            self._byte_to_token[ord(base)] = int(vocab[base])

    @staticmethod
    def _validate_record(record: Mapping[str, object], line_number: int) -> None:
        required = {
            "transcript_id",
            "gene_id",
            "sequence",
            "m6a_positions",
            "mrna_coordinate_system_reliable",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")
        sequence = str(record["sequence"])
        invalid = set(sequence) - set("ACGUN")
        if invalid:
            raise ValueError(f"Line {line_number} contains invalid RNA symbols: {sorted(invalid)}")
        positions = list(record["m6a_positions"])
        if positions != sorted(set(positions)):
            raise ValueError(f"Line {line_number} has unsorted or repeated m6A positions")
        if any(not isinstance(position, int) or position < 0 or position >= len(sequence) for position in positions):
            raise ValueError(f"Line {line_number} has an out-of-range m6A position")
        if any(sequence[position] != "A" for position in positions):
            raise ValueError(f"Line {line_number} maps m6A to a non-A base")

    def _label_counts(self) -> Tuple[int, int]:
        candidates = positives = 0
        for record_index, _, _, loss_start, loss_end in self.window_index:
            record = self.records[record_index]
            sequence = str(record["sequence"])
            positions = record["m6a_positions"]
            candidates += sequence[loss_start:loss_end].count("A")
            positives += bisect.bisect_left(positions, loss_end) - bisect.bisect_left(
                positions, loss_start
            )
        return candidates, positives

    def __len__(self) -> int:
        return len(self.window_index)

    def window_metadata(self, index: int) -> Dict[str, object]:
        record_index, start, end, loss_start, loss_end = self.window_index[index]
        record = self.records[record_index]
        return {
            "transcript_id": record["transcript_id"],
            "gene_id": record["gene_id"],
            "start": start,
            "end": end,
            "valid_length": end - start,
            "loss_start": loss_start,
            "loss_end": loss_end,
        }

    def __getitem__(self, index: int):
        record_index, start, end, loss_start, loss_end = self.window_index[index]
        record = self.records[record_index]
        sequence = str(record["sequence"])
        window = sequence[start:end]

        input_ids = torch.full(
            (self.window_length,), int(self.tokenizer.pad_token_id), dtype=torch.long
        )
        raw = np.frombuffer(window.encode("ascii"), dtype=np.uint8)
        input_ids[: len(window)] = torch.from_numpy(self._byte_to_token[raw].copy())

        labels = torch.full((self.window_length,), IGNORE_INDEX, dtype=torch.float32)
        owned = sequence[loss_start:loss_end]
        owned_raw = np.frombuffer(owned.encode("ascii"), dtype=np.uint8)
        candidate_local = np.flatnonzero(owned_raw == ord("A")) + (loss_start - start)
        if candidate_local.size:
            labels[torch.from_numpy(candidate_local.astype(np.int64))] = 0.0

        positions = record["m6a_positions"]
        left = bisect.bisect_left(positions, loss_start)
        right = bisect.bisect_left(positions, loss_end)
        for position in positions[left:right]:
            labels[int(position) - start] = 1.0

        return input_ids, labels
