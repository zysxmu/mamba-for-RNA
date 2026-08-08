"""Data module for long-context MLM on complete mRNA and ncRNA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hydra.utils import to_absolute_path

from caduceus.tokenization_caduceus import CaduceusTokenizer
from src.dataloaders.base import SequenceDataset
from src.dataloaders.datasets.full_transcript_mlm_dataset import (
    FullTranscriptMLMDataset,
    collate_full_transcript_mlm,
)


class FullTranscriptRNAMLM(SequenceDataset):
    """Long-context MLM with split-safe complete mRNA and ncRNA samples."""

    _name_ = "full_transcript_rna_mlm"
    _collate_arg_names = ["attention_mask"]

    def __init__(
        self,
        _name_: str,
        data_dir,
        ncrna_fasta=None,
        include_ncrna: bool = True,
        tokenizer_name: str = "char",
        max_sequence_length: int = 10240,
        require_mrna_coordinate_reliable: bool = True,
        require_cds_boundary_reliable: bool = True,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        pad_to_multiple: int = 8,
        batch_size: int = 1,
        batch_size_eval: Optional[int] = None,
        shuffle: bool = True,
        split_seed: int = 2357,
        max_train_sequences: Optional[int] = None,
        max_val_sequences: Optional[int] = None,
        max_test_sequences: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(_name_=_name_, data_dir=to_absolute_path(str(data_dir)))
        if not mlm:
            raise ValueError("FullTranscriptRNAMLM requires mlm=true")
        self.ncrna_fasta = (
            None if ncrna_fasta is None else to_absolute_path(str(ncrna_fasta))
        )
        self.include_ncrna = bool(include_ncrna)
        self.tokenizer_name = tokenizer_name
        self.max_sequence_length = int(max_sequence_length)
        self.max_length = self.max_sequence_length
        self.require_mrna_coordinate_reliable = bool(require_mrna_coordinate_reliable)
        self.require_cds_boundary_reliable = bool(require_cds_boundary_reliable)
        self.mlm = True
        self.mlm_probability = float(mlm_probability)
        self.pad_to_multiple = int(pad_to_multiple)
        self.batch_size = int(batch_size)
        self.batch_size_eval = int(batch_size_eval or batch_size)
        self.shuffle = bool(shuffle)
        self.split_seed = int(split_seed)
        self.max_train_sequences = max_train_sequences
        self.max_val_sequences = max_val_sequences
        self.max_test_sequences = max_test_sequences
        self.tokenizer = None
        self.vocab_size = 0

    def _split_path(self, split: str) -> Path:
        compressed = Path(self.data_dir) / f"{split}.jsonl.gz"
        if compressed.exists():
            return compressed
        plain = Path(self.data_dir) / f"{split}.jsonl"
        if plain.exists():
            return plain
        raise FileNotFoundError(f"Missing {compressed} (or uncompressed {plain})")

    def setup(self, stage=None) -> None:
        if self.tokenizer_name != "char":
            raise NotImplementedError("Long-context RNA MLM supports only char tokenization")
        self.tokenizer = CaduceusTokenizer(
            model_max_length=self.max_sequence_length,
            add_special_tokens=False,
            padding_side="right",
        )
        self.vocab_size = len(self.tokenizer)
        common = {
            "tokenizer": self.tokenizer,
            "max_sequence_length": self.max_sequence_length,
            "ncrna_fasta": self.ncrna_fasta,
            "include_ncrna": self.include_ncrna,
            "require_mrna_coordinate_reliable": self.require_mrna_coordinate_reliable,
            "require_cds_boundary_reliable": self.require_cds_boundary_reliable,
            "mlm_probability": self.mlm_probability,
            "seed": self.split_seed,
        }
        self.dataset_train = FullTranscriptMLMDataset(
            self._split_path("train"),
            split="train",
            deterministic_mlm=False,
            max_sequences=self.max_train_sequences,
            **common,
        )
        self.dataset_val = FullTranscriptMLMDataset(
            self._split_path("val"),
            split="val",
            deterministic_mlm=True,
            max_sequences=self.max_val_sequences,
            **common,
        )
        self.dataset_test = FullTranscriptMLMDataset(
            self._split_path("test"),
            split="test",
            deterministic_mlm=True,
            max_sequences=self.max_test_sequences,
            **common,
        )
        for split, dataset in (
            ("train", self.dataset_train),
            ("val", self.dataset_val),
            ("test", self.dataset_test),
        ):
            print(
                "[FullTranscriptRNAMLM] "
                f"split={split} sequences={len(dataset)} sources={dataset.source_counts} "
                f"nucleotides={dataset.nucleotides} "
                f"excluded_overlength={dataset.excluded_overlength} "
                f"excluded_invalid={dataset.excluded_invalid} "
                f"excluded_unreliable_mrna={dataset.excluded_unreliable_mrna} "
                f"excluded_unreliable_cds={dataset.excluded_unreliable_cds}"
            )

    def _collate_fn(self, batch, *args, **kwargs):
        return collate_full_transcript_mlm(
            batch,
            pad_token_id=int(self.tokenizer.pad_token_id),
            pad_to_multiple=self.pad_to_multiple,
        )

    def train_dataloader(self, **kwargs: Any):
        kwargs.setdefault("drop_last", False)
        return self._dataloader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **kwargs,
        )

    def val_dataloader(self, **kwargs: Any):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._dataloader(
            self.dataset_val, batch_size=self.batch_size_eval, **kwargs
        )

    def test_dataloader(self, **kwargs: Any):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._dataloader(
            self.dataset_test, batch_size=self.batch_size_eval, **kwargs
        )
