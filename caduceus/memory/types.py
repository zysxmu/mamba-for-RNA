"""Typed outputs for the RNA memory modules."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class MemoryWriterOutput:
    """Output of a memory writer.

    Shapes:
        memory_slots: ``[batch, slots, d_mem]``
        slot_mask: ``[batch, slots]``
    """

    memory_slots: torch.Tensor
    slot_mask: torch.Tensor
    stats: Dict[str, torch.Tensor]


@dataclass
class MemoryReaderOutput:
    """Output of token-to-memory cross-attention."""

    memory_output: torch.Tensor
    stats: Dict[str, torch.Tensor]
    attention_weights: Optional[torch.Tensor] = None
