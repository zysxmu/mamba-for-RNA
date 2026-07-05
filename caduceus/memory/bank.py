"""Differentiable, forward-local cross-layer memory bank."""

from typing import Optional, Tuple

import torch


class CrossLayerMemoryBank:
    """Collect memory slots created earlier in the current forward pass.

    Entries are intentionally not detached: gradients from later readers must
    reach the writer that produced each slot. The bank is recreated for every
    batch and therefore never shares information between unrelated RNA samples.
    """

    def __init__(self, max_slots: int, replacement: str = "fifo") -> None:
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        if replacement != "fifo":
            raise ValueError(f"Unsupported memory replacement strategy: {replacement}")
        self.max_slots = int(max_slots)
        self.replacement = replacement
        self._memory: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None

    def append(self, memory_slots: torch.Tensor, slot_mask: torch.Tensor) -> None:
        if memory_slots.ndim != 3:
            raise ValueError("memory_slots must have shape [B, S, D]")
        if slot_mask.shape != memory_slots.shape[:2]:
            raise ValueError("slot_mask must have shape [B, S]")

        slot_mask = slot_mask.to(device=memory_slots.device, dtype=torch.bool)
        if self._memory is None:
            memory = memory_slots
            mask = slot_mask
        else:
            if self._memory.shape[0] != memory_slots.shape[0]:
                raise ValueError("All memory writes in a forward pass must share the same batch size")
            if self._memory.shape[2] != memory_slots.shape[2]:
                raise ValueError("All memory writes must share the same memory dimension")
            memory = torch.cat((self._memory, memory_slots), dim=1)
            mask = torch.cat((self._mask, slot_mask), dim=1)

        if memory.shape[1] > self.max_slots:
            memory = memory[:, -self.max_slots :, :]
            mask = mask[:, -self.max_slots :]

        self._memory = memory
        self._mask = mask

    def get(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._memory is None:
            return None
        return self._memory, self._mask

    @property
    def num_slots(self) -> int:
        return 0 if self._memory is None else self._memory.shape[1]
