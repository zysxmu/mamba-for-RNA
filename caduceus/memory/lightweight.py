"""Low-overhead bidirectional writing and cross-layer memory reading."""

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .types import MemoryReaderOutput, MemoryWriterOutput
from .writer import DirectionalFusion


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = mask.to(device=values.device, dtype=torch.bool)
    valid = mask.any(dim=1)
    weights = mask.unsqueeze(-1).to(values.dtype)
    summary = (values * weights).sum(dim=1)
    summary = summary / weights.sum(dim=1).clamp_min(1.0)
    summary = summary * valid.unsqueeze(-1).to(summary.dtype)
    return summary, valid


class AlignedDirectionalSlotPool(nn.Module):
    """Pool aligned directional states before applying learned BCW projections."""

    def __init__(self, num_global_slots: int, num_local_slots: int) -> None:
        super().__init__()
        self.num_global_slots = int(num_global_slots)
        self.num_local_slots = int(num_local_slots)

    def _region_ids(self, valid: torch.Tensor) -> torch.Tensor:
        rank = valid.long().cumsum(dim=1) - 1
        lengths = valid.long().sum(dim=1, keepdim=True).clamp_min(1)
        ids = torch.div(
            rank * self.num_local_slots,
            lengths,
            rounding_mode="floor",
        ).clamp(max=max(0, self.num_local_slots - 1))
        return ids.masked_fill(~valid, -1)

    def forward(
        self,
        h_fwd: torch.Tensor,
        h_bwd: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_fwd.shape != h_bwd.shape:
            raise ValueError("Forward and backward states must have identical shapes")
        if attention_mask.shape != h_fwd.shape[:2]:
            raise ValueError("attention_mask must have shape [B, L]")

        valid = attention_mask.to(device=h_fwd.device, dtype=torch.bool)
        fwd_slots = []
        bwd_slots = []
        slot_masks = []

        if self.num_global_slots:
            fwd, slot_valid = _masked_mean(h_fwd, valid)
            bwd, _ = _masked_mean(h_bwd, valid)
            fwd_slots.append(fwd)
            bwd_slots.append(bwd)
            slot_masks.append(slot_valid)

        if self.num_local_slots:
            region_ids = self._region_ids(valid)
            for slot_idx in range(self.num_local_slots):
                region_mask = valid & region_ids.eq(slot_idx)
                fwd, slot_valid = _masked_mean(h_fwd, region_mask)
                bwd, _ = _masked_mean(h_bwd, region_mask)
                fwd_slots.append(fwd)
                bwd_slots.append(bwd)
                slot_masks.append(slot_valid)

        return (
            torch.stack(fwd_slots, dim=1),
            torch.stack(bwd_slots, dim=1),
            torch.stack(slot_masks, dim=1),
        )


class LightweightSlotEncoder(nn.Module):
    """Compress a small set of already pooled BCW summaries into memory slots."""

    def __init__(
        self,
        d_sum: int,
        d_mem: int,
        n_layer: int,
        num_global_slots: int,
        num_local_slots: int,
        use_layer_embedding: bool,
        use_slot_type_embedding: bool,
        use_slot_position_embedding: bool,
    ) -> None:
        super().__init__()
        self.num_global_slots = int(num_global_slots)
        self.num_local_slots = int(num_local_slots)
        self.compressor = nn.Sequential(
            nn.Linear(d_sum, d_mem),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_mem)
        self.layer_embedding = (
            nn.Embedding(n_layer, d_mem) if use_layer_embedding else None
        )
        self.slot_type_embedding = (
            nn.Embedding(2, d_mem) if use_slot_type_embedding else None
        )
        self.slot_position_embedding = (
            nn.Embedding(num_local_slots, d_mem)
            if use_slot_position_embedding and num_local_slots > 0
            else None
        )

    def forward(
        self,
        fused_slots: torch.Tensor,
        write_score: torch.Tensor,
        slot_mask: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        memory = self.compressor(fused_slots)

        if self.layer_embedding is not None:
            layer = torch.tensor(layer_idx, device=memory.device, dtype=torch.long)
            memory = memory + self.layer_embedding(layer).view(1, 1, -1)

        if self.slot_type_embedding is not None:
            type_ids = torch.tensor(
                [0] * self.num_global_slots + [1] * self.num_local_slots,
                device=memory.device,
                dtype=torch.long,
            )
            memory = memory + self.slot_type_embedding(type_ids).unsqueeze(0)

        if self.slot_position_embedding is not None:
            positions = torch.arange(
                self.num_local_slots,
                device=memory.device,
                dtype=torch.long,
            )
            local_encoding = self.slot_position_embedding(positions)
            global_encoding = memory.new_zeros(
                (self.num_global_slots, memory.shape[-1])
            )
            memory = memory + torch.cat(
                (global_encoding, local_encoding),
                dim=0,
            ).unsqueeze(0)

        memory = self.norm(memory)
        memory = memory * write_score
        return memory * slot_mask.unsqueeze(-1).to(memory.dtype)


class LightweightBidirectionalConsistentMemoryWriter(nn.Module):
    """BCW with pool-before-project complexity independent of sequence length."""

    def __init__(
        self,
        d_model: int,
        d_sum: int,
        d_mem: int,
        n_layer: int,
        writer_mode: str = "bcw",
        share_direction_projection: bool = True,
        single_direction: str = "fwd",
        use_write_score: bool = True,
        num_global_slots: int = 1,
        num_local_slots: int = 2,
        pooling: str = "weighted",
        use_layer_embedding: bool = True,
        use_slot_type_embedding: bool = True,
        use_slot_position_embedding: bool = True,
    ) -> None:
        super().__init__()
        del pooling
        self.slot_pool = AlignedDirectionalSlotPool(
            num_global_slots=num_global_slots,
            num_local_slots=num_local_slots,
        )
        self.directional_fusion = DirectionalFusion(
            d_model=d_model,
            d_sum=d_sum,
            mode=writer_mode,
            share_projection=share_direction_projection,
            single_direction=single_direction,
            use_write_score=use_write_score,
        )
        self.summarizer = LightweightSlotEncoder(
            d_sum=d_sum,
            d_mem=d_mem,
            n_layer=n_layer,
            num_global_slots=num_global_slots,
            num_local_slots=num_local_slots,
            use_layer_embedding=use_layer_embedding,
            use_slot_type_embedding=use_slot_type_embedding,
            use_slot_position_embedding=use_slot_position_embedding,
        )

    def forward(
        self,
        h_fwd: torch.Tensor,
        h_bwd: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        collect_stats: bool = True,
    ) -> MemoryWriterOutput:
        if attention_mask is None:
            attention_mask = torch.ones(
                h_fwd.shape[:2],
                device=h_fwd.device,
                dtype=torch.bool,
            )
        pooled_fwd, pooled_bwd, slot_mask = self.slot_pool(
            h_fwd,
            h_bwd,
            attention_mask,
        )
        fused, write_score, stats = self.directional_fusion(
            pooled_fwd,
            pooled_bwd,
            slot_mask,
            collect_stats=collect_stats,
        )
        memory_slots = self.summarizer(
            fused,
            write_score,
            slot_mask,
            layer_idx,
        )
        if collect_stats:
            stats["num_memory_slots_written"] = memory_slots.new_tensor(
                memory_slots.shape[1]
            ).detach()
            stats["num_valid_slots_written"] = (
                slot_mask.float().sum(dim=1).mean().detach()
            )
        return MemoryWriterOutput(
            memory_slots=memory_slots,
            slot_mask=slot_mask,
            stats=stats,
        )


class PooledMemoryReader(nn.Module):
    """Read a masked memory-bank summary without token-to-slot attention."""

    def __init__(self, d_model: int, d_mem: int) -> None:
        super().__init__()
        self.memory_norm = nn.LayerNorm(d_mem)
        self.out_proj = nn.Linear(d_mem, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_bank: torch.Tensor,
        memory_mask: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        collect_stats: bool = True,
    ) -> MemoryReaderOutput:
        if hidden_states.ndim != 3 or memory_bank.ndim != 3:
            raise ValueError("hidden_states and memory_bank must be rank-3 tensors")
        if memory_mask.shape != memory_bank.shape[:2]:
            raise ValueError("memory_mask must have shape [B, M]")
        if hidden_states.shape[0] != memory_bank.shape[0]:
            raise ValueError("hidden_states and memory_bank must share the batch dimension")
        if query_mask is not None and query_mask.shape != hidden_states.shape[:2]:
            raise ValueError("query_mask must have shape [B, L]")

        memory_mask = memory_mask.to(device=memory_bank.device, dtype=torch.bool)
        weights = memory_mask.to(memory_bank.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        summary = torch.bmm(
            weights.unsqueeze(1),
            self.memory_norm(memory_bank),
        ).squeeze(1)
        projected = self.out_proj(summary).unsqueeze(1)
        has_memory = memory_mask.any(dim=1)
        output = projected * has_memory[:, None, None].to(projected.dtype)

        if query_mask is not None:
            output = output.expand(-1, hidden_states.shape[1], -1)
            output = output * query_mask.unsqueeze(-1).to(output.dtype)

        stats: Dict[str, torch.Tensor] = {}
        if collect_stats:
            stats["memory_output_norm"] = (
                projected.detach().float().norm(dim=-1).mean().to(projected.dtype)
            )

        attention = weights[:, None, :].detach() if return_attention else None
        return MemoryReaderOutput(
            memory_output=output,
            stats=stats,
            attention_weights=attention,
        )
