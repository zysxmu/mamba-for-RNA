"""Bidirectional writing and multi-slot summarization for RNA sequences."""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .types import MemoryWriterOutput


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a scalar mean over valid elements without retaining its graph."""
    valid = values.masked_select(mask)
    if valid.numel() == 0:
        return values.new_zeros(())
    return valid.float().mean().to(values.dtype)


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = values.masked_select(mask)
    if valid.numel() <= 1:
        return values.new_zeros(())
    return valid.float().std(unbiased=False).to(values.dtype)


class DirectionalFusion(nn.Module):
    """Align and fuse forward/backward states at every RNA position."""

    SUPPORTED_MODES = {"single", "average", "scalar_gate", "bcw"}

    def __init__(
        self,
        d_model: int,
        d_sum: int,
        mode: str = "bcw",
        share_projection: bool = True,
        single_direction: str = "fwd",
        use_write_score: bool = True,
    ) -> None:
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported memory writer mode: {mode}")
        if single_direction not in {"fwd", "bwd"}:
            raise ValueError("single_direction must be 'fwd' or 'bwd'")

        self.mode = mode
        self.share_projection = bool(share_projection)
        self.single_direction = single_direction
        self.use_write_score = bool(use_write_score)
        self.d_sum = int(d_sum)

        if mode == "single":
            self.single_projection = nn.Linear(d_model, d_sum)
            self.single_norm = nn.LayerNorm(d_sum)
            relation_dim = d_sum
        elif self.share_projection:
            self.shared_projection = nn.Linear(d_model, d_sum)
            self.shared_norm = nn.LayerNorm(d_sum)
            relation_dim = 4 * d_sum
        else:
            self.proj_fwd = nn.Linear(d_model, d_sum)
            self.proj_bwd = nn.Linear(d_model, d_sum)
            self.norm_fwd = nn.LayerNorm(d_sum)
            self.norm_bwd = nn.LayerNorm(d_sum)
            relation_dim = 4 * d_sum

        if mode in {"scalar_gate", "bcw"}:
            gate_dim = 1 if mode == "scalar_gate" else d_sum
            self.direction_gate_mlp = nn.Sequential(
                nn.Linear(relation_dim, d_sum),
                nn.GELU(),
                nn.Linear(d_sum, gate_dim),
            )

        if self.use_write_score:
            self.write_score_mlp = nn.Sequential(
                nn.Linear(relation_dim, d_sum),
                nn.GELU(),
                nn.Linear(d_sum, 1),
            )

    def _project_directions(
        self,
        h_fwd: torch.Tensor,
        h_bwd: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mode == "single":
            source = h_fwd if self.single_direction == "fwd" else h_bwd
            return self.single_norm(self.single_projection(source)), None

        if self.share_projection:
            projected = self.shared_norm(
                self.shared_projection(torch.cat((h_fwd, h_bwd), dim=0))
            )
            z_fwd, z_bwd = projected.chunk(2, dim=0)
        else:
            z_fwd = self.norm_fwd(self.proj_fwd(h_fwd))
            z_bwd = self.norm_bwd(self.proj_bwd(h_bwd))
        return z_fwd, z_bwd

    @staticmethod
    def _relation(z_fwd: torch.Tensor, z_bwd: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (z_fwd, z_bwd, torch.abs(z_fwd - z_bwd), z_fwd * z_bwd),
            dim=-1,
        )

    def forward(
        self,
        h_fwd: torch.Tensor,
        h_bwd: torch.Tensor,
        attention_mask: torch.Tensor,
        collect_stats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if h_fwd.shape != h_bwd.shape:
            raise ValueError("Forward and backward states must have identical shapes")
        if attention_mask.shape != h_fwd.shape[:2]:
            raise ValueError("attention_mask must have shape [B, L]")

        valid = attention_mask.to(device=h_fwd.device, dtype=torch.bool)
        z_fwd, z_bwd = self._project_directions(h_fwd, h_bwd)

        if z_bwd is None:
            relation = z_fwd
            z_fused = z_fwd
            direction_gate = z_fwd.new_ones((*z_fwd.shape[:2], 1))
        else:
            relation = self._relation(z_fwd, z_bwd)
            if self.mode == "average":
                direction_gate = z_fwd.new_full((*z_fwd.shape[:2], 1), 0.5)
            else:
                direction_gate = torch.sigmoid(self.direction_gate_mlp(relation))
            z_fused = direction_gate * z_fwd + (1.0 - direction_gate) * z_bwd

        if self.use_write_score:
            write_score = torch.sigmoid(self.write_score_mlp(relation))
        else:
            write_score = z_fused.new_ones((*z_fused.shape[:2], 1))

        valid_f = valid.unsqueeze(-1).to(z_fused.dtype)
        z_fused = z_fused * valid_f
        write_score = write_score * valid_f

        stats = {}
        if collect_stats:
            cosine = (
                z_fwd.new_ones(z_fwd.shape[:2])
                if z_bwd is None
                else F.cosine_similarity(
                    z_fwd.float(),
                    z_bwd.float(),
                    dim=-1,
                ).to(z_fwd.dtype)
            )
            gate_mask = valid.unsqueeze(-1).expand_as(direction_gate)
            stats = {
                "direction_gate_mean": _masked_mean(direction_gate, gate_mask).detach(),
                "direction_gate_std": _masked_std(direction_gate, gate_mask).detach(),
                "write_score_mean": _masked_mean(write_score, valid.unsqueeze(-1)).detach(),
                "write_score_std": _masked_std(write_score, valid.unsqueeze(-1)).detach(),
                "fwd_bwd_cosine": _masked_mean(cosine, valid).detach(),
            }
        return z_fused, write_score, stats


class MultiSlotMemorySummarizer(nn.Module):
    """Create global and valid-length-adaptive regional memory slots."""

    SUPPORTED_POOLING = {"mean", "max", "weighted"}

    def __init__(
        self,
        d_sum: int,
        d_mem: int,
        num_global_slots: int = 1,
        num_local_slots: int = 8,
        pooling: str = "weighted",
        use_write_score: bool = True,
        n_layer: int = 1,
        use_layer_embedding: bool = True,
        use_slot_type_embedding: bool = True,
        use_slot_position_embedding: bool = True,
    ) -> None:
        super().__init__()
        if num_global_slots not in {0, 1}:
            raise ValueError("num_global_slots currently supports only 0 or 1")
        if num_local_slots < 0:
            raise ValueError("num_local_slots must be non-negative")
        if num_global_slots + num_local_slots == 0:
            raise ValueError("At least one global or local memory slot is required")
        if pooling not in self.SUPPORTED_POOLING:
            raise ValueError(f"Unsupported memory pooling mode: {pooling}")
        if pooling == "weighted" and not use_write_score:
            raise ValueError("weighted pooling requires memory_use_write_score=true")
        if pooling != "weighted" and use_write_score:
            raise ValueError("mean/max pooling require use_write_score=false")

        self.num_global_slots = int(num_global_slots)
        self.num_local_slots = int(num_local_slots)
        self.pooling = pooling
        self.use_write_score = bool(use_write_score)
        self.d_mem = int(d_mem)

        self.memory_compressor = nn.Sequential(
            nn.Linear(d_sum, d_sum),
            nn.GELU(),
            nn.Linear(d_sum, d_mem),
        )
        self.memory_norm = nn.LayerNorm(d_mem)

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

    def _pool(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        write_score: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slot_valid = valid.any(dim=1)
        if self.pooling == "max":
            fill_value = torch.finfo(values.dtype).min
            masked = values.masked_fill(~valid.unsqueeze(-1), fill_value)
            summary = masked.max(dim=1).values
            summary = torch.where(slot_valid.unsqueeze(-1), summary, torch.zeros_like(summary))
            return summary, slot_valid

        if self.pooling == "weighted":
            weights = write_score.squeeze(-1) * valid.to(write_score.dtype)
        else:
            weights = valid.to(values.dtype)

        # Accumulate in FP32 for stable AMP/BF16 pooling.
        numerator = (
            values.float() * weights.float().unsqueeze(-1)
        ).sum(dim=1)
        denominator = weights.float().sum(dim=1, keepdim=True).clamp_min(1e-6)
        summary = (numerator / denominator).to(values.dtype)
        summary = torch.where(slot_valid.unsqueeze(-1), summary, torch.zeros_like(summary))
        return summary, slot_valid

    def _regional_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Assign valid tokens to equal regions based on each sample's length."""
        valid = attention_mask.to(dtype=torch.bool)
        rank = valid.long().cumsum(dim=1) - 1
        lengths = valid.long().sum(dim=1, keepdim=True).clamp_min(1)
        region_ids = torch.div(
            rank * self.num_local_slots,
            lengths,
            rounding_mode="floor",
        ).clamp(max=max(0, self.num_local_slots - 1))
        return region_ids.masked_fill(~valid, -1)

    def _pool_regions(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        write_score: torch.Tensor,
        region_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool all non-overlapping regions in one scatter pass."""
        batch, _, width = values.shape
        safe_ids = region_ids.clamp_min(0)
        region_valid = valid & region_ids.ge(0)

        if self.pooling == "weighted":
            weights = write_score.squeeze(-1) * region_valid.to(write_score.dtype)
        else:
            weights = region_valid.to(values.dtype)

        weights_fp32 = weights.float()
        denominator = values.new_zeros(
            (batch, self.num_local_slots),
            dtype=torch.float,
        )
        denominator.scatter_add_(1, safe_ids, weights_fp32)

        numerator = values.new_zeros(
            (batch, self.num_local_slots, width),
            dtype=torch.float,
        )
        expanded_ids = safe_ids.unsqueeze(-1).expand(-1, -1, width)
        numerator.scatter_add_(
            1,
            expanded_ids,
            values.float() * weights_fp32.unsqueeze(-1),
        )

        slot_valid = denominator.gt(0)
        summaries = numerator / denominator.clamp_min(1e-6).unsqueeze(-1)
        summaries = summaries.to(values.dtype)
        summaries = summaries * slot_valid.unsqueeze(-1).to(summaries.dtype)
        return summaries, slot_valid

    def forward(
        self,
        z_fused: torch.Tensor,
        write_score: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
        collect_stats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        valid = attention_mask.to(device=z_fused.device, dtype=torch.bool)
        summaries = []
        masks = []
        slot_types = []
        local_positions = []

        if self.num_global_slots:
            summary, slot_valid = self._pool(z_fused, valid, write_score)
            summaries.append(summary)
            masks.append(slot_valid)
            slot_types.append(0)
            local_positions.append(-1)

        if self.num_local_slots:
            region_ids = self._regional_ids(valid)
            if self.pooling in {"mean", "weighted"}:
                regional_summaries, regional_masks = self._pool_regions(
                    z_fused,
                    valid,
                    write_score,
                    region_ids,
                )
                summaries.extend(regional_summaries.unbind(dim=1))
                masks.extend(regional_masks.unbind(dim=1))
                slot_types.extend([1] * self.num_local_slots)
                local_positions.extend(range(self.num_local_slots))
            else:
                for slot_idx in range(self.num_local_slots):
                    region_valid = valid & region_ids.eq(slot_idx)
                    summary, slot_valid = self._pool(
                        z_fused,
                        region_valid,
                        write_score,
                    )
                    summaries.append(summary)
                    masks.append(slot_valid)
                    slot_types.append(1)
                    local_positions.append(slot_idx)

        slot_summaries = torch.stack(summaries, dim=1)  # [B, S, d_sum]
        slot_mask = torch.stack(masks, dim=1)  # [B, S]
        memory_slots = self.memory_compressor(slot_summaries)  # [B, S, d_mem]

        if self.layer_embedding is not None:
            layer = torch.tensor(layer_idx, device=z_fused.device, dtype=torch.long)
            memory_slots = memory_slots + self.layer_embedding(layer).view(1, 1, -1)

        if self.slot_type_embedding is not None:
            type_ids = torch.tensor(slot_types, device=z_fused.device, dtype=torch.long)
            memory_slots = memory_slots + self.slot_type_embedding(type_ids).unsqueeze(0)

        if self.slot_position_embedding is not None:
            local_ids = torch.tensor(local_positions, device=z_fused.device, dtype=torch.long)
            local_mask = local_ids.ge(0)
            position_ids = local_ids.clamp_min(0)
            position_encoding = self.slot_position_embedding(position_ids)
            position_encoding = position_encoding * local_mask.unsqueeze(-1)
            memory_slots = memory_slots + position_encoding.unsqueeze(0)

        memory_slots = self.memory_norm(memory_slots)
        memory_slots = memory_slots * slot_mask.unsqueeze(-1).to(memory_slots.dtype)

        stats = {}
        if collect_stats:
            global_mask = (
                slot_mask[:, :1] if self.num_global_slots else slot_mask[:, :0]
            )
            local_mask = slot_mask[:, self.num_global_slots :]
            stats = {
                "num_memory_slots_written": memory_slots.new_tensor(
                    memory_slots.shape[1]
                ).detach(),
                "num_valid_slots_written": slot_mask.float().sum(dim=1).mean().detach(),
                "global_slot_norm": self._slot_norm(
                    memory_slots[:, : self.num_global_slots], global_mask
                ).detach(),
                "local_slot_norm": self._slot_norm(
                    memory_slots[:, self.num_global_slots :], local_mask
                ).detach(),
            }
        return memory_slots, slot_mask, stats

    @staticmethod
    def _slot_norm(slots: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if slots.numel() == 0 or not mask.any():
            return slots.new_zeros(())
        norms = slots.float().norm(dim=-1)
        return norms.masked_select(mask).mean().to(slots.dtype)


class BidirectionalConsistentMemoryWriter(nn.Module):
    """Write aligned bidirectional RNA states into global/regional slots."""

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
        num_local_slots: int = 8,
        pooling: str = "weighted",
        use_layer_embedding: bool = True,
        use_slot_type_embedding: bool = True,
        use_slot_position_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.directional_fusion = DirectionalFusion(
            d_model=d_model,
            d_sum=d_sum,
            mode=writer_mode,
            share_projection=share_direction_projection,
            single_direction=single_direction,
            use_write_score=use_write_score,
        )
        self.summarizer = MultiSlotMemorySummarizer(
            d_sum=d_sum,
            d_mem=d_mem,
            num_global_slots=num_global_slots,
            num_local_slots=num_local_slots,
            pooling=pooling,
            use_write_score=use_write_score,
            n_layer=n_layer,
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
        z_fused, write_score, fusion_stats = self.directional_fusion(
            h_fwd,
            h_bwd,
            attention_mask,
            collect_stats=collect_stats,
        )
        memory_slots, slot_mask, slot_stats = self.summarizer(
            z_fused,
            write_score,
            attention_mask,
            layer_idx,
            collect_stats=collect_stats,
        )
        return MemoryWriterOutput(
            memory_slots=memory_slots,
            slot_mask=slot_mask,
            stats={**fusion_stats, **slot_stats},
        )


# Backward-compatible import name. Its behavior now follows the formal BCW
# writer interface and returns MemoryWriterOutput.
BidirectionalMemoryWriter = BidirectionalConsistentMemoryWriter
