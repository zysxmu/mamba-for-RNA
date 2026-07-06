"""Token-to-memory cross-attention for cross-layer RNA memory."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .types import MemoryReaderOutput


class MemoryCrossAttentionReader(nn.Module):
    """Retrieve cross-layer memory for every current-layer token."""

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("memory attention dropout must be in [0, 1)")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)

        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_mem)
        self.attention_dim = int(d_mem)
        if self.attention_dim % self.n_heads != 0:
            raise ValueError("d_mem must be divisible by memory_n_heads")
        self.head_dim = self.attention_dim // self.n_heads
        self.q_proj = nn.Linear(d_model, self.attention_dim)
        self.k_proj = nn.Linear(d_mem, self.attention_dim)
        self.v_proj = nn.Linear(d_mem, self.attention_dim)
        self.out_proj = nn.Linear(self.attention_dim, d_model)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

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
        if memory_bank.shape[1] == 0:
            raise ValueError("memory_bank must contain at least one slot")
        if query_mask is not None and query_mask.shape != hidden_states.shape[:2]:
            raise ValueError("query_mask must have shape [B, L]")

        memory_mask = memory_mask.to(device=memory_bank.device, dtype=torch.bool)
        if query_mask is None:
            query_mask = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            query_mask = query_mask.to(device=hidden_states.device, dtype=torch.bool)
        has_memory = memory_mask.any(dim=1)
        valid_queries = query_mask & has_memory.unsqueeze(1)

        # Multi-head shapes: Q [B,H,L,Dh], K/V [B,H,M,Dh].
        q = self._split_heads(self.q_proj(self.query_norm(hidden_states)))
        normalized_memory = self.memory_norm(memory_bank)
        k = self._split_heads(self.k_proj(normalized_memory))
        v = self._split_heads(self.v_proj(normalized_memory))

        # Compute attention logits in FP32 for stable masked softmax under AMP.
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1))
        logits = logits / math.sqrt(self.head_dim)
        key_mask = memory_mask[:, None, None, :]
        logits = logits.masked_fill(~key_mask, torch.finfo(logits.dtype).min)

        # Fully masked samples receive a zero output instead of NaN.
        safe_logits = torch.where(
            has_memory[:, None, None, None],
            logits,
            torch.zeros_like(logits),
        )
        attention = torch.softmax(safe_logits, dim=-1)
        attention = attention * key_mask.to(attention.dtype)
        attention = attention * has_memory[:, None, None, None].to(attention.dtype)
        attention = attention * query_mask[:, None, :, None].to(attention.dtype)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        context = torch.matmul(attention.to(v.dtype), v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.attention_dim,
        )
        memory_output = self.out_proj(context)
        memory_output = memory_output * valid_queries.unsqueeze(-1).to(memory_output.dtype)

        stats = {}
        if collect_stats:
            valid_outputs = memory_output[valid_queries]
            output_norm = (
                valid_outputs.float().norm(dim=-1).mean()
                if valid_outputs.numel()
                else memory_output.new_zeros((), dtype=torch.float)
            )
            stats["memory_output_norm"] = output_norm.detach().to(memory_output.dtype)

        returned_attention: Optional[torch.Tensor] = None
        if return_attention:
            returned_attention = attention.detach()
            probs = attention.float().clamp_min(1e-12)
            entropy = -(probs * probs.log()).sum(dim=-1)
            entropy_mask = valid_queries[:, None, :].expand_as(entropy)
            stats["attention_entropy"] = (
                entropy.masked_select(entropy_mask).mean().detach()
                if entropy_mask.any()
                else entropy.new_zeros(())
            ).to(memory_output.dtype)

        return MemoryReaderOutput(
            memory_output=memory_output,
            stats=stats,
            attention_weights=returned_attention,
        )
