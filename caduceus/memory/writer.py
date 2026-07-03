import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalMemoryWriter(nn.Module):
    """
    Build an aligned memory entry from forward and backward hidden states:
      s_fwd = summarize(h_fwd)
      s_bwd = summarize(h_bwd)
      s = gate(s_fwd, s_bwd)
      entry = compress(s)
    """

    def __init__(
        self,
        d_model: int,
        d_sum: int = 256,      
        d_mem: int = 128,     
        pool: str = "mean",   
        gate: str = "scalar", 
    ):
        super().__init__()
        self.pool = pool
        self.gate = gate

        # Project token-level hidden states into summary vectors.
        self.proj_fwd = nn.Linear(d_model, d_sum)
        self.proj_bwd = nn.Linear(d_model, d_sum)

        # Learn how much each direction contributes to the summary.
        if gate == "scalar":
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d_sum, d_sum),
                nn.GELU(),
                nn.Linear(d_sum, 1),
            )
        elif gate == "vector":
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d_sum, d_sum),
                nn.GELU(),
                nn.Linear(d_sum, d_sum),
            )
        else:
            raise ValueError(f"Unknown gate: {gate}")

        # Compress the fused summary into a memory entry.
        self.compress = nn.Sequential(
            nn.Linear(d_sum, d_sum),
            nn.GELU(),
            nn.Linear(d_sum, d_mem),
        )

        # Normalize entries to stabilize similarity-based retrieval.
        self.norm = nn.LayerNorm(d_mem)

    def _pool(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        h: [B, L, D]
        attn_mask: optional [B, L] tensor (1=valid, 0=padding)
        """
        if self.pool == "mean":
            if attn_mask is None:
                return h.mean(dim=1)
            w = attn_mask.float().unsqueeze(-1)  # [B, L, 1]
            return (h * w).sum(dim=1) / (w.sum(dim=1).clamp_min(1.0))
        if self.pool == "last":
            return h[:, -1, :]
        if self.pool == "cls":
            return h[:, 0, :]
        raise ValueError(f"Unknown pool: {self.pool}")

    def forward(
        self,
        h_fwd: torch.Tensor,
        h_bwd: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        """
        h_fwd/h_bwd: [B, L, D]
        """
        p_fwd = self._pool(h_fwd, attn_mask)  # [B, D]
        p_bwd = self._pool(h_bwd, attn_mask)  # [B, D]

        s_fwd = self.proj_fwd(p_fwd)          # [B, d_sum]
        s_bwd = self.proj_bwd(p_bwd)          # [B, d_sum]

        g_in = torch.cat([s_fwd, s_bwd], dim=-1)  # [B, 2*d_sum]
        g = torch.sigmoid(self.gate_mlp(g_in))    # [B,1] or [B,d_sum]

        s = g * s_fwd + (1.0 - g) * s_bwd         # [B, d_sum]

        entry = self.compress(s)                  # [B, d_mem]
        entry = self.norm(entry)

        # Expose intermediate values for diagnostics and visualization.
        aux = {
            "s_fwd": s_fwd,
            "s_bwd": s_bwd,
            "gate": g,
            "s": s,
        }
        return entry, aux
