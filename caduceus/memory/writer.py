import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalMemoryWriter(nn.Module):
    """
    从 (h_fwd, h_bwd) 生成对齐的 memory entry：
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

        # summarize：把 token-level hidden -> summary 向量
        self.proj_fwd = nn.Linear(d_model, d_sum)
        self.proj_bwd = nn.Linear(d_model, d_sum)

        # gate：融合 fwd/bwd
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

        # compress：把 summary 压缩成 memory entry
        self.compress = nn.Sequential(
            nn.Linear(d_sum, d_sum),
            nn.GELU(),
            nn.Linear(d_sum, d_mem),
        )

        # 可选：归一化让相似度检索更稳
        self.norm = nn.LayerNorm(d_mem)

    def _pool(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        h: [B, L, D]
        attn_mask: [B, L] (1=valid, 0=pad) 可选
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

        # 调试/可视化时会很有用
        aux = {
            "s_fwd": s_fwd,
            "s_bwd": s_bwd,
            "gate": g,
            "s": s,
        }
        return entry, aux
