import torch
import torch.nn as nn

class MemoryCrossAttention(nn.Module):
    def __init__(self, d_model, d_mem, n_heads=4):
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_mem, d_model)
        self.v_proj = nn.Linear(d_mem, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        self.out_proj = nn.Linear(d_model, d_model)

        self.gate = nn.Parameter(torch.tensor(-4.0))  # sigmoid ≈ 0.018

        self.scale = 0.1

    def forward(self, hidden_states, memory):
        Q = self.q_proj(hidden_states)
        K = self.k_proj(memory)
        V = self.v_proj(memory)

        attn_out, _ = self.attn(Q, K, V)
        attn_out = self.out_proj(attn_out)

        return self.scale * torch.sigmoid(self.gate) * attn_out
