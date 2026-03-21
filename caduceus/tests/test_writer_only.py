import torch

def test_writer_only():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from caduceus.memory.writer import BidirectionalMemoryWriter

    B, L, D = 2, 16, 64
    writer = BidirectionalMemoryWriter(
        d_model=D,
        d_sum=32,
        d_mem=128,
    ).to(device).eval()

    h = torch.randn(B, L, D, device=device)
    attn_mask = torch.ones(B, L, device=device)  # 1=valid, 0=pad

    with torch.no_grad():
        entry, aux = writer(h, h, attn_mask=attn_mask)

    print("entry shape:", entry.shape)
    print("entry device:", entry.device)
    print("aux keys:", aux.keys())

    # pytest 风格断言（可选，但强烈推荐）
    assert entry.shape == (B, 128)
