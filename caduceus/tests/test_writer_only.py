import torch
import pytest


def _make_writer(device="cpu", **overrides):
    from caduceus.memory.writer import BidirectionalConsistentMemoryWriter

    kwargs = {
        "d_model": 32,
        "d_sum": 16,
        "d_mem": 8,
        "n_layer": 4,
        "writer_mode": "bcw",
        "use_write_score": True,
        "num_global_slots": 1,
        "num_local_slots": 4,
        "pooling": "weighted",
    }
    kwargs.update(overrides)
    return BidirectionalConsistentMemoryWriter(**kwargs).to(device).eval()


def test_writer_shape_and_stats():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    writer = _make_writer(device)
    h_fwd = torch.randn(2, 16, 32, device=device)
    h_bwd = torch.randn(2, 16, 32, device=device)
    attention_mask = torch.ones(2, 16, device=device, dtype=torch.bool)

    output = writer(h_fwd, h_bwd, attention_mask, layer_idx=1)

    assert output.memory_slots.shape == (2, 5, 8)
    assert output.slot_mask.shape == (2, 5)
    assert output.slot_mask.all()
    assert {
        "direction_gate_mean",
        "direction_gate_std",
        "write_score_mean",
        "write_score_std",
        "fwd_bwd_cosine",
        "global_slot_norm",
        "local_slot_norm",
    }.issubset(output.stats)


def test_writer_can_skip_diagnostics_without_changing_slots():
    torch.manual_seed(11)
    writer = _make_writer()
    h_fwd = torch.randn(2, 16, 32)
    h_bwd = torch.randn(2, 16, 32)
    attention_mask = torch.ones(2, 16, dtype=torch.bool)

    with_stats = writer(
        h_fwd,
        h_bwd,
        attention_mask,
        layer_idx=1,
        collect_stats=True,
    )
    without_stats = writer(
        h_fwd,
        h_bwd,
        attention_mask,
        layer_idx=1,
        collect_stats=False,
    )

    assert torch.equal(with_stats.memory_slots, without_stats.memory_slots)
    assert torch.equal(with_stats.slot_mask, without_stats.slot_mask)
    assert with_stats.stats
    assert without_stats.stats == {}


def test_writer_padding_invariance_and_empty_safety():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    writer = _make_writer(device)
    valid_fwd = torch.randn(1, 9, 32, device=device)
    valid_bwd = torch.randn(1, 9, 32, device=device)

    def padded(total_length):
        pad_length = total_length - valid_fwd.shape[1]
        fwd = torch.cat(
            (valid_fwd, torch.randn(1, pad_length, 32, device=device)),
            dim=1,
        )
        bwd = torch.cat(
            (valid_bwd, torch.randn(1, pad_length, 32, device=device)),
            dim=1,
        )
        mask = torch.zeros(1, total_length, device=device, dtype=torch.bool)
        mask[:, : valid_fwd.shape[1]] = True
        return writer(fwd, bwd, mask, layer_idx=0)

    short = padded(12)
    long = padded(20)
    assert torch.equal(short.slot_mask, long.slot_mask)
    assert torch.allclose(short.memory_slots, long.memory_slots, atol=1e-5, rtol=1e-5)

    empty_states = torch.randn(2, 8, 32, device=device)
    empty_mask = torch.zeros(2, 8, device=device, dtype=torch.bool)
    empty = writer(empty_states, empty_states, empty_mask, layer_idx=0)
    assert not empty.slot_mask.any()
    assert torch.isfinite(empty.memory_slots).all()
    assert torch.equal(empty.memory_slots, torch.zeros_like(empty.memory_slots))


def test_writer_uses_aligned_directional_positions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2)
    writer = _make_writer(device)
    h_fwd = torch.randn(1, 12, 32, device=device)
    h_bwd = h_fwd + 0.1 * torch.randn_like(h_fwd)
    mask = torch.ones(1, 12, device=device, dtype=torch.bool)

    aligned = writer(h_fwd, h_bwd, mask, layer_idx=0)
    misaligned = writer(h_fwd, h_bwd.flip(1), mask, layer_idx=0)

    assert not torch.allclose(aligned.memory_slots, misaligned.memory_slots)
    assert aligned.stats["fwd_bwd_cosine"] > misaligned.stats["fwd_bwd_cosine"]


@pytest.mark.parametrize("writer_mode", ["single", "average", "scalar_gate", "bcw"])
def test_writer_ablation_modes_are_executable(writer_mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(5)
    writer = _make_writer(device, writer_mode=writer_mode)
    h_fwd = torch.randn(2, 10, 32, device=device, requires_grad=True)
    h_bwd = torch.randn(2, 10, 32, device=device, requires_grad=True)
    mask = torch.ones(2, 10, device=device, dtype=torch.bool)

    output = writer(h_fwd, h_bwd, mask, layer_idx=0)
    output.memory_slots.square().mean().backward()

    gradients = [
        parameter.grad
        for parameter in writer.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
