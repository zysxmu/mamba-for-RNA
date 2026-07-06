import torch


def test_lightweight_bcw_pools_before_projection_and_backpropagates():
    from caduceus.memory.lightweight import (
        LightweightBidirectionalConsistentMemoryWriter,
    )

    writer = LightweightBidirectionalConsistentMemoryWriter(
        d_model=32,
        d_sum=8,
        d_mem=8,
        n_layer=4,
        num_global_slots=1,
        num_local_slots=2,
    )
    projected_lengths = []
    handle = writer.directional_fusion.shared_projection.register_forward_pre_hook(
        lambda _module, args: projected_lengths.append(args[0].shape[1])
    )
    h_fwd = torch.randn(2, 128, 32, requires_grad=True)
    h_bwd = torch.randn(2, 128, 32, requires_grad=True)
    mask = torch.ones(2, 128, dtype=torch.bool)
    output = writer(h_fwd, h_bwd, mask, layer_idx=0)
    handle.remove()

    assert output.memory_slots.shape == (2, 3, 8)
    assert output.slot_mask.shape == (2, 3)
    assert projected_lengths == [3]

    output.memory_slots.square().mean().backward()
    assert h_fwd.grad is not None and h_fwd.grad.abs().sum() > 0
    assert h_bwd.grad is not None and h_bwd.grad.abs().sum() > 0
    assert writer.directional_fusion.shared_projection.weight.grad.abs().sum() > 0


def test_pooled_reader_projects_once_per_sample_and_masks_padding():
    from caduceus.memory.lightweight import PooledMemoryReader

    reader = PooledMemoryReader(d_model=32, d_mem=8)
    hidden = torch.randn(2, 128, 32)
    memory = torch.randn(2, 6, 8, requires_grad=True)
    memory_mask = torch.tensor(
        [[True, True, True, False, False, False], [True] * 6]
    )
    query_mask = torch.ones(2, 128, dtype=torch.bool)
    query_mask[0, 100:] = False

    output = reader(hidden, memory, memory_mask, query_mask=query_mask)
    assert output.memory_output.shape == hidden.shape
    assert torch.equal(
        output.memory_output[0, 100:],
        torch.zeros_like(output.memory_output[0, 100:]),
    )
    assert torch.allclose(
        output.memory_output[1, 0],
        output.memory_output[1, -1],
    )

    output.memory_output.square().mean().backward()
    assert memory.grad is not None and memory.grad.abs().sum() > 0
