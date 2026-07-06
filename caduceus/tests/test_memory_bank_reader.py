import torch


def test_cross_layer_bank_fifo_preserves_graph():
    from caduceus.memory.bank import CrossLayerMemoryBank

    bank = CrossLayerMemoryBank(max_slots=5)
    first = torch.randn(2, 3, 4, requires_grad=True)
    second = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    bank.append(first, mask)
    bank.append(second, mask)

    memory, memory_mask = bank.get()
    assert memory.shape == (2, 5, 4)
    assert memory_mask.shape == (2, 5)
    memory.sum().backward()
    assert first.grad is not None
    assert second.grad is not None
    assert first.grad[:, 0].abs().sum() == 0  # Oldest FIFO slot was removed.
    assert second.grad.abs().sum() > 0


def test_reader_masks_empty_slots_and_tokens_can_select_different_memory():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(3)

    from caduceus.memory.reader import MemoryCrossAttentionReader

    reader = MemoryCrossAttentionReader(
        d_model=32,
        d_mem=16,
        n_heads=4,
    ).to(device).eval()
    hidden = torch.randn(2, 5, 32, device=device)
    memory = torch.randn(2, 4, 16, device=device)
    mask = torch.tensor(
        [[True, True, True, False], [False, False, False, False]],
        device=device,
    )

    output = reader(hidden, memory, mask, return_attention=True)
    assert reader.q_proj.out_features == 16
    assert reader.out_proj.in_features == 16
    assert output.memory_output.shape == hidden.shape
    assert output.attention_weights.shape == (2, 4, 5, 4)
    assert torch.isfinite(output.memory_output).all()
    assert torch.equal(output.memory_output[1], torch.zeros_like(output.memory_output[1]))
    assert torch.equal(
        output.attention_weights[0, :, :, 3],
        torch.zeros_like(output.attention_weights[0, :, :, 3]),
    )
    assert not torch.allclose(
        output.attention_weights[0, :, 0],
        output.attention_weights[0, :, 1],
    )


def test_reader_can_skip_diagnostics_without_changing_output():
    from caduceus.memory.reader import MemoryCrossAttentionReader

    torch.manual_seed(13)
    reader = MemoryCrossAttentionReader(
        d_model=32,
        d_mem=16,
        n_heads=4,
    ).eval()
    hidden = torch.randn(2, 5, 32)
    memory = torch.randn(2, 3, 16)
    mask = torch.ones(2, 3, dtype=torch.bool)

    with_stats = reader(hidden, memory, mask, collect_stats=True)
    without_stats = reader(hidden, memory, mask, collect_stats=False)

    assert torch.equal(with_stats.memory_output, without_stats.memory_output)
    assert "memory_output_norm" in with_stats.stats
    assert without_stats.stats == {}


def test_reader_masks_padding_queries():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from caduceus.memory.reader import MemoryCrossAttentionReader

    reader = MemoryCrossAttentionReader(
        d_model=32,
        d_mem=16,
        n_heads=4,
    ).to(device).eval()
    hidden = torch.randn(2, 5, 32, device=device)
    memory = torch.randn(2, 3, 16, device=device)
    memory_mask = torch.ones(2, 3, dtype=torch.bool, device=device)
    query_mask = torch.tensor(
        [[True, True, True, False, False], [True, False, False, False, False]],
        device=device,
    )

    output = reader(
        hidden,
        memory,
        memory_mask,
        query_mask=query_mask,
        return_attention=True,
    )
    assert torch.equal(
        output.memory_output[~query_mask],
        torch.zeros_like(output.memory_output[~query_mask]),
    )
    expanded_padding = (~query_mask)[:, None, :, None].expand_as(
        output.attention_weights
    )
    assert torch.equal(
        output.attention_weights[expanded_padding],
        torch.zeros_like(output.attention_weights[expanded_padding]),
    )


def test_memory_disable_intervention_and_baseline_equivalence():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from caduceus.configuration_caduceus import CaduceusConfig
    from caduceus.modeling_caduceus import CaduceusMixerModel

    common = dict(
        d_model=64,
        n_layer=4,
        vocab_size=32,
        fused_add_norm=False,
        memory_d_sum=32,
        memory_d_mem=16,
        memory_n_heads=4,
        memory_num_global_slots=1,
        memory_num_local_slots=3,
        memory_write_stride=1,
        memory_read_stride=1,
    )
    torch.manual_seed(4)
    baseline = CaduceusMixerModel(CaduceusConfig(use_memory=False, **common)).to(device).eval()
    torch.manual_seed(4)
    full = CaduceusMixerModel(CaduceusConfig(use_memory=True, **common)).to(device).eval()
    input_ids = torch.randint(5, 32, (2, 16), device=device)

    with torch.no_grad():
        baseline_output, _ = baseline(input_ids=input_ids)
        disabled_output, _ = full(
            input_ids=input_ids,
            disable_memory_read=True,
        )
        enabled_output, _ = full(input_ids=input_ids)

    assert torch.equal(baseline_output, disabled_output)
    assert enabled_output.shape == baseline_output.shape
    assert not torch.equal(enabled_output, baseline_output)
