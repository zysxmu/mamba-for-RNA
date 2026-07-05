import torch


def test_memory_stride_counts_and_slot_growth():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from caduceus.configuration_caduceus import CaduceusConfig
    from caduceus.modeling_caduceus import CaduceusMixerModel

    config = CaduceusConfig(
        d_model=64,
        n_layer=4,
        vocab_size=32,
        fused_add_norm=False,
        use_memory=True,
        memory_d_sum=32,
        memory_d_mem=16,
        memory_n_heads=4,
        memory_num_global_slots=1,
        memory_num_local_slots=3,
        memory_write_stride=2,
        memory_read_stride=1,
    )
    model = CaduceusMixerModel(config).to(device).eval()
    input_ids = torch.randint(5, config.vocab_size, (2, 16), device=device)

    write_calls = []
    memory_sizes = []
    write_handle = model.memory_writer.register_forward_hook(
        lambda _module, _inputs, _output: write_calls.append(1)
    )

    def reader_hook(_module, _args, kwargs, _output):
        memory_sizes.append(kwargs["memory_bank"].shape[1])

    read_handle = model.memory_attn.register_forward_hook(
        reader_hook,
        with_kwargs=True,
    )
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    write_handle.remove()
    read_handle.remove()

    assert len(write_calls) == 2  # layers 0 and 2
    assert memory_sizes == [4, 4, 8]  # reads at layers 1, 2, and 3
