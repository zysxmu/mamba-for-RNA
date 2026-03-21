import torch

def test_memory_stride_counts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from caduceus.modeling_caduceus import CaduceusMixerModel
    from caduceus.configuration_caduceus import CaduceusConfig

    config = CaduceusConfig(vocab_size=128)
    model = CaduceusMixerModel(config).to(device).eval()

    # 关键：我们在 test 里覆盖 stride
    model.memory_write_stride = 2
    model.memory_read_stride = 1

    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)

    with torch.no_grad():
        _ = model(input_ids=input_ids)

    n_layer = len(model.layers)
    pool = model.memory_pool

    print("n_layer:", n_layer)
    print("get_calls :", pool._get_calls)
    print("push_calls:", pool._push_calls)

    # 预期：每层都读 -> get_calls == n_layer
    assert pool._get_calls == n_layer

    # 预期：i=0,2,4,... 写 -> 次数 = ceil(n_layer / 2)
    expected_push = (n_layer + model.memory_write_stride - 1) // model.memory_write_stride
    assert pool._push_calls == expected_push
