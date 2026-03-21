import torch

def test_memory_read_before_write():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from caduceus.modeling_caduceus import CaduceusMixerModel
    from caduceus.configuration_caduceus import CaduceusConfig

    config = CaduceusConfig(vocab_size=128)
    model = CaduceusMixerModel(config).to(device).eval()

    # 每层都读/写，最容易暴露“先写后读”的问题
    model.memory_read_stride = 1
    model.memory_write_stride = 1

    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)

    with torch.no_grad():
        _ = model(input_ids=input_ids)

    sizes = getattr(model.memory_pool, "_get_sizes", None)
    assert sizes is not None and len(sizes) == len(model.layers)

    print("first few get sizes:", sizes[:5])

    # 关键断言：第0层 read 时 pool 必须是空
    sizes = model.memory_pool._get_sizes
    print("first few get sizes:", sizes[:5])
    assert sizes[0] == 0

