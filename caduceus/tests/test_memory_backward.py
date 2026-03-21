import torch

def test_memory_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from caduceus.modeling_caduceus import CaduceusMixerModel
    from caduceus.configuration_caduceus import CaduceusConfig

    config = CaduceusConfig(vocab_size=128)
    model = CaduceusMixerModel(config).to(device).train()

    model.memory_read_stride = 1
    model.memory_write_stride = 2

    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)

    out, _ = model(input_ids=input_ids)

    loss = out.float().pow(2).mean()
    loss.backward()

    found_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            found_grad = True
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
            break

    assert found_grad, "No gradients found anywhere"
    print("loss:", float(loss.item()))
