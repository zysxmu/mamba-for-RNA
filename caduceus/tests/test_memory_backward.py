import torch


def _assert_module_has_finite_nonzero_grad(module):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients, f"No gradients found for {type(module).__name__}"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_memory_specific_gradients():
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
        memory_write_stride=1,
        memory_read_stride=1,
    )
    model = CaduceusMixerModel(config).to(device).train()
    input_ids = torch.randint(5, config.vocab_size, (2, 16), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    output, _ = model(input_ids=input_ids, attention_mask=attention_mask)
    target = torch.randn_like(output)
    loss = (output.float() * target.float()).mean()
    loss.backward()

    _assert_module_has_finite_nonzero_grad(model.memory_writer.directional_fusion)
    _assert_module_has_finite_nonzero_grad(model.memory_writer.summarizer)
    _assert_module_has_finite_nonzero_grad(model.memory_attn)
    assert model.memory_read_gates.grad is not None
    assert torch.isfinite(model.memory_read_gates.grad).all()
    assert model.memory_read_gates.grad.abs().sum() > 0

    named = dict(model.memory_writer.named_parameters())
    assert named["directional_fusion.write_score_mlp.0.weight"].grad.abs().sum() > 0
    assert named["summarizer.layer_embedding.weight"].grad.abs().sum() > 0
    assert named["summarizer.slot_type_embedding.weight"].grad.abs().sum() > 0
    assert named["summarizer.slot_position_embedding.weight"].grad.abs().sum() > 0
