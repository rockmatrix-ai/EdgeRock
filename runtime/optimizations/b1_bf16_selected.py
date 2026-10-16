import torch


def install(model):
    """Install the selected BF16 fusion stack before prefill on a BF16/FP32 runtime."""
    assert model.config.gdn_recurrent_state_storage == 'fp32'
    assert all(p.is_cuda and p.dtype == torch.bfloat16 for p in model.parameters())
    from . import b1_bf16_fusions
    b1_bf16_fusions.install(model)
    from . import b1_runtime_joint_projection
    b1_runtime_joint_projection.install(model)
    from . import b1_bf16_crosslayer
    b1_bf16_crosslayer.install(model)
    from . import b1_bf16_swa_prepare
    b1_bf16_swa_prepare.install(model)
    from . import b1_bf16_output_fusion
    b1_bf16_output_fusion.install(model)
    return b1_bf16_output_fusion.generation_function()
