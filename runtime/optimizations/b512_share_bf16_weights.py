import gc
import torch


@torch.no_grad()
def install(model):
    before = torch.cuda.memory_allocated()
    records = []
    for name, module in model.named_modules():
        groups = []
        if hasattr(module, '_qwen35_gate_up_weight'):
            groups.append(('_qwen35_gate_up_weight', (module.gate_proj, module.up_proj)))
        if hasattr(module, '_qwen35_all_native_weight'):
            groups.append(('_qwen35_all_native_weight',
                           (module.in_proj_qkv, module.in_proj_z, module.in_proj_b, module.in_proj_a)))
        if hasattr(module, '_qwen35_swa_qkv_weight'):
            groups.append(('_qwen35_swa_qkv_weight', (module.q_proj, module.k_proj, module.v_proj)))
        if hasattr(module, '_inference_writer_gate_up'):
            groups.append(('_inference_writer_gate_up', (module.write_gate_proj, module.write_up_proj)))
        if hasattr(module, '_inference_kv'):
            groups.append(('_inference_kv', (module.k_proj, module.v_proj)))
        for buffer_name, projections in groups:
            packed = getattr(module, buffer_name)
            offset = 0
            for projection in projections:
                weight = projection.weight
                view = packed[offset:offset + weight.shape[0]]
                assert weight.is_cuda and view.is_cuda
                assert weight.dtype == view.dtype == torch.bfloat16
                assert weight.shape == view.shape and weight.stride() == view.stride()
                assert torch.equal(weight.view(torch.uint8), view.view(torch.uint8))
                old_pointer = weight.data_ptr()
                weight.data = view
                assert weight.data_ptr() == view.data_ptr()
                records.append(dict(module=name, packed_buffer=buffer_name, offset=offset,
                                    shape=list(weight.shape), bitwise_equal=True,
                                    shared=old_pointer != weight.data_ptr()))
                offset += weight.shape[0]
            assert offset == packed.shape[0]
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    assert after < before
    return dict(before_bytes=before, after_bytes=after, saved_bytes=before-after,
                bitwise_weight_checks=len(records), records=records)
