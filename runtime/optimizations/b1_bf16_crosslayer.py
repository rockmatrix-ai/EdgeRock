import inspect
import textwrap
from types import MethodType


def install(model):
    text_model = next(m for m in model.modules()
        if hasattr(m, 'layers') and hasattr(m, 'rotary_emb') and hasattr(m, 'norm'))
    layers = list(text_model.layers)
    assert len(layers) == 32
    original = type(layers[0]).forward
    source = textwrap.dedent(inspect.getsource(original))
    old = "        raise RuntimeError('precomputed input norm requires cross-layer fusion')"
    assert source.count(old) == 1
    source = source.replace(old, '        residual, hidden_states = hidden_states')
    old = '    hidden_states = self.mlp(hidden_states)\n    hidden_states = residual + hidden_states\n'
    assert source.count(old) == 1
    new = '''    hidden_states = self.mlp(hidden_states)
    if hidden_states.shape[:2] == (1, 1):
        return _cuda_module('inference_kernels').fused_residual_qwen35_rmsnorm(
            residual, hidden_states, self.__dict__['_b1_next_input_norm'])
    hidden_states = residual + hidden_states
'''
    source = source.replace(old, new)
    namespace = dict(original.__globals__)
    exec(compile(source, '<bf16-crosslayer-decoder>', 'exec'), namespace)
    for index, layer in enumerate(layers):
        # An existing norm is referenced without registering a second module path.
        layer.__dict__['_b1_next_input_norm'] = (
            layers[index + 1].input_layernorm if index + 1 < len(layers) else text_model.norm)
        layer.forward = MethodType(namespace['forward'], layer)

    original = type(text_model).forward
    source = textwrap.dedent(inspect.getsource(original))
    old = "        raise RuntimeError('unexpected precomputed final RMSNorm')"
    assert source.count(old) == 1
    source = source.replace(old, '        hidden_states = hidden_states[1]')
    source = source.replace('            all_hidden_states += (hidden_states,)',
        '            all_hidden_states += (hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states,)')
    namespace = dict(original.__globals__)
    exec(compile(source, '<bf16-crosslayer-text-model>', 'exec'), namespace)
    text_model.forward = MethodType(namespace['forward'], text_model)
