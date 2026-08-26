import torch

from cl.common.model_adapter import TinyTransformerLM
from cl.experiments.paper08_trace import Patch, forward_patched


def test_qkv_trace_shapes_and_exact_attention():
    model = TinyTransformerLM(12, 8, width=8, layers=2, heads=2)
    tokens = torch.tensor([[3, 4, 1, 2, 3]])
    logits, trace = model(tokens, capture=True)
    layer = trace.layers[0]
    assert layer.queries.shape == (1, 2, 5, 4)
    assert layer.keys.shape == layer.values.shape == layer.queries.shape
    reconstructed = torch.softmax(layer.qk_scores + model.causal_mask(5, "cpu")[None, None], -1)
    torch.testing.assert_close(layer.attention, reconstructed)
    assert layer.ff_activations.shape == (1, 5, 16)
    assert logits.shape == (1, 5, 12)


def test_empty_patched_forward_matches_model_and_q_patch_changes_output():
    torch.manual_seed(3)
    model = TinyTransformerLM(12, 8, width=8, layers=2, heads=2).eval()
    first = torch.tensor([[3, 4, 1, 2, 3]])
    second = torch.tensor([[3, 5, 1, 2, 3]])
    expected, _ = model(first)
    torch.testing.assert_close(forward_patched(model, first, []), expected)
    _, control = model(second, capture=True)
    patched = forward_patched(model, first, [Patch(0, "q", control.layers[0].queries, (4,))])
    assert not torch.equal(patched, expected)
