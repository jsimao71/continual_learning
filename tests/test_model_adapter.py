import torch

from cl.common.hooks import final_position_trace, validate_residual_identities
from cl.common.model_adapter import Intervention, TinyTransformerLM


def make_model():
    torch.manual_seed(3)
    model = TinyTransformerLM(vocab_size=20, max_length=8, width=16, layers=2, heads=2)
    model.eval()
    return model


def test_capture_and_disabled_instrumentation_have_logit_parity():
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4]])
    plain, trace_none = model(tokens)
    captured, trace = model(tokens, capture=True)
    assert trace_none is None
    assert trace is not None
    assert torch.equal(plain, captured)
    validate_residual_identities(trace)
    compact = final_position_trace(trace)
    assert compact[0]["pre_sa"].shape == (1, 16)
    assert compact[0]["attention"].shape == (1, 2, 4)


def test_zero_ablation_changes_only_selected_component_path():
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4]])
    intact, _ = model(tokens)
    ablated, trace = model(tokens, capture=True, intervention=Intervention(layer=0, component="sa"))
    assert not torch.equal(intact, ablated)
    assert trace is not None
    assert torch.count_nonzero(trace.layers[0].post_sa - trace.layers[0].pre_sa) == 0
    assert torch.count_nonzero(trace.layers[0].delta_sa) > 0


def test_head_ablation_is_selective_and_finite():
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4]])
    intact, _ = model(tokens)
    ablated, _ = model(
        tokens,
        intervention=Intervention(layer=1, component="sa", mode="head_zero", head=0),
    )
    assert torch.isfinite(ablated).all()
    assert not torch.equal(intact, ablated)


def test_local_attention_window_mask_and_reachability():
    model = TinyTransformerLM(vocab_size=20, max_length=8, width=16, layers=2, heads=2, attention_window=2)
    mask = model.causal_mask(6, "cpu")
    assert mask[5, 3] == 0 and torch.isneginf(mask[5, 2])
    assert torch.isneginf(mask[2, 3])


def test_head_contributions_and_replacement_preserve_tensor_semantics():
    model=make_model();tokens=torch.tensor([[1,2,3,4]]);_,trace=model(tokens,capture=True)
    heads=trace.layers[0].head_outputs
    assert heads.shape == (1,2,4,16) and torch.isfinite(heads).all()
    zero=torch.zeros_like(heads[:,0]);replaced,_=model(tokens,intervention=Intervention(0,"sa","head_replace",replacement=zero,head=0))
    ablated,_=model(tokens,intervention=Intervention(0,"sa","head_zero",head=0))
    assert torch.allclose(replaced,ablated,atol=1e-5,rtol=1e-5)
