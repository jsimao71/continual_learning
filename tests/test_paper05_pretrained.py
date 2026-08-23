from cl.experiments.paper05_pretrained import probes
from cl.analysis.paper05_inference import controlled_contrasts


def test_pretrained_probe_factorial_is_balanced():
    rows = probes()
    assert len(rows) == 2 * 2 * 3 * 3 * 4
    assert len({row["family"] for row in rows}) == 4
    assert {(row["evidence"], row["noise"]) for row in rows} == {
        (evidence, noise) for evidence in (1, 2, 4) for noise in (0, 2, 4)
    }


def test_controlled_contrast_uses_run_relation_units():
    base = {"model_setting": "m", "seed": "1", "relation_id": "r", "control_type": "irrelevant_nuisance",
            "target_control": "observed", "prefix_evidence": "pattern", "noise_level": "4",
            "training_stage": "trained", "location": "post_block"}
    rows = [{**base, "layer": "0", "mean_js_to_pattern_centroid": "0.4"},
            {**base, "layer": "2", "mean_js_to_pattern_centroid": "0.1"}]
    result = controlled_contrasts(rows)
    assert result[0]["n_relation_run_units"] == 1
    assert abs(result[0]["estimate"] + 0.3) < 1e-12
