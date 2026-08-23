from cl.experiments.paper05_pretrained import probes
from cl.experiments.paper05_next_iter import DOMAINS, SYNTAXES, donor_families, expanded_family_probes
from cl.analysis.paper05_inference import controlled_contrasts
from cl.experiments.paper05_subspace import fit_raw_subspace, project
import torch


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


def test_next_iteration_family_matrix_has_disjoint_identity_splits():
    rows = expanded_family_probes()
    assert len(rows) == len(DOMAINS) * len(SYNTAXES) * 4
    assert len({row["family"] for row in rows}) == 32
    for family in {row["family"] for row in rows}:
        identities = {row["identity"] for row in rows if row["family"] == family}
        assert identities == {0, 1, 2, 3}
    assert all(row["target_text"].startswith(" ") for row in rows)


def test_factorial_donors_change_only_requested_axes():
    rows = expanded_family_probes()
    probe = rows[0]
    donors = donor_families(probe, {row["family"] for row in rows})
    assert donors["replace_equivalent"] == probe["family"]
    assert donors["replace_syntax_mismatch"].endswith(":" + probe["semantic"])
    assert donors["replace_semantic_mismatch"].startswith(probe["syntax"] + ":")
    assert not donors["replace_nonequivalent"].endswith(":" + probe["semantic"])
    assert not donors["replace_nonequivalent"].startswith(probe["syntax"] + ":")


def test_common_subspace_rank_and_projection_are_well_defined():
    vectors = [torch.tensor([2.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])]
    rank1, explained1 = fit_raw_subspace(vectors, 1)
    rank2, explained2 = fit_raw_subspace(vectors, 2)
    assert rank1.shape == (1, 3) and rank2.shape == (2, 3)
    assert 0 < explained1 < explained2 <= 1
    residual = vectors[0] - project(vectors[0], rank2)
    assert torch.linalg.vector_norm(residual) < 1e-5
