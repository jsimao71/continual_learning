import random

import numpy as np

from cl.analysis.hierarchy_geometry import hierarchy_metrics
from cl.semantic.synthetic import build_semantic_corpus


def test_hierarchy_paths_distances_and_metadata():
    corpus = build_semantic_corpus(seed=3, repeats=1)
    hierarchy = corpus.hierarchy
    falcon = "natural:noun:falcon"
    penguin = "natural:noun:penguin"
    dog = "natural:noun:dog"
    assert hierarchy.path(falcon) == ("natural:noun:entity", "natural:noun:bird", falcon)
    assert hierarchy.distance(falcon, penguin) == 2
    assert hierarchy.distance(falcon, dog) == 4
    assert penguin in hierarchy.siblings(falcon)
    for item in hierarchy.items.values():
        assert item.depth == len(hierarchy.path(item.item_id)) - 1


def test_semantic_generation_is_deterministic_and_balanced():
    first = build_semantic_corpus(seed=7, repeats=2)
    second = build_semantic_corpus(seed=7, repeats=2)
    assert first.corpus_hash == second.corpus_hash
    assert first.train_sequences == second.train_sequences
    lengths = {len(sequence) for sequence in first.train_sequences}
    assert lengths == {12}
    counts = {}
    for probe in first.probes:
        key = (probe.family, probe.natural, probe.abstraction_level, probe.template_id)
        counts[key] = counts.get(key, 0) + 1
    assert len(set(counts.values())) == 1


def test_known_hierarchy_geometry_beats_permuted_control():
    corpus = build_semantic_corpus(seed=0, repeats=1)
    hierarchy = corpus.hierarchy
    leaves = hierarchy.leaves(family="noun", natural=True)
    rng = random.Random(0)
    parent_vectors = {}
    rows = []
    for item in leaves:
        parent_vectors.setdefault(item.parent_id, np.asarray([len(parent_vectors), 1.0, 0.0]))
        for template in range(3):
            vector = parent_vectors[item.parent_id] + np.asarray([0.0, 0.0, template * 0.001])
            rows.append({
                "model_setting": "toy", "seed": 0, "layer": 0, "location": "post_block",
                "family": "noun", "natural": True, "abstraction_level": "parent",
                "item_id": item.item_id, "vector": vector.tolist(),
            })
    result = hierarchy_metrics(hierarchy, rows, seed=4)[0]
    assert result["normalized_separation"] > 0.9
    assert result["tree_neighbor_recovery"] == 1.0


def test_paper05_control_join_key_is_stable():
    corpus = build_semantic_corpus(seed=1, repeats=1)
    rows = [(probe.item_id, probe.template_id, probe.abstraction_level) for probe in corpus.probes]
    assert rows == [(probe.item_id, probe.template_id, probe.abstraction_level) for probe in build_semantic_corpus(seed=1, repeats=1).probes]
