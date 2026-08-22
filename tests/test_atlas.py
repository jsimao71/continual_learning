import math

from cl.ngram.atlas import build_atlas, occurrence_positions, sample_strata


def test_ngram_counts_match_brute_force():
    sequences = [(1, 2, 3, 1, 2, 4), (1, 2, 3)]
    atlas = build_atlas(sequences, n_values=(2,))
    entry = next(item for item in atlas if item.prefix_token_ids == (1, 2))
    assert entry.reference_corpus_frequency == 3
    assert entry.continuation_counts == {3: 2, 4: 1}
    assert entry.top_continuation_probability == 2 / 3
    expected = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    assert abs(entry.continuation_entropy - expected) < 1e-12


def test_occurrence_alignment_and_strata_are_deterministic():
    assert occurrence_positions((1, 2, 3, 1, 2, 4), (1, 2)) == [2, 5]
    atlas = build_atlas([(1, 2, 3), (1, 2, 4), (5, 6, 7)], n_values=(1, 2))
    first = sample_strata(atlas, seed=9)
    second = sample_strata(atlas, seed=9)
    assert first == second

