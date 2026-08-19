import numpy as np

from riskcal_tkg.corruptions import delete_training_edges


def test_deletion_is_reproducible_and_preserves_entities() -> None:
    values = np.asarray(
        [
            [0, 0, 1, 0],
            [1, 0, 2, 0],
            [2, 0, 0, 1],
            [0, 1, 2, 1],
            [1, 1, 0, 2],
            [2, 1, 1, 2],
        ],
        dtype=np.int64,
    )
    first = delete_training_edges(values, rate=0.5, seed=17)
    second = delete_training_edges(values, rate=0.5, seed=17)
    assert np.array_equal(first.keep_mask, second.keep_mask)
    assert set(np.unique(values[:, [0, 2]])) <= set(np.unique(first.values[:, [0, 2]]))
    assert first.mask_sha256 == second.mask_sha256


def test_zero_deletion_keeps_every_edge() -> None:
    values = np.asarray([[0, 0, 1, 0], [1, 0, 0, 1]], dtype=np.int64)
    result = delete_training_edges(values, rate=0.0, seed=17)
    assert result.keep_mask.tolist() == [True, True]
    assert result.actual_rate == 0.0


def test_invalid_deletion_rate_is_rejected() -> None:
    values = np.asarray([[0, 0, 1, 0]], dtype=np.int64)
    for rate in (-0.1, 1.0):
        try:
            delete_training_edges(values, rate=rate, seed=17)
        except ValueError as error:
            assert "rate" in str(error)
        else:
            raise AssertionError("invalid rate was accepted")


def test_deletion_always_protects_each_entity_earliest_fact() -> None:
    values = np.asarray(
        [
            [0, 0, 1, 0],
            [0, 0, 2, 1],
            [1, 0, 2, 1],
            [0, 0, 1, 2],
        ],
        dtype=np.int64,
    )
    for seed in range(20):
        result = delete_training_edges(values, rate=0.5, seed=seed)
        assert result.keep_mask[0]
        assert result.keep_mask[1]
