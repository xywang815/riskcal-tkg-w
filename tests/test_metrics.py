import numpy as np

from riskcal_tkg.metrics import (
    coverage_and_size,
    filtered_rank,
    ranking_metrics,
    selective_metrics,
)


def test_filtered_rank_removes_other_true_answers() -> None:
    scores = np.asarray([0.1, 0.9, 0.8, 0.2])
    assert filtered_rank(scores, true_id=2, other_true_ids={1}) == 1


def test_coverage_and_size() -> None:
    mask = np.asarray([[True, False, True], [False, True, False]])
    result = coverage_and_size(mask, np.asarray([2, 0]))
    assert result.coverage == 0.5
    assert result.mean_size == 1.5
    assert result.median_size == 1.5


def test_selective_metrics_all_answer_and_all_abstain() -> None:
    sizes = np.asarray([1, 3])
    correct = np.asarray([True, False])
    answered = selective_metrics(sizes, correct, max_set_size=3)
    abstained = selective_metrics(sizes, correct, max_set_size=0)
    assert answered.answer_rate == 1.0 and answered.risk == 0.5
    assert abstained.answer_rate == 0.0 and np.isnan(abstained.risk)


def test_ranking_metrics_from_known_ranks() -> None:
    result = ranking_metrics(np.asarray([1, 2, 10]))
    assert np.isclose(result.mrr, (1 + 0.5 + 0.1) / 3)
    assert result.hits_at_1 == 1 / 3
    assert result.hits_at_3 == 2 / 3
    assert result.hits_at_10 == 1.0
