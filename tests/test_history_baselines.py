import numpy as np

from scripts.export_history_baselines import HistoryRanker, _ranking_metrics


def test_relation_frequency_rank_filters_other_true_answers() -> None:
    history = np.asarray(
        [
            [0, 0, 1, 0],
            [2, 0, 1, 0],
            [3, 0, 2, 0],
        ],
        dtype=np.int64,
    )
    ranker = HistoryRanker(4, history)

    assert ranker.relation_frequency_rank(0, 2, {1}) == 1


def test_repeat_rank_prioritizes_repeated_subject_relation_pair() -> None:
    history = np.asarray(
        [
            [0, 0, 2, 0],
            [0, 0, 2, 1],
            [1, 0, 3, 0],
        ],
        dtype=np.int64,
    )
    ranker = HistoryRanker(5, history)

    assert ranker.repeat_rank(0, 0, 2, set()) == 1
    assert ranker.repeat_rank(1, 0, 2, set()) > 1


def test_ranking_metrics_from_history_ranks() -> None:
    result = _ranking_metrics([1, 2, 4])

    assert np.isclose(result["mrr"], (1.0 + 0.5 + 0.25) / 3)
    assert result["hits_at_1"] == 1 / 3
    assert result["hits_at_3"] == 2 / 3
    assert result["hits_at_10"] == 1.0
