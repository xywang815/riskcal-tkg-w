import math

import numpy as np
import pandas as pd
import pytest

from scripts.export_window_ablation import (
    AdaptiveAlphaState,
    ScoreHistory,
    aggregate_method_summary,
    build_interaction_effects,
    effective_sample_size,
    summarize_method_rows,
    weighted_threshold_from_history,
)


def test_aci_controller_tightens_after_excess_miscoverage() -> None:
    state = AdaptiveAlphaState(alpha=0.1, gamma=0.05, target_error=0.1)
    state.update(observed_error=0.3)
    assert state.alpha == pytest.approx(0.09)
    state.update(observed_error=0.0)
    assert state.alpha == pytest.approx(0.095)


def test_score_history_keeps_most_recent_scores_by_count() -> None:
    history = ScoreHistory()
    history.add(1, np.asarray([10.0, 11.0]))
    history.add(2, np.asarray([20.0, 21.0]))
    history.add(3, np.asarray([30.0]))

    scores, timestamps = history.values_before(4, max_count=3)

    assert scores.tolist() == [20.0, 21.0, 30.0]
    assert timestamps.tolist() == [2, 2, 3]


def test_score_history_uses_strictly_earlier_time_window() -> None:
    history = ScoreHistory()
    history.add(5, np.asarray([5.0]))
    history.add(6, np.asarray([6.0]))
    history.add(7, np.asarray([7.0]))

    scores, timestamps = history.values_before(8, lookback_blocks=2)

    assert scores.tolist() == [6.0, 7.0]
    assert timestamps.tolist() == [6, 7]


def test_effective_sample_size_matches_equal_weights_for_infinite_half_life() -> None:
    timestamps = np.asarray([1, 1, 2, 3])
    assert effective_sample_size(timestamps, 4, math.inf) == pytest.approx(4.0)


def test_weighted_threshold_prefers_recent_scores() -> None:
    scores = np.asarray([10.0, 1.0])
    timestamps = np.asarray([1, 9])
    threshold = weighted_threshold_from_history(
        scores,
        timestamps,
        timestamp=10,
        target_coverage=0.5,
        half_life=1.0,
    )
    assert threshold == 1.0


def test_method_summary_and_interaction_use_positive_undercoverage() -> None:
    rows = pd.DataFrame(
        [
            {
                "seed": 17,
                "deletion_rate": 0.0,
                "method": "static",
                "timestamp": 1,
                "query_count": 10,
                "coverage": 0.80,
                "mean_size": 5.0,
                "median_size": 4.0,
                "p90_size": 9.0,
            },
            {
                "seed": 17,
                "deletion_rate": 0.0,
                "method": "rolling_count_1000",
                "timestamp": 1,
                "query_count": 10,
                "coverage": 0.90,
                "mean_size": 6.0,
                "median_size": 5.0,
                "p90_size": 10.0,
            },
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "static",
                "timestamp": 1,
                "query_count": 10,
                "coverage": 0.70,
                "mean_size": 7.0,
                "median_size": 6.0,
                "p90_size": 11.0,
            },
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "rolling_count_1000",
                "timestamp": 1,
                "query_count": 10,
                "coverage": 0.85,
                "mean_size": 8.0,
                "median_size": 7.0,
                "p90_size": 12.0,
            },
        ]
    )

    by_seed = summarize_method_rows(rows, target_coverage=0.9)
    aggregate = aggregate_method_summary(by_seed)
    interaction = build_interaction_effects(by_seed)

    static_30 = by_seed[
        (by_seed["method"] == "static") & (by_seed["deletion_rate"] == 0.3)
    ].iloc[0]
    assert static_30["positive_undercoverage"] == pytest.approx(0.2)
    assert not aggregate.empty
    gain_30 = interaction[interaction["deletion_rate"] == 0.3].iloc[0]
    assert gain_30["rolling1000_undercoverage_reduction_vs_static"] == pytest.approx(
        0.15
    )
    assert gain_30["rolling_gain_interaction_vs_delete0"] == pytest.approx(0.05)
