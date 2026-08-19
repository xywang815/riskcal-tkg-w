import numpy as np
import pandas as pd
import pytest

from scripts.export_delay_feedback_sensitivity import (
    bootstrap_delay_effects,
    release_due_feedback,
    summarize_delay_rows,
)
from scripts.export_window_ablation import ScoreHistory


def test_release_due_feedback_uses_release_index() -> None:
    history = ScoreHistory()
    history.add(0, np.asarray([0.0]))
    pending = [
        (1, 1, np.asarray([1.0])),
        (3, 2, np.asarray([2.0])),
    ]

    pending = release_due_feedback(history, pending, current_index=1)

    scores, timestamps = history.values_before(10)
    assert scores.tolist() == [0.0, 1.0]
    assert timestamps.tolist() == [0, 1]
    assert len(pending) == 1


def test_delay_effects_use_positive_undercoverage_reduction() -> None:
    records = []
    for seed, offset in ((17, 0.0), (29, 0.0)):
        for timestamp in range(1, 5):
            for delay, rolling_coverage in ((0, 0.90), (1, 0.85)):
                for method, coverage in (
                    ("static", 0.80),
                    ("rolling", rolling_coverage),
                ):
                    records.append(
                        {
                            "seed": seed,
                            "deletion_rate": 0.3,
                            "extra_delay_blocks": delay,
                            "method": method,
                            "timestamp": timestamp,
                            "query_count": 10,
                            "coverage": coverage + offset,
                            "mean_size": 5.0,
                            "median_size": 4.0,
                            "p90_size": 9.0,
                            "pool_score_count": 100,
                            "pool_span_blocks": 3,
                        }
                    )
    rows = pd.DataFrame(records)

    summary = summarize_delay_rows(rows, target_coverage=0.9)
    effects = bootstrap_delay_effects(
        rows,
        deletion_rate=0.3,
        target_coverage=0.9,
        block_length=1,
        iterations=50,
        bootstrap_seed=7,
    )

    assert not summary.empty
    primary = effects[
        effects["statistic"] == "rolling_undercoverage_reduction_vs_static"
    ].sort_values("extra_delay_blocks")
    assert primary["observed"].tolist() == pytest.approx([0.10, 0.05])

    micro = effects[
        effects["statistic"] == "rolling_micro_coverage_gain_vs_static"
    ].sort_values("extra_delay_blocks")
    assert micro["observed"].tolist() == pytest.approx([0.10, 0.05])
