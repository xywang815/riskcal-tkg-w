import numpy as np
import pandas as pd
import pytest

from scripts.export_shortlist_calibration import (
    _summarize_mask,
    aggregate_shortlist_summary,
    bootstrap_shortlist_effects,
    build_shortlist_effect_frames,
    summarize_shortlist_rows,
    topk_prediction_mask,
    true_label_ranks,
)


def test_true_label_ranks_use_stable_entity_id_ties() -> None:
    scores = np.asarray(
        [
            [0.8, 0.8, 0.2],
            [0.1, 0.5, 0.9],
        ]
    )
    ranks = true_label_ranks(scores, np.asarray([1, 2]))
    mask = topk_prediction_mask(scores, 1)

    assert ranks.tolist() == [2.0, 1.0]
    assert mask.tolist() == [
        [True, False, False],
        [False, False, True],
    ]


def test_summarize_mask_reports_query_full_set_coverage() -> None:
    facts = np.asarray(
        [
            [0, 0, 1, 5],
            [0, 0, 2, 5],
            [3, 1, 0, 5],
        ],
        dtype=np.int64,
    )
    mask = np.asarray(
        [
            [False, True, False, False],
            [False, True, False, False],
            [True, False, False, False],
        ],
        dtype=bool,
    )

    row = _summarize_mask(
        seed=17,
        deletion_rate=0.3,
        method="rank_rolling",
        timestamp=5,
        facts=facts,
        prediction_sides=np.asarray(["object", "object", "subject"]),
        labels=facts[:, 2],
        ranks=np.asarray([1, 2, 1]),
        mask=mask,
        threshold=1,
        pool_score_count=100,
        pool_span_blocks=3,
        threshold_units="rank",
        num_entities=4,
    )

    assert row["label_row_count"] == 3
    assert row["unique_query_count"] == 2
    assert row["observed_label_coverage"] == pytest.approx(2 / 3)
    assert row["full_set_coverage"] == pytest.approx(0.5)
    assert row["partial_answer_recall"] == pytest.approx(0.75)
    assert row["mean_size"] == pytest.approx(1.0)


def test_summary_and_effects_compare_rank_shortlist_to_margin_baseline() -> None:
    records = []
    for seed, adjustment in ((17, 0.0), (29, 0.01)):
        for timestamp in range(1, 5):
            records.extend(
                [
                    {
                        "seed": seed,
                        "deletion_rate": 0.3,
                        "method": "margin_rolling",
                        "timestamp": timestamp,
                        "threshold": 2.0,
                        "threshold_units": "score_margin",
                        "pool_score_count": 100,
                        "pool_span_blocks": 3,
                        "label_row_count": 20,
                        "unique_query_count": 10,
                        "observed_label_coverage": 0.91 + adjustment,
                        "full_set_coverage": 0.88 + adjustment,
                        "partial_answer_recall": 0.90,
                        "mean_size": 500.0,
                        "median_size": 480.0,
                        "p90_size": 900.0,
                        "singleton_rate": 0.0,
                        "full_vocabulary_set_rate": 0.2,
                    },
                    {
                        "seed": seed,
                        "deletion_rate": 0.3,
                        "method": "rank_rolling",
                        "timestamp": timestamp,
                        "threshold": 125.0,
                        "threshold_units": "rank",
                        "pool_score_count": 100,
                        "pool_span_blocks": 3,
                        "label_row_count": 20,
                        "unique_query_count": 10,
                        "observed_label_coverage": 0.90 + adjustment,
                        "full_set_coverage": 0.87 + adjustment,
                        "partial_answer_recall": 0.89,
                        "mean_size": 125.0,
                        "median_size": 125.0,
                        "p90_size": 125.0,
                        "singleton_rate": 0.0,
                        "full_vocabulary_set_rate": 0.0,
                    },
                ]
            )
    rows = pd.DataFrame(records)

    by_seed = summarize_shortlist_rows(rows, target_coverage=0.9)
    summary = aggregate_shortlist_summary(by_seed)
    effects = build_shortlist_effect_frames(rows, deletion_rate=0.3)
    bootstrapped = bootstrap_shortlist_effects(
        rows,
        deletion_rate=0.3,
        block_length=1,
        iterations=30,
        bootstrap_seed=5,
    )

    rank = summary[summary["method"] == "rank_rolling"].iloc[0]
    assert rank["observed_label_coverage_mean"] == pytest.approx(0.905)
    assert effects["mean_size_reduction"].mean() == pytest.approx(375.0)
    size_effect = bootstrapped[
        bootstrapped["statistic"]
        == "rank_rolling_mean_size_reduction_vs_margin_rolling"
    ].iloc[0]
    assert size_effect["observed"] == pytest.approx(375.0)
