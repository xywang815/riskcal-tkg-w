import numpy as np
import pandas as pd
import pytest

from riskcal_tkg.followup import (
    build_query_grouping,
    candidate_nonconformity,
    query_max_true_nonconformity,
    summarize_query_mask,
)
from scripts.export_followup_calibration_study import (
    aggregate_by_seed,
    build_query_objective_contrasts,
)


def test_candidate_nonconformity_matches_score_definitions() -> None:
    scores = np.asarray([[3.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    assert candidate_nonconformity(scores, "margin")[0].tolist() == [0.0, 2.0, 1.0]
    assert candidate_nonconformity(scores, "negscore")[0].tolist() == [-3.0, -1.0, -2.0]
    assert candidate_nonconformity(scores, "minmax")[0].tolist() == [-1.0, -0.0, -0.5]
    assert candidate_nonconformity(scores, "minmax")[1].tolist() == [0.0, 0.0, 0.0]
    softmax = candidate_nonconformity(scores, "softmax")
    assert np.all((softmax >= 0.0) & (softmax <= 1.0))


def test_query_max_score_and_metrics_use_unique_queries() -> None:
    facts = np.asarray(
        [
            [0, 0, 1, 4],
            [0, 0, 2, 4],
            [3, 1, 0, 4],
        ],
        dtype=np.int64,
    )
    candidates = np.asarray(
        [
            [0.0, 0.2, 0.7, 1.0],
            [0.0, 0.2, 0.7, 1.0],
            [0.3, 0.4, 0.5, 0.0],
        ]
    )
    grouping = build_query_grouping(facts)
    assert grouping.query_count == 2
    query_scores = query_max_true_nonconformity(candidates, facts, grouping)
    assert sorted(query_scores.tolist()) == [0.3, 0.7]

    unique_mask = candidates[grouping.first_indices] <= 0.4
    summary = summarize_query_mask(unique_mask, facts, grouping)
    assert summary["label_coverage"] == pytest.approx(2 / 3)
    assert summary["full_set_coverage"] == pytest.approx(0.5)
    assert summary["partial_answer_recall"] == pytest.approx(0.75)
    assert summary["single_query_count"] == 1
    assert summary["multi_query_count"] == 1


def test_aggregation_and_query_contrast_keep_estimands_separate() -> None:
    rows = []
    for objective, adjustment in (("label", 0.0), ("query_max", 0.1)):
        for timestamp, count in ((1, 10), (2, 30)):
            rows.append(
                {
                    "case": "toy",
                    "dataset_mode": "toy",
                    "model_name": "temporal_distmult",
                    "negative_sampling": "filtered",
                    "seed": 17,
                    "deletion_rate": 0.3,
                    "objective": objective,
                    "score": "margin",
                    "history": "rolling",
                    "method": f"{objective}_margin_rolling",
                    "label_count": count * 2,
                    "query_count": count,
                    "single_query_count": count // 2,
                    "multi_query_count": count // 2,
                    "label_coverage": 0.8 + adjustment,
                    "full_set_coverage": 0.7 + adjustment,
                    "partial_answer_recall": 0.75 + adjustment,
                    "mean_set_size": 4.0 + 10 * adjustment,
                    "single_full_set_coverage": 0.9,
                    "multi_full_set_coverage": 0.5 + adjustment,
                    "single_mean_set_size": 3.0,
                    "multi_mean_set_size": 5.0,
                }
            )
    by_seed = aggregate_by_seed(pd.DataFrame(rows))
    assert len(by_seed) == 2
    contrast = build_query_objective_contrasts(by_seed).iloc[0]
    assert contrast["full_set_coverage_query_minus_label"] == pytest.approx(0.1)
    assert contrast["mean_set_size_query_minus_label"] == pytest.approx(1.0)


def test_aggregation_ignores_undefined_zero_weight_subgroups() -> None:
    rows = pd.DataFrame(
        [
            {
                "case": "toy",
                "dataset_mode": "toy",
                "model_name": "temporal_distmult",
                "negative_sampling": "filtered",
                "seed": 17,
                "deletion_rate": 0.3,
                "objective": "label",
                "score": "margin",
                "history": "rolling",
                "method": "label_margin_rolling",
                "label_count": 10,
                "query_count": 10,
                "single_query_count": 10,
                "multi_query_count": 0,
                "label_coverage": 0.9,
                "full_set_coverage": 0.9,
                "partial_answer_recall": 0.9,
                "mean_set_size": 4.0,
                "single_full_set_coverage": 0.9,
                "multi_full_set_coverage": np.nan,
                "single_mean_set_size": 4.0,
                "multi_mean_set_size": np.nan,
            },
            {
                "case": "toy",
                "dataset_mode": "toy",
                "model_name": "temporal_distmult",
                "negative_sampling": "filtered",
                "seed": 17,
                "deletion_rate": 0.3,
                "objective": "label",
                "score": "margin",
                "history": "rolling",
                "method": "label_margin_rolling",
                "label_count": 12,
                "query_count": 10,
                "single_query_count": 8,
                "multi_query_count": 2,
                "label_coverage": 0.9,
                "full_set_coverage": 0.8,
                "partial_answer_recall": 0.85,
                "mean_set_size": 5.0,
                "single_full_set_coverage": 0.875,
                "multi_full_set_coverage": 0.5,
                "single_mean_set_size": 4.5,
                "multi_mean_set_size": 7.0,
            },
        ]
    )
    summary = aggregate_by_seed(rows).iloc[0]
    assert summary["multi_full_set_coverage"] == pytest.approx(0.5)
    assert summary["multi_mean_set_size"] == pytest.approx(7.0)
    assert summary["multi_minus_single_full_set_coverage"] == pytest.approx(
        0.5 - ((10 * 0.9 + 8 * 0.875) / 18)
    )
