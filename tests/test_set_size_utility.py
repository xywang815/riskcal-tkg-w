import math

import pandas as pd
import pytest

from scripts.export_query_level_diagnostics import build_query_level_rows
from scripts.export_set_size_utility import (
    add_uncapped_deltas,
    aggregate_over_seeds,
    build_paper_table,
    parse_caps,
    summarize_operating_points,
)


def _per_label_rows() -> pd.DataFrame:
    base = {
        "seed": 17,
        "deletion_rate": 0.3,
        "method": "rolling",
        "prediction_side": "object",
        "relation_id": 2,
        "rank": 1,
        "frequency_rank": 3,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "timestamp": 1,
                "subject_id": 1,
                "true_object_id": 10,
                "set_size": 100,
                "covered": True,
                "top1_correct": True,
            },
            {
                **base,
                "timestamp": 1,
                "subject_id": 1,
                "true_object_id": 11,
                "set_size": 100,
                "covered": True,
                "top1_correct": False,
            },
            {
                **base,
                "timestamp": 1,
                "subject_id": 2,
                "true_object_id": 12,
                "set_size": 1000,
                "covered": True,
                "top1_correct": False,
            },
            {
                **base,
                "timestamp": 2,
                "subject_id": 3,
                "true_object_id": 13,
                "set_size": 5000,
                "covered": True,
                "top1_correct": False,
            },
        ]
    )


def test_parse_caps_keeps_infinity_as_uncapped() -> None:
    assert parse_caps("1000,500,inf,1000") == (500.0, 1000.0, math.inf)


def test_set_size_cap_metrics_make_abstention_explicit() -> None:
    queries = build_query_level_rows(_per_label_rows())
    rows = add_uncapped_deltas(
        summarize_operating_points(
            queries,
            caps=(1000.0, math.inf),
            num_entities=5000,
        )
    )

    capped = rows[rows["cap_label"] == "1000"].iloc[0]
    uncapped = rows[rows["cap_label"] == "inf"].iloc[0]

    assert capped["query_count"] == 3
    assert capped["answer_rate"] == pytest.approx(2 / 3)
    assert capped["conditional_full_set_coverage"] == pytest.approx(1.0)
    assert capped["unconditional_full_set_recall"] == pytest.approx(2 / 3)
    assert capped["conditional_label_recall"] == pytest.approx(1.0)
    assert capped["unconditional_label_recall"] == pytest.approx(3 / 4)
    assert capped["candidate_load"] == pytest.approx((100 + 1000) / 3)
    assert capped["mean_answered_set_size"] == pytest.approx(550.0)
    assert capped["top1_accuracy_on_answered"] == pytest.approx(0.5)
    assert capped["full_vocabulary_answered_rate"] == pytest.approx(0.0)

    assert uncapped["unconditional_full_set_recall"] == pytest.approx(1.0)
    assert capped["candidate_load_saved_vs_uncapped"] == pytest.approx(5000 / 3)
    assert capped["unconditional_full_set_recall_loss_vs_uncapped"] == pytest.approx(1 / 3)


def test_aggregate_and_paper_table_include_route_a_utility_columns() -> None:
    queries = build_query_level_rows(_per_label_rows())
    by_seed = add_uncapped_deltas(
        summarize_operating_points(
            queries,
            caps=(1000.0, math.inf),
            num_entities=5000,
        )
    )
    summary = aggregate_over_seeds(by_seed)
    table = build_paper_table(
        summary,
        deletion_rate=0.3,
        methods=("rolling",),
        caps=(1000.0, math.inf),
    )

    assert list(table["cap_label"]) == ["1000", "inf"]
    assert "answer_rate_mean" in table.columns
    assert "candidate_load_reduction_vs_uncapped_mean" in table.columns
    assert "unconditional_full_set_recall_mean" in table.columns
