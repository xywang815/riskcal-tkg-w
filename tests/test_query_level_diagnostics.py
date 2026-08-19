import pandas as pd
import pytest

from scripts.export_query_level_diagnostics import (
    aggregate_query_level,
    build_paper_table,
    build_query_level_rows,
    summarize_query_level,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "static",
                "prediction_side": "object",
                "timestamp": 293,
                "subject_id": 1,
                "relation_id": 2,
                "true_object_id": 10,
                "set_size": 5,
                "covered": True,
                "rank": 1,
                "frequency_rank": 3,
                "top1_correct": True,
            },
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "static",
                "prediction_side": "object",
                "timestamp": 293,
                "subject_id": 1,
                "relation_id": 2,
                "true_object_id": 11,
                "set_size": 5,
                "covered": False,
                "rank": 9,
                "frequency_rank": 4,
                "top1_correct": False,
            },
            {
                # Duplicate observed label should not inflate answer_count.
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "static",
                "prediction_side": "object",
                "timestamp": 293,
                "subject_id": 1,
                "relation_id": 2,
                "true_object_id": 11,
                "set_size": 5,
                "covered": False,
                "rank": 9,
                "frequency_rank": 4,
                "top1_correct": False,
            },
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "static",
                "prediction_side": "subject",
                "timestamp": 293,
                "subject_id": 20,
                "relation_id": 232,
                "true_object_id": 1,
                "set_size": 10,
                "covered": True,
                "rank": 2,
                "frequency_rank": 8,
                "top1_correct": False,
            },
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "rolling",
                "prediction_side": "object",
                "timestamp": 293,
                "subject_id": 1,
                "relation_id": 2,
                "true_object_id": 10,
                "set_size": 10,
                "covered": True,
                "rank": 1,
                "frequency_rank": 3,
                "top1_correct": True,
            },
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "rolling",
                "prediction_side": "object",
                "timestamp": 293,
                "subject_id": 1,
                "relation_id": 2,
                "true_object_id": 11,
                "set_size": 10,
                "covered": True,
                "rank": 9,
                "frequency_rank": 4,
                "top1_correct": False,
            },
        ]
    )


def test_query_level_rows_collapse_observed_labels_into_queries() -> None:
    queries = build_query_level_rows(_rows())
    static = queries[queries["method"] == "static"].sort_values("answer_count")

    assert len(static) == 2
    assert list(static["answer_count"]) == [1, 2]
    multi = static[static["multi_answer"]].iloc[0]
    assert not bool(multi["full_set_covered"])
    assert multi["partial_answer_recall"] == 0.5
    assert multi["covered_answer_count"] == 1


def test_query_level_summary_separates_full_set_from_label_weighted_recall() -> None:
    summary = summarize_query_level(build_query_level_rows(_rows()), num_entities=10)
    static = summary[summary["method"] == "static"].iloc[0]
    rolling = summary[summary["method"] == "rolling"].iloc[0]

    assert static["query_count"] == 2
    assert static["label_count"] == 3
    assert static["multi_answer_query_count"] == 1
    assert static["full_set_coverage"] == 0.5
    assert static["partial_answer_recall"] == 0.75
    assert static["label_weighted_partial_recall"] == pytest.approx(2 / 3)
    assert static["multi_answer_full_set_coverage"] == 0.0
    assert static["multi_answer_partial_recall"] == 0.5
    assert static["p90_size"] == pytest.approx(9.5)
    assert static["full_vocabulary_set_rate"] == 0.5

    assert rolling["query_count"] == 1
    assert rolling["full_set_coverage"] == 1.0
    assert rolling["partial_answer_recall"] == 1.0


def test_aggregate_and_paper_table_keep_route_a_columns() -> None:
    summary = summarize_query_level(build_query_level_rows(_rows()), num_entities=10)
    aggregate = aggregate_query_level(summary)
    table = build_paper_table(aggregate)

    assert set(table["method"]) == {"static", "rolling"}
    assert "full_set_coverage_mean" in table.columns
    assert "partial_answer_recall_mean" in table.columns
    assert "full_vocabulary_set_rate_mean" in table.columns
