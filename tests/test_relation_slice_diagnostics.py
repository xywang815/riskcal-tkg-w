import pandas as pd
import pytest

from scripts.export_relation_slice_diagnostics import (
    add_relation_metadata,
    aggregate_relation_slices,
    build_paper_table,
    build_relation_slice_by_seed,
    build_worst_group_summary,
)


def _rows() -> pd.DataFrame:
    records = []
    for method, covered_values, size in [
        ("static", [True, False, False, True, False, True], 5),
        ("rolling", [True, True, True, True, False, True], 8),
    ]:
        for index, covered in enumerate(covered_values):
            prediction_side = "object" if index < 3 else "subject"
            relation_id = 1 if prediction_side == "object" else 4
            records.append(
                {
                    "seed": 17,
                    "deletion_rate": 0.3,
                    "method": method,
                    "prediction_side": prediction_side,
                    "timestamp": 293 + index,
                    "subject_id": 10 + index,
                    "relation_id": relation_id,
                    "true_object_id": 100 + index,
                    "set_size": size,
                    "covered": covered,
                    "rank": 1 if covered else 10,
                    "frequency_rank": 9,
                    "top1_correct": covered,
                }
            )
    return pd.DataFrame(records)


def test_relation_metadata_maps_inverse_relation_back_to_base_relation() -> None:
    rows = pd.DataFrame(
        {
            "relation_id": [1, 4],
            "prediction_side": ["object", "subject"],
        }
    )

    enriched = add_relation_metadata(
        rows,
        num_relations=3,
        relation_labels={1: "rel-1"},
    )

    assert enriched["base_relation_id"].tolist() == [1, 1]
    assert enriched["relation_label"].tolist() == ["rel-1", "rel-1"]
    assert enriched["relation_side"].tolist() == ["object:rel-1", "subject:rel-1"]


def test_relation_slice_summary_keeps_label_and_query_metrics() -> None:
    by_seed = build_relation_slice_by_seed(
        _rows(),
        target_coverage=0.9,
        num_entities=8,
        num_relations=3,
        relation_labels={1: "rel-1"},
    )

    static_object = by_seed[
        (by_seed["method"] == "static")
        & (by_seed["prediction_side"] == "object")
    ].iloc[0]
    rolling_object = by_seed[
        (by_seed["method"] == "rolling")
        & (by_seed["prediction_side"] == "object")
    ].iloc[0]

    assert static_object["label_count"] == 3
    assert static_object["unique_query_count"] == 3
    assert static_object["label_coverage"] == pytest.approx(1 / 3)
    assert static_object["positive_undercoverage"] == pytest.approx(0.9 - 1 / 3)
    assert static_object["query_full_set_coverage"] == pytest.approx(1 / 3)
    assert rolling_object["label_coverage"] == 1.0
    assert rolling_object["full_vocabulary_set_rate"] == 1.0


def test_worst_group_summary_filters_by_support_and_builds_paper_table() -> None:
    by_seed = build_relation_slice_by_seed(
        _rows(),
        target_coverage=0.9,
        num_entities=8,
        num_relations=3,
        relation_labels={1: "rel-1"},
    )
    relation_summary = aggregate_relation_slices(by_seed)
    worst = build_worst_group_summary(
        relation_summary,
        target_coverage=0.9,
        min_total_labels=3,
        min_seed_count=1,
    )
    paper = build_paper_table(worst)

    static = worst[worst["method"] == "static"].iloc[0]
    assert static["eligible_relation_side_groups"] == 2
    assert static["relation_side_coverage_min"] == pytest.approx(1 / 3)
    assert static["worst_prediction_side"] == "object"
    assert static["worst_relation_label"] == "rel-1"
    assert set(paper["method"]) == {"static", "rolling"}

    filtered = build_worst_group_summary(
        relation_summary,
        target_coverage=0.9,
        min_total_labels=4,
        min_seed_count=1,
    )
    assert filtered["eligible_relation_side_groups"].sum() == 0
