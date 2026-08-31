import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.export_expansion_analysis import (
    build_deletion_effects,
    build_method_contrasts,
    build_sampling_contrasts,
    export_expansion_analysis,
    summarize_conditions,
    summarize_multi_answer,
)


def _window_rows(sampling: str, adjustment: float = 0.0) -> pd.DataFrame:
    records = []
    for deletion_rate in (0.0, 0.3):
        for timestamp in (10, 11):
            query_count = 10 if timestamp == 10 else 30
            for method, coverage, size in (
                ("static_margin", 0.80, 4.0),
                ("rolling_margin", 0.90, 6.0),
                ("kgcp_softmax_static", 0.82, 5.0),
            ):
                records.append(
                    {
                        "seed": 17,
                        "deletion_rate": deletion_rate,
                        "method": method,
                        "timestamp": timestamp,
                        "query_count": query_count,
                        "dataset_mode": "toy",
                        "model_name": "temporal_distmult",
                        "negative_sampling": sampling,
                        "coverage": coverage + adjustment - 0.01 * deletion_rate,
                        "mean_size": size,
                        "mrr": 0.30,
                        "frequency_mrr": 0.10,
                    }
                )
    return pd.DataFrame(records)


def _query_rows(sampling: str) -> pd.DataFrame:
    records = []
    for deletion_rate in (0.0, 0.3):
        for method in ("static_margin", "rolling_margin"):
            records.append(
                {
                    "seed": 17,
                    "deletion_rate": deletion_rate,
                    "method": method,
                    "prediction_side": "object",
                    "timestamp": 10,
                    "subject_id": 1,
                    "relation_id": 2,
                    "true_object_id": 3,
                    "answer_count": 1,
                    "is_multi_answer": False,
                    "set_size": 4,
                    "covered": True,
                    "dataset_mode": "toy",
                    "model_name": "temporal_distmult",
                    "negative_sampling": sampling,
                }
            )
            for object_id, covered in ((4, True), (5, method == "rolling_margin")):
                records.append(
                    {
                        "seed": 17,
                        "deletion_rate": deletion_rate,
                        "method": method,
                        "prediction_side": "object",
                        "timestamp": 11,
                        "subject_id": 6,
                        "relation_id": 7,
                        "true_object_id": object_id,
                        "answer_count": 2,
                        "is_multi_answer": True,
                        "set_size": 8,
                        "covered": covered,
                        "dataset_mode": "toy",
                        "model_name": "temporal_distmult",
                        "negative_sampling": sampling,
                    }
                )
    return pd.DataFrame(records)


def test_condition_contrasts_preserve_reliability_and_deletion_direction() -> None:
    filtered = summarize_conditions(_window_rows("filtered"), "filtered_run")
    uniform = summarize_conditions(
        _window_rows("uniform", adjustment=-0.02), "uniform_run"
    )
    rows = pd.concat([filtered, uniform], ignore_index=True)

    rolling = filtered[
        (filtered["method"] == "rolling_margin")
        & (filtered["deletion_rate"] == 0.0)
    ].iloc[0]
    assert rolling["coverage"] == 0.90
    effects = build_deletion_effects(filtered)
    assert set(effects["coverage_change_vs_delete0"].round(3)) == {-0.003}

    method = build_method_contrasts(filtered, target_coverage=0.90)
    static = method[
        (method["baseline"] == "static_margin")
        & (method["deletion_rate"] == 0.0)
    ].iloc[0]
    assert static["coverage_gain"] == pytest.approx(0.10)
    assert static["undercoverage_reduction"] == pytest.approx(0.10)

    sampling = build_sampling_contrasts(rows)
    assert set(sampling["coverage_filtered_minus_uniform"].round(2)) == {0.02}


def test_multi_answer_summary_reports_full_set_degradation() -> None:
    by_group, degradation = summarize_multi_answer(
        _query_rows("filtered"), "filtered_run"
    )

    static_multi = by_group[
        (by_group["method"] == "static_margin")
        & (by_group["answer_group"] == "multi")
        & (by_group["deletion_rate"] == 0.0)
    ].iloc[0]
    rolling_multi = by_group[
        (by_group["method"] == "rolling_margin")
        & (by_group["answer_group"] == "multi")
        & (by_group["deletion_rate"] == 0.0)
    ].iloc[0]
    assert static_multi["full_set_coverage"] == 0.0
    assert static_multi["partial_answer_recall"] == 0.5
    assert rolling_multi["full_set_coverage"] == 1.0
    static_drop = degradation[
        (degradation["method"] == "static_margin")
        & (degradation["deletion_rate"] == 0.0)
    ].iloc[0]
    assert static_drop["full_set_coverage_multi_minus_single"] == -1.0


def test_full_export_accepts_relocated_complete_runs_and_writes_checksums(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "matrix"
    runs = {}
    for key, sampling, adjustment in (
        ("filtered_run", "filtered", 0.0),
        ("uniform_run", "uniform", -0.02),
    ):
        run_root = matrix / key / "run-1"
        metrics = run_root / "metrics"
        metrics.mkdir(parents=True)
        _window_rows(sampling, adjustment).to_csv(
            metrics / "per_window.csv", index=False
        )
        _query_rows(sampling).to_parquet(metrics / "per_query.parquet", index=False)
        (run_root / "run_manifest.json").write_text(
            json.dumps({"status": "complete"}) + "\n", encoding="utf-8"
        )
        runs[key] = {
            "status": "complete",
            "run_root": f"/relocated/{key}",
        }
    (matrix / "matrix_progress.json").write_text(
        json.dumps({"git_commit": "a" * 40, "runs": runs}) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "analysis"
    manifest = export_expansion_analysis(
        matrix,
        output,
        expected_runs=("filtered_run", "uniform_run"),
    )

    assert manifest["git_commit"] == "a" * 40
    assert (output / "analysis_manifest.json").is_file()
    checksums = (output / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert "condition_aggregate.csv" in checksums
    assert "analysis_manifest.json" in checksums
