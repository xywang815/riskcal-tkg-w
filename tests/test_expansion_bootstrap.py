import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.export_expansion_bootstrap import (
    _deletion_effect_frame,
    _method_contrast_frames,
    _multi_answer_gap_frames,
    _sampling_effect_frame,
    bootstrap_statistic,
    export_expansion_bootstrap,
)


METHODS = (
    "static_margin",
    "rolling_margin",
    "kgcp_negscore_static",
    "kgcp_minmax_static",
    "kgcp_softmax_static",
)


def _window_rows(sampling: str, adjustment: float = 0.0) -> pd.DataFrame:
    records = []
    for seed in (17, 29):
        for deletion_rate in (0.0, 0.3):
            for timestamp, query_count in ((10, 10), (11, 30)):
                for method in METHODS:
                    base = {
                        "static_margin": 0.80,
                        "rolling_margin": 0.90,
                        "kgcp_negscore_static": 0.83,
                        "kgcp_minmax_static": 0.84,
                        "kgcp_softmax_static": 0.82,
                    }[method]
                    records.append(
                        {
                            "seed": seed,
                            "deletion_rate": deletion_rate,
                            "method": method,
                            "timestamp": timestamp,
                            "query_count": query_count,
                            "dataset_mode": "toy",
                            "model_name": "temporal_distmult",
                            "negative_sampling": sampling,
                            "coverage": base + adjustment - 0.01 * deletion_rate,
                        }
                    )
    return pd.DataFrame(records)


def _query_rows(sampling: str) -> pd.DataFrame:
    records = []
    for seed in (17, 29):
        for method in ("static_margin", "rolling_margin", "kgcp_softmax_static"):
            for timestamp in (10, 11):
                records.append(
                    {
                        "seed": seed,
                        "deletion_rate": 0.3,
                        "method": method,
                        "prediction_side": "object",
                        "timestamp": timestamp,
                        "subject_id": 1,
                        "relation_id": 2,
                        "true_object_id": 3,
                        "answer_count": 1,
                        "is_multi_answer": False,
                        "covered": True,
                    }
                )
                for object_id, covered in (
                    (4, True),
                    (5, method == "rolling_margin"),
                ):
                    records.append(
                        {
                            "seed": seed,
                            "deletion_rate": 0.3,
                            "method": method,
                            "prediction_side": "object",
                            "timestamp": timestamp,
                            "subject_id": 6,
                            "relation_id": 7,
                            "true_object_id": object_id,
                            "answer_count": 2,
                            "is_multi_answer": True,
                            "covered": covered,
                        }
                    )
    return pd.DataFrame(records)


def test_vectorized_bootstrap_reports_equal_seed_weighted_observed() -> None:
    rows = pd.DataFrame(
        [
            {"seed": 17, "timestamp": 1, "value": 0.1, "weight": 1},
            {"seed": 17, "timestamp": 2, "value": 0.3, "weight": 1},
            {"seed": 29, "timestamp": 1, "value": 0.2, "weight": 2},
            {"seed": 29, "timestamp": 2, "value": 0.4, "weight": 2},
        ]
    )

    result = bootstrap_statistic(
        rows,
        "value",
        weight_column="weight",
        block_length=1,
        iterations=50,
        bootstrap_seed=7,
        chunk_size=11,
    )

    assert result["observed"] == pytest.approx(0.25)
    assert result["seed_count"] == 2
    assert result["timestamp_count"] == 2
    assert result["excluded_timestamp_count"] == 0


def test_contrast_frames_preserve_declared_directions() -> None:
    filtered = _window_rows("filtered")
    uniform = _window_rows("uniform", adjustment=-0.02)

    gain, reliability = _method_contrast_frames(
        filtered, 0.3, "static_margin", 0.9
    )
    assert set(gain["value"].round(3)) == {0.1}
    assert set(reliability["value"].round(3)) == {0.1}
    deletion = _deletion_effect_frame(filtered, "static_margin", 0.3)
    assert set(deletion["value"].round(3)) == {-0.003}
    sampling = _sampling_effect_frame(
        filtered, uniform, "rolling_margin", 0.3
    )
    assert set(sampling["value"].round(3)) == {0.02}


def test_multi_answer_gap_uses_full_query_answer_sets() -> None:
    gaps = _multi_answer_gap_frames(
        _query_rows("filtered"),
        0.3,
        ("static_margin", "rolling_margin", "kgcp_softmax_static"),
    )

    assert set(gaps["static_margin"]["value"]) == {-1.0}
    assert set(gaps["rolling_margin"]["value"]) == {0.0}


def test_full_export_requires_complete_runs_and_writes_checksums(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    filtered_key = "toy_filtered"
    uniform_key = "toy_uniform"
    runs = {}
    for key, sampling, adjustment in (
        (filtered_key, "filtered", 0.0),
        (uniform_key, "uniform", -0.02),
    ):
        root = matrix / key / "run-1"
        metrics = root / "metrics"
        metrics.mkdir(parents=True)
        _window_rows(sampling, adjustment).to_csv(
            metrics / "per_window.csv", index=False
        )
        _query_rows(sampling).to_parquet(metrics / "per_query.parquet", index=False)
        (root / "run_manifest.json").write_text(
            json.dumps({"status": "complete"}) + "\n", encoding="utf-8"
        )
        runs[key] = {"status": "complete", "run_root": f"/relocated/{key}"}
    (matrix / "matrix_progress.json").write_text(
        json.dumps({"git_commit": "b" * 40, "runs": runs}) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "bootstrap"
    manifest = export_expansion_bootstrap(
        matrix,
        output,
        expected_runs=(filtered_key, uniform_key),
        filtered_runs=(filtered_key,),
        sampling_pairs=((filtered_key, uniform_key),),
        block_lengths=(1,),
        iterations=20,
    )

    assert manifest["git_commit"] == "b" * 40
    assert manifest["statistics_count"] > 0
    checksums = (output / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert "expansion_bootstrap_summary.csv" in checksums
    assert "expansion_bootstrap_manifest.json" in checksums
