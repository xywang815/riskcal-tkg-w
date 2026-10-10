import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.export_time_encoder_audit import (
    bootstrap_statistic,
    build_timestamp_contrasts,
    export_time_encoder_audit,
)


def _rows() -> pd.DataFrame:
    records = []
    for seed, adjustment in ((17, 0.0), (29, 0.01)):
        for timestamp, query_count in ((10, 10), (11, 30)):
            for method, coverage, size, threshold in (
                ("static_margin", 0.80 + adjustment, 20.0, 0.4),
                ("rolling_margin", 0.90 + adjustment, 25.0, 0.5),
            ):
                records.append(
                    {
                        "case": "toy_distmult_none",
                        "dataset_mode": "toy",
                        "model_name": "temporal_distmult",
                        "time_encoding": "none",
                        "negative_sampling": "filtered",
                        "seed": seed,
                        "deletion_rate": 0.0,
                        "timestamp": timestamp,
                        "query_count": query_count,
                        "method": method,
                        "coverage": coverage,
                        "mean_size": size,
                        "threshold": threshold,
                        "score_global_min": -1.0,
                        "score_global_max": 1.0,
                        "score_global_range": 2.0,
                        "score_mean": 0.0,
                        "score_std": 0.5,
                        "mean_query_score_range": 1.5,
                        "true_score_mean": 0.3,
                        "true_margin_mean": 0.7,
                    }
                )
    return pd.DataFrame(records)


def test_timestamp_contrasts_align_static_and_rolling_rows() -> None:
    contrasted = build_timestamp_contrasts(_rows())
    assert len(contrasted) == 4
    assert set(contrasted["rolling_minus_static_coverage"].round(6)) == {0.1}
    assert set(contrasted["rolling_minus_static_mean_size"]) == {5.0}


def test_bootstrap_reports_equal_seed_weighted_observed() -> None:
    contrasted = build_timestamp_contrasts(_rows())
    result = bootstrap_statistic(
        contrasted,
        "rolling_minus_static_coverage",
        block_length=1,
        iterations=50,
        bootstrap_seed=7,
    )
    assert result["observed"] == pytest.approx(0.1)
    assert result["seed_count"] == 2
    assert result["timestamp_count"] == 2


def test_export_requires_a_complete_matrix(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    (matrix / "matrix_progress.json").write_text(
        json.dumps({"git_commit": "a" * 40, "runs": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        export_time_encoder_audit(
            matrix,
            tmp_path / "output",
            expected_runs=("toy_distmult_none",),
            block_lengths=(1,),
            iterations=10,
        )
