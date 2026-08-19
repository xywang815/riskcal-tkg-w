from pathlib import Path

import numpy as np
import pandas as pd

import scripts.export_timestamp_block_bootstrap as bootstrap
from scripts.export_timestamp_block_bootstrap import (
    _bootstrap_statistic,
    _rolling_static_frames,
    _sample_circular_block_indices,
    export_timestamp_block_bootstrap,
)


def test_circular_block_sampler_returns_contiguous_wrapped_blocks() -> None:
    rng = np.random.default_rng(1)
    indices = _sample_circular_block_indices(rng, item_count=5, block_length=3)

    assert len(indices) == 5
    assert set(indices).issubset(set(range(5)))
    assert (indices[1] - indices[0]) % 5 == 1
    assert (indices[2] - indices[1]) % 5 == 1


def test_rolling_static_undercoverage_reduction_is_positive() -> None:
    rows = pd.DataFrame(
        [
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "static",
                "timestamp": 10,
                "coverage": 0.80,
                "query_count": 10,
            },
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "method": "rolling",
                "timestamp": 10,
                "coverage": 0.90,
                "query_count": 10,
            },
        ]
    )

    undercoverage, coverage_gain = _rolling_static_frames(rows, 0.3, 0.90)

    assert abs(undercoverage.iloc[0]["undercoverage_reduction"] - 0.10) < 1e-12
    assert abs(coverage_gain.iloc[0]["coverage_gain"] - 0.10) < 1e-12


def test_block_bootstrap_reports_observed_seed_average() -> None:
    frame = pd.DataFrame(
        [
            {"seed": 17, "timestamp": 1, "value": 0.1, "weight": 1},
            {"seed": 17, "timestamp": 2, "value": 0.3, "weight": 1},
            {"seed": 29, "timestamp": 1, "value": 0.2, "weight": 2},
            {"seed": 29, "timestamp": 2, "value": 0.4, "weight": 2},
        ]
    )

    result = _bootstrap_statistic(
        frame,
        "value",
        weight_column="weight",
        block_length=1,
        iterations=20,
        bootstrap_seed=7,
    )

    assert result["seed_count"] == 2
    assert result["timestamp_count"] == 2
    assert result["observed"] == 0.25


def test_block_bootstrap_uses_one_time_draw_per_replicate(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_block_indices(
        rng: np.random.Generator, item_count: int, block_length: int
    ) -> np.ndarray:
        calls.append((item_count, block_length))
        return np.asarray([1, 0], dtype=np.int64)

    monkeypatch.setattr(bootstrap, "_sample_circular_block_indices", fake_block_indices)
    frame = pd.DataFrame(
        [
            {"seed": 17, "timestamp": 1, "value": 0.1},
            {"seed": 17, "timestamp": 2, "value": 0.3},
            {"seed": 29, "timestamp": 1, "value": 0.2},
            {"seed": 29, "timestamp": 2, "value": 0.4},
        ]
    )

    bootstrap._bootstrap_statistic(
        frame,
        "value",
        weight_column=None,
        block_length=1,
        iterations=3,
        bootstrap_seed=7,
    )

    assert calls == [(2, 1), (2, 1), (2, 1)]


def test_export_writes_seed_effects(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    metrics = run_root / "metrics"
    metrics.mkdir(parents=True)
    records = []
    for seed, adjustment in ((17, 0.0), (29, 0.01)):
        for timestamp in range(1, 3):
            for method, coverage in (
                ("static", 0.82 + adjustment),
                ("rolling", 0.90 + adjustment),
            ):
                records.append(
                    {
                        "seed": seed,
                        "timestamp": timestamp,
                        "deletion_rate": 0.3,
                        "method": method,
                        "coverage": coverage,
                        "query_count": 10,
                        "mrr": 0.31,
                        "frequency_mrr": 0.09,
                    }
                )
    pd.DataFrame(records).to_csv(metrics / "per_window.csv", index=False)

    export_timestamp_block_bootstrap(
        run_root,
        tmp_path / "paper",
        block_length=1,
        iterations=20,
        bootstrap_seed=3,
    )

    seed_effects = pd.read_csv(
        tmp_path
        / "paper"
        / "data"
        / "final_confirmatory"
        / "timestamp_block_seed_effects.csv"
    )
    assert set(seed_effects["statistic"]) == {
        "rolling_undercoverage_reduction_vs_static",
        "rolling_micro_coverage_gain_vs_static",
        "scorer_mrr_gain_vs_frequency",
    }
    assert set(seed_effects["seed"]) == {17, 29}
