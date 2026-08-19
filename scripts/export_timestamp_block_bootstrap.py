"""Export timestamp-block bootstrap diagnostics for the confirmatory TKG run."""

from argparse import ArgumentParser
from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BOOTSTRAP_SEED = 20260816
DEFAULT_BLOCK_LENGTH = 7
DEFAULT_ITERATIONS = 20_000


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g", lineterminator="\n")
    temporary.replace(path)


def _sample_circular_block_indices(
    rng: np.random.Generator, item_count: int, block_length: int
) -> np.ndarray:
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    block_count = math.ceil(item_count / block_length)
    starts = rng.integers(0, item_count, size=block_count)
    indices: list[int] = []
    for start in starts:
        indices.extend((int(start) + offset) % item_count for offset in range(block_length))
    return np.asarray(indices[:item_count], dtype=np.int64)


def _weighted_average(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        return float(np.mean(values))
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return float(np.sum(values * weights) / total)


def _prepare_series(
    frame: pd.DataFrame,
    value_column: str,
    *,
    weight_column: str | None = None,
) -> tuple[list[int], np.ndarray, dict[int, tuple[np.ndarray, np.ndarray | None]]]:
    required = {"seed", "timestamp", value_column}
    if weight_column is not None:
        required.add(weight_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    timestamps = np.asarray(sorted(int(value) for value in frame["timestamp"].unique()))
    if len(timestamps) == 0:
        raise ValueError("no timestamps available for bootstrap")
    seeds = sorted(int(value) for value in frame["seed"].unique())
    series: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    for seed, seed_rows in frame.groupby("seed", sort=True):
        ordered = (
            seed_rows.assign(timestamp=seed_rows["timestamp"].astype(int))
            .set_index("timestamp")
            .reindex(timestamps)
        )
        if ordered[value_column].isna().any():
            raise ValueError(f"seed {seed} is missing one or more timestamps")
        values = ordered[value_column].to_numpy(dtype=float)
        weights: np.ndarray | None = None
        if weight_column is not None:
            if ordered[weight_column].isna().any():
                raise ValueError(f"seed {seed} is missing one or more weights")
            weights = ordered[weight_column].to_numpy(dtype=float)
            if np.any(weights <= 0):
                raise ValueError(f"seed {seed} contains nonpositive weights")
        series[int(seed)] = (values, weights)
    return seeds, timestamps, series


def _bootstrap_statistic(
    frame: pd.DataFrame,
    value_column: str,
    *,
    weight_column: str | None,
    block_length: int,
    iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Bootstrap a seed-averaged statistic with shared resampled time blocks."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    seeds, timestamps, series = _prepare_series(
        frame, value_column, weight_column=weight_column
    )
    observed_seed_values = [
        _weighted_average(values, weights) for values, weights in series.values()
    ]
    observed = float(np.mean(observed_seed_values))
    rng = np.random.default_rng(bootstrap_seed)
    seed_array = np.asarray(seeds, dtype=np.int64)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_seeds = rng.choice(seed_array, size=len(seed_array), replace=True)
        positions = _sample_circular_block_indices(
            rng, len(timestamps), block_length
        )
        seed_values: list[float] = []
        for seed in sampled_seeds:
            values, weights = series[int(seed)]
            sampled_weights = None if weights is None else weights[positions]
            seed_values.append(_weighted_average(values[positions], sampled_weights))
        samples[index] = float(np.mean(seed_values))
    return {
        "observed": observed,
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "pvalue_positive": float(
            (np.count_nonzero(samples <= 0.0) + 1) / (iterations + 1)
        ),
        "seed_count": len(seeds),
        "timestamp_count": int(len(timestamps)),
        "observed_by_seed": {
            str(seed): float(value) for seed, value in zip(seeds, observed_seed_values, strict=True)
        },
    }


def _rolling_static_frames(
    rows: pd.DataFrame, deletion_rate: float, target: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = rows[
        (rows["deletion_rate"] == deletion_rate)
        & (rows["method"].isin(["static", "rolling"]))
    ].copy()
    required = {"seed", "timestamp", "method", "coverage", "query_count"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"per-window metrics are missing columns: {missing}")
    pivot = selected.pivot_table(
        index=["seed", "timestamp"],
        columns="method",
        values=["coverage", "query_count"],
        aggfunc="first",
    )
    if pivot.isna().any().any():
        raise ValueError("static and rolling rows must exist for every seed/timestamp")
    base = pivot.reset_index()
    undercoverage = pd.DataFrame(
        {
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "undercoverage_reduction": np.maximum(
                target - base[("coverage", "static")], 0.0
            )
            - np.maximum(target - base[("coverage", "rolling")], 0.0),
        }
    )
    coverage_gain = pd.DataFrame(
        {
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "query_count": base[("query_count", "static")].astype(float),
            "coverage_gain": base[("coverage", "rolling")]
            - base[("coverage", "static")],
        }
    )
    return undercoverage, coverage_gain


def _mrr_frame(rows: pd.DataFrame, deletion_rate: float) -> pd.DataFrame:
    selected = rows[
        (rows["deletion_rate"] == deletion_rate) & (rows["method"] == "static")
    ].copy()
    required = {"seed", "timestamp", "query_count", "mrr", "frequency_mrr"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"per-window metrics are missing columns: {missing}")
    selected["mrr_gain"] = selected["mrr"] - selected["frequency_mrr"]
    return selected[["seed", "timestamp", "query_count", "mrr_gain"]]


def export_timestamp_block_bootstrap(
    run_root: Path,
    paper_root: Path,
    *,
    deletion_rate: float | None = None,
    target: float = 0.90,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    iterations: int = DEFAULT_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rows = pd.read_csv(run_root / "metrics" / "per_window.csv")
    if deletion_rate is None:
        deletion_rate = float(rows["deletion_rate"].max())
    if not 0.0 <= deletion_rate < 1.0:
        raise ValueError("deletion_rate must be in [0, 1)")
    if not 0.0 < target < 1.0:
        raise ValueError("target must be between 0 and 1")
    undercoverage, coverage_gain = _rolling_static_frames(rows, deletion_rate, target)
    mrr = _mrr_frame(rows, deletion_rate)
    statistics = {
        "rolling_undercoverage_reduction_vs_static": _bootstrap_statistic(
            undercoverage,
            "undercoverage_reduction",
            weight_column=None,
            block_length=block_length,
            iterations=iterations,
            bootstrap_seed=bootstrap_seed,
        ),
        "rolling_micro_coverage_gain_vs_static": _bootstrap_statistic(
            coverage_gain,
            "coverage_gain",
            weight_column="query_count",
            block_length=block_length,
            iterations=iterations,
            bootstrap_seed=bootstrap_seed + 1,
        ),
        "scorer_mrr_gain_vs_frequency": _bootstrap_statistic(
            mrr,
            "mrr_gain",
            weight_column="query_count",
            block_length=block_length,
            iterations=iterations,
            bootstrap_seed=bootstrap_seed + 2,
        ),
    }
    records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    for name, result in statistics.items():
        records.append(
            {
                "statistic": name,
                "deletion_rate": deletion_rate,
                "target": target,
                "observed": result["observed"],
                "ci95_low": result["ci95"][0],
                "ci95_high": result["ci95"][1],
                "pvalue_positive": result["pvalue_positive"],
                "iterations": iterations,
                "block_length": block_length,
                "seed_count": result["seed_count"],
                "timestamp_count": result["timestamp_count"],
                "weighting": (
                    "query_count" if name != "rolling_undercoverage_reduction_vs_static" else "timestamp_macro"
                ),
            }
        )
        for seed, value in result["observed_by_seed"].items():
            seed_records.append(
                {
                    "statistic": name,
                    "deletion_rate": deletion_rate,
                    "target": target,
                    "seed": int(seed),
                    "observed": value,
                    "weighting": (
                        "query_count"
                        if name != "rolling_undercoverage_reduction_vs_static"
                        else "timestamp_macro"
                    ),
                }
            )
    summary = pd.DataFrame(records)
    data_dir = paper_root / "data" / "final_confirmatory"
    _write_csv(summary, data_dir / "timestamp_block_bootstrap_summary.csv")
    _write_csv(
        pd.DataFrame(seed_records),
        data_dir / "timestamp_block_seed_effects.csv",
    )
    manifest = {
        "bootstrap_seed": bootstrap_seed,
        "block_length": block_length,
        "block_scheme": (
            "seed-resampled circular moving-block bootstrap with one shared "
            "timestamp-block draw applied to all sampled seeds in each replicate"
        ),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "deletion_rate": deletion_rate,
        "iterations": iterations,
        "run_root": str(run_root),
        "statistics": statistics,
        "target": target,
    }
    _write_json(manifest, data_dir / "timestamp_block_bootstrap.json")
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--deletion-rate", type=float)
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    manifest = export_timestamp_block_bootstrap(
        args.run_root,
        args.paper_root,
        deletion_rate=args.deletion_rate,
        target=args.target,
        block_length=args.block_length,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(manifest["statistics"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
