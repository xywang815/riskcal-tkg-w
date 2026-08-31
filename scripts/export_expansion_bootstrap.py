"""Export predeclared timestamp-block bootstrap results for E1-E6.

This script is intentionally separate from training and manuscript generation. It
requires a complete expansion matrix, validates every input run, and reports a
small set of reviewer-facing paired contrasts with temporal dependence preserved
through a circular moving-block bootstrap.
"""

from argparse import ArgumentParser
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


EXPECTED_RUNS = (
    "icews14_distmult_filtered",
    "icews14_tcomplex_filtered",
    "icews05_15_distmult_filtered",
    "icews05_15_tcomplex_filtered",
    "icews05_15_distmult_uniform_sensitivity",
    "icews14_tcomplex_uniform_sensitivity",
)
FILTERED_RUNS = (
    "icews14_distmult_filtered",
    "icews14_tcomplex_filtered",
    "icews05_15_distmult_filtered",
    "icews05_15_tcomplex_filtered",
)
SAMPLING_PAIRS = (
    (
        "icews05_15_distmult_filtered",
        "icews05_15_distmult_uniform_sensitivity",
    ),
    (
        "icews14_tcomplex_filtered",
        "icews14_tcomplex_uniform_sensitivity",
    ),
)
METHOD_BASELINES = (
    "static_margin",
    "kgcp_negscore_static",
    "kgcp_minmax_static",
    "kgcp_softmax_static",
)
QUERY_METHODS = (
    "static_margin",
    "rolling_margin",
    "kgcp_softmax_static",
)
DEFAULT_BLOCK_LENGTHS = (3, 7, 14, 21)
DEFAULT_BOOTSTRAP_SEED = 20260901
DEFAULT_ITERATIONS = 20_000
PRIMARY_DELETION_RATE = 0.30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g", lineterminator="\n")
    temporary.replace(path)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _single_value(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"expected one {column}, found {values.tolist()}")
    return str(values[0])


def _resolve_run_root(matrix_root: Path, key: str, record: dict[str, Any]) -> Path:
    recorded = Path(str(record.get("run_root", "")))
    if recorded.is_dir():
        return recorded
    candidates = [
        path.parent
        for path in (matrix_root / key).glob("*/run_manifest.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    ]
    if len(candidates) != 1:
        raise ValueError(f"could not resolve one complete run for {key}")
    return candidates[0]


def _load_complete_runs(
    matrix_root: Path, expected_runs: Iterable[str]
) -> tuple[str, dict[str, Path], dict[str, dict[str, str]]]:
    progress_path = matrix_root / "matrix_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    commit = progress.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("matrix progress must record a full Git commit")
    records = progress.get("runs")
    if not isinstance(records, dict):
        raise ValueError("matrix progress runs field must be a mapping")
    roots: dict[str, Path] = {}
    inputs: dict[str, dict[str, str]] = {}
    for key in expected_runs:
        record = records.get(key)
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise ValueError(f"required expansion run is incomplete: {key}")
        root = _resolve_run_root(matrix_root, key, record)
        manifest_path = root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"run manifest is incomplete: {root}")
        window_path = root / "metrics" / "per_window.csv"
        query_path = root / "metrics" / "per_query.parquet"
        roots[key] = root
        inputs[key] = {
            "run_root": str(root),
            "run_manifest_sha256": _sha256(manifest_path),
            "per_window_sha256": _sha256(window_path),
            "per_query_sha256": _sha256(query_path),
        }
    return commit, roots, inputs


def _prepare_matrix(
    frame: pd.DataFrame,
    value_column: str,
    *,
    weight_column: str | None,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray | None, int]:
    required = {"seed", "timestamp", value_column}
    if weight_column is not None:
        required.add(weight_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"bootstrap frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("bootstrap frame is empty")
    if frame.duplicated(["seed", "timestamp"]).any():
        raise ValueError("bootstrap frame has duplicate seed/timestamp rows")
    seeds = sorted(int(value) for value in frame["seed"].unique())
    timestamp_sets = [
        set(group["timestamp"].astype(int))
        for _, group in frame.groupby("seed", sort=True)
    ]
    common = set.intersection(*timestamp_sets)
    union = set.union(*timestamp_sets)
    if not common:
        raise ValueError("bootstrap frame has no timestamp shared by every seed")
    timestamps = np.asarray(sorted(common), dtype=np.int64)
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for seed in seeds:
        ordered = (
            frame[frame["seed"] == seed]
            .assign(timestamp=lambda rows: rows["timestamp"].astype(int))
            .set_index("timestamp")
            .reindex(timestamps)
        )
        if ordered[value_column].isna().any():
            raise ValueError(f"seed {seed} is missing a shared timestamp value")
        values.append(ordered[value_column].to_numpy(dtype=float))
        if weight_column is not None:
            if ordered[weight_column].isna().any():
                raise ValueError(f"seed {seed} is missing a shared timestamp weight")
            seed_weights = ordered[weight_column].to_numpy(dtype=float)
            if np.any(seed_weights <= 0):
                raise ValueError("bootstrap weights must be positive")
            weights.append(seed_weights)
    weight_matrix = np.stack(weights) if weights else None
    return (
        seeds,
        timestamps,
        np.stack(values),
        weight_matrix,
        len(union) - len(common),
    )


def _circular_block_positions(
    rng: np.random.Generator,
    replicate_count: int,
    item_count: int,
    block_length: int,
) -> np.ndarray:
    if item_count <= 0 or block_length <= 0 or replicate_count <= 0:
        raise ValueError("replicate count, item count, and block length must be positive")
    block_count = math.ceil(item_count / block_length)
    starts = rng.integers(0, item_count, size=(replicate_count, block_count))
    offsets = np.arange(block_length, dtype=np.int64)
    return ((starts[:, :, None] + offsets) % item_count).reshape(
        replicate_count, -1
    )[:, :item_count]


def bootstrap_statistic(
    frame: pd.DataFrame,
    value_column: str,
    *,
    weight_column: str | None,
    block_length: int,
    iterations: int,
    bootstrap_seed: int,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Bootstrap an equal-seed average with shared circular time-block draws."""
    if iterations <= 0 or chunk_size <= 0:
        raise ValueError("iterations and chunk_size must be positive")
    seeds, timestamps, values, weights, excluded = _prepare_matrix(
        frame, value_column, weight_column=weight_column
    )
    if weights is None:
        observed_by_seed = values.mean(axis=1)
    else:
        observed_by_seed = (values * weights).sum(axis=1) / weights.sum(axis=1)
    observed = float(observed_by_seed.mean())
    rng = np.random.default_rng(bootstrap_seed)
    samples = np.empty(iterations, dtype=float)
    seed_count, timestamp_count = values.shape
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        count = stop - start
        sampled_seeds = rng.integers(0, seed_count, size=(count, seed_count))
        positions = _circular_block_positions(
            rng, count, timestamp_count, block_length
        )
        selected_values = values[
            sampled_seeds[:, :, None], positions[:, None, :]
        ]
        if weights is None:
            samples[start:stop] = selected_values.mean(axis=(1, 2))
        else:
            selected_weights = weights[
                sampled_seeds[:, :, None], positions[:, None, :]
            ]
            per_seed = (selected_values * selected_weights).sum(axis=2) / (
                selected_weights.sum(axis=2)
            )
            samples[start:stop] = per_seed.mean(axis=1)
    return {
        "observed": observed,
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "pvalue_positive": float(
            (np.count_nonzero(samples <= 0.0) + 1) / (iterations + 1)
        ),
        "pvalue_negative": float(
            (np.count_nonzero(samples >= 0.0) + 1) / (iterations + 1)
        ),
        "seed_count": seed_count,
        "timestamp_count": timestamp_count,
        "excluded_timestamp_count": excluded,
        "observed_by_seed": {
            str(seed): float(value)
            for seed, value in zip(seeds, observed_by_seed, strict=True)
        },
    }


def _method_contrast_frames(
    rows: pd.DataFrame,
    deletion_rate: float,
    baseline: str,
    target: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = rows[
        np.isclose(rows["deletion_rate"], deletion_rate)
        & rows["method"].isin(["rolling_margin", baseline])
    ]
    pivot = selected.pivot(
        index=["seed", "timestamp"],
        columns="method",
        values=["coverage", "query_count"],
    )
    required = {
        ("coverage", "rolling_margin"),
        ("coverage", baseline),
        ("query_count", "rolling_margin"),
    }
    if not required.issubset(pivot.columns) or pivot[list(required)].isna().any().any():
        raise ValueError(f"incomplete rolling/{baseline} method contrast")
    base = pivot.reset_index()
    coverage = pd.DataFrame(
        {
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "query_count": base[("query_count", "rolling_margin")].astype(float),
            "value": base[("coverage", "rolling_margin")]
            - base[("coverage", baseline)],
        }
    )
    reliability = pd.DataFrame(
        {
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "value": np.maximum(target - base[("coverage", baseline)], 0.0)
            - np.maximum(target - base[("coverage", "rolling_margin")], 0.0),
        }
    )
    return coverage, reliability


def _deletion_effect_frame(
    rows: pd.DataFrame, method: str, deletion_rate: float
) -> pd.DataFrame:
    selected = rows[
        rows["method"].eq(method)
        & (
            np.isclose(rows["deletion_rate"], 0.0)
            | np.isclose(rows["deletion_rate"], deletion_rate)
        )
    ]
    pivot = selected.pivot(
        index=["seed", "timestamp"],
        columns="deletion_rate",
        values=["coverage", "query_count"],
    )
    rates = sorted(float(value) for value in selected["deletion_rate"].unique())
    if len(rates) != 2 or not math.isclose(rates[0], 0.0) or not math.isclose(
        rates[1], deletion_rate
    ):
        raise ValueError(f"missing deletion pair for {method} at {deletion_rate}")
    required = {
        ("coverage", rates[0]),
        ("coverage", rates[1]),
        ("query_count", rates[0]),
    }
    if pivot[list(required)].isna().any().any():
        raise ValueError(f"incomplete deletion contrast for {method}")
    base = pivot.reset_index()
    return pd.DataFrame(
        {
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "query_count": base[("query_count", rates[0])].astype(float),
            "value": base[("coverage", rates[1])]
            - base[("coverage", rates[0])],
        }
    )


def _sampling_effect_frame(
    filtered: pd.DataFrame,
    uniform: pd.DataFrame,
    method: str,
    deletion_rate: float,
) -> pd.DataFrame:
    columns = ["seed", "timestamp", "coverage", "query_count"]
    left = filtered[
        filtered["method"].eq(method)
        & np.isclose(filtered["deletion_rate"], deletion_rate)
    ][columns].rename(
        columns={"coverage": "coverage_filtered", "query_count": "query_count_filtered"}
    )
    right = uniform[
        uniform["method"].eq(method)
        & np.isclose(uniform["deletion_rate"], deletion_rate)
    ][columns].rename(
        columns={"coverage": "coverage_uniform", "query_count": "query_count_uniform"}
    )
    merged = left.merge(right, on=["seed", "timestamp"], how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("filtered and uniform runs do not share complete seed/timestamp support")
    if not np.array_equal(
        merged["query_count_filtered"].to_numpy(),
        merged["query_count_uniform"].to_numpy(),
    ):
        raise ValueError("filtered and uniform runs have different test query counts")
    return pd.DataFrame(
        {
            "seed": merged["seed"].astype(int),
            "timestamp": merged["timestamp"].astype(int),
            "query_count": merged["query_count_filtered"].astype(float),
            "value": merged["coverage_filtered"] - merged["coverage_uniform"],
        }
    )


def _multi_answer_gap_frames(
    rows: pd.DataFrame,
    deletion_rate: float,
    methods: Sequence[str],
) -> dict[str, pd.DataFrame]:
    required = {
        "seed",
        "deletion_rate",
        "method",
        "prediction_side",
        "timestamp",
        "subject_id",
        "relation_id",
        "true_object_id",
        "answer_count",
        "is_multi_answer",
        "covered",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"per-query metrics are missing columns: {missing}")
    selected = rows[
        np.isclose(rows["deletion_rate"], deletion_rate)
        & rows["method"].isin(methods)
    ].copy()
    query_key = [
        "seed",
        "method",
        "prediction_side",
        "timestamp",
        "subject_id",
        "relation_id",
    ]
    grouped = selected.groupby(query_key, as_index=False, sort=False).agg(
        distinct_answer_count=("true_object_id", "nunique"),
        answer_count=("answer_count", "first"),
        answer_count_nunique=("answer_count", "nunique"),
        is_multi_answer=("is_multi_answer", "first"),
        full_set_covered=("covered", "all"),
    )
    invalid = grouped[
        (grouped["answer_count_nunique"] != 1)
        | (grouped["distinct_answer_count"] != grouped["answer_count"])
        | (
            grouped["is_multi_answer"].astype(bool)
            != (grouped["answer_count"] > 1)
        )
    ]
    if len(invalid):
        raise ValueError(f"inconsistent answer multiplicity in {len(invalid)} queries")
    grouped["answer_group"] = np.where(
        grouped["answer_count"] > 1, "multi", "single"
    )
    timestamp = grouped.groupby(
        ["seed", "method", "timestamp", "answer_group"], as_index=False
    ).agg(full_set_coverage=("full_set_covered", "mean"))
    outputs: dict[str, pd.DataFrame] = {}
    for method in methods:
        method_rows = timestamp[timestamp["method"].eq(method)]
        pivot = method_rows.pivot(
            index=["seed", "timestamp"],
            columns="answer_group",
            values="full_set_coverage",
        ).reset_index()
        if not {"single", "multi"}.issubset(pivot.columns):
            raise ValueError(f"{method} has no estimable single/multi-answer contrast")
        complete = pivot.dropna(subset=["single", "multi"])
        outputs[method] = pd.DataFrame(
            {
                "seed": complete["seed"].astype(int),
                "timestamp": complete["timestamp"].astype(int),
                "value": complete["multi"] - complete["single"],
            }
        )
    return outputs


def _metadata(rows: pd.DataFrame) -> dict[str, str]:
    return {
        "dataset_mode": _single_value(rows, "dataset_mode"),
        "model_name": _single_value(rows, "model_name"),
        "negative_sampling": _single_value(rows, "negative_sampling"),
    }


def export_expansion_bootstrap(
    matrix_root: Path,
    output_dir: Path,
    *,
    expected_runs: Iterable[str] = EXPECTED_RUNS,
    filtered_runs: Iterable[str] = FILTERED_RUNS,
    sampling_pairs: Iterable[tuple[str, str]] = SAMPLING_PAIRS,
    method_baselines: Sequence[str] = METHOD_BASELINES,
    query_methods: Sequence[str] = QUERY_METHODS,
    target_coverage: float = 0.90,
    deletion_rate: float = PRIMARY_DELETION_RATE,
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    iterations: int = DEFAULT_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    matrix_root = matrix_root.resolve()
    output_dir = output_dir.resolve()
    expected_runs = tuple(expected_runs)
    filtered_runs = tuple(filtered_runs)
    sampling_pairs = tuple(sampling_pairs)
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target coverage must be between zero and one")
    if not 0.0 < deletion_rate < 1.0:
        raise ValueError("deletion rate must be between zero and one")
    if iterations <= 0 or not block_lengths or any(value <= 0 for value in block_lengths):
        raise ValueError("iterations and block lengths must be positive")
    commit, roots, inputs = _load_complete_runs(matrix_root, expected_runs)
    windows = {
        key: pd.read_csv(root / "metrics" / "per_window.csv")
        for key, root in roots.items()
    }
    metadata = {key: _metadata(frame) for key, frame in windows.items()}
    statistics: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    sequence = 0

    def evaluate(
        *,
        statistic: str,
        family: str,
        run: str,
        comparison: str,
        frame: pd.DataFrame,
        weight_column: str | None,
        details: dict[str, Any],
    ) -> None:
        nonlocal sequence
        for block_length in block_lengths:
            result = bootstrap_statistic(
                frame,
                "value",
                weight_column=weight_column,
                block_length=int(block_length),
                iterations=iterations,
                bootstrap_seed=bootstrap_seed + sequence,
            )
            sequence += 1
            record = {
                "statistic": statistic,
                "family": family,
                "run": run,
                "comparison": comparison,
                **details,
                "observed": result["observed"],
                "ci95_low": result["ci95"][0],
                "ci95_high": result["ci95"][1],
                "pvalue_positive": result["pvalue_positive"],
                "pvalue_negative": result["pvalue_negative"],
                "iterations": iterations,
                "block_length": int(block_length),
                "seed_count": result["seed_count"],
                "timestamp_count": result["timestamp_count"],
                "excluded_timestamp_count": result["excluded_timestamp_count"],
                "weighting": "query_micro" if weight_column else "timestamp_macro",
            }
            statistics.append(record)
            for seed, value in result["observed_by_seed"].items():
                seed_records.append(
                    {
                        "statistic": statistic,
                        "family": family,
                        "run": run,
                        "comparison": comparison,
                        **details,
                        "block_length": int(block_length),
                        "seed": int(seed),
                        "observed": value,
                        "weighting": record["weighting"],
                    }
                )

    for key in filtered_runs:
        rows = windows[key]
        run_details = {
            **metadata[key],
            "deletion_rate": deletion_rate,
            "target_coverage": target_coverage,
        }
        for baseline in method_baselines:
            gain, reliability = _method_contrast_frames(
                rows, deletion_rate, baseline, target_coverage
            )
            evaluate(
                statistic="coverage_gain",
                family="rolling_vs_static_baseline",
                run=key,
                comparison=f"rolling_margin-minus-{baseline}",
                frame=gain,
                weight_column="query_count",
                details=run_details,
            )
            evaluate(
                statistic="undercoverage_reduction",
                family="rolling_vs_static_baseline",
                run=key,
                comparison=f"rolling_margin-minus-{baseline}",
                frame=reliability,
                weight_column=None,
                details=run_details,
            )
        for method in ("static_margin", "rolling_margin"):
            effect = _deletion_effect_frame(rows, method, deletion_rate)
            evaluate(
                statistic="coverage_change",
                family="deletion_effect",
                run=key,
                comparison=f"delete{deletion_rate:g}-minus-delete0__{method}",
                frame=effect,
                weight_column="query_count",
                details=run_details,
            )

        query_columns = [
            "seed",
            "deletion_rate",
            "method",
            "prediction_side",
            "timestamp",
            "subject_id",
            "relation_id",
            "true_object_id",
            "answer_count",
            "is_multi_answer",
            "covered",
        ]
        query_rows = pd.read_parquet(
            roots[key] / "metrics" / "per_query.parquet", columns=query_columns
        )
        for method, gap in _multi_answer_gap_frames(
            query_rows, deletion_rate, query_methods
        ).items():
            evaluate(
                statistic="full_set_coverage_gap",
                family="multi_answer_degradation",
                run=key,
                comparison=f"multi-minus-single__{method}",
                frame=gap,
                weight_column=None,
                details=run_details,
            )
        del query_rows

    for filtered_key, uniform_key in sampling_pairs:
        filtered = windows[filtered_key]
        uniform = windows[uniform_key]
        filtered_meta = metadata[filtered_key]
        uniform_meta = metadata[uniform_key]
        for field in ("dataset_mode", "model_name"):
            if filtered_meta[field] != uniform_meta[field]:
                raise ValueError(f"sampling pair differs in {field}")
        for rate in (0.0, deletion_rate):
            for method in ("static_margin", "rolling_margin"):
                effect = _sampling_effect_frame(filtered, uniform, method, rate)
                evaluate(
                    statistic="coverage_change",
                    family="negative_sampling_sensitivity",
                    run=f"{filtered_key}|{uniform_key}",
                    comparison=f"filtered-minus-uniform__{method}",
                    frame=effect,
                    weight_column="query_count",
                    details={
                        "dataset_mode": filtered_meta["dataset_mode"],
                        "model_name": filtered_meta["model_name"],
                        "negative_sampling": "filtered-minus-uniform",
                        "deletion_rate": rate,
                        "target_coverage": target_coverage,
                    },
                )

    summary = pd.DataFrame(statistics).sort_values(
        ["family", "run", "comparison", "statistic", "block_length"],
        kind="stable",
    )
    by_seed = pd.DataFrame(seed_records).sort_values(
        ["family", "run", "comparison", "statistic", "block_length", "seed"],
        kind="stable",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "expansion_bootstrap_summary.csv"
    seed_path = output_dir / "expansion_bootstrap_by_seed.csv"
    _write_csv(summary, summary_path)
    _write_csv(by_seed, seed_path)
    manifest = {
        "block_lengths": [int(value) for value in block_lengths],
        "bootstrap_seed": bootstrap_seed,
        "created_at": datetime.now(UTC).isoformat(),
        "deletion_rate": deletion_rate,
        "expected_runs": list(expected_runs),
        "git_commit": commit,
        "inputs": inputs,
        "iterations": iterations,
        "method_baselines": list(method_baselines),
        "query_methods": list(query_methods),
        "resampling_scheme": (
            "equal-seed circular moving-block bootstrap; each replicate resamples "
            "seeds and applies one shared timestamp-block draw to sampled seeds"
        ),
        "statistics_count": int(len(summary)),
        "target_coverage": target_coverage,
    }
    manifest_path = output_dir / "expansion_bootstrap_manifest.json"
    _write_json(manifest, manifest_path)
    files = [summary_path, seed_path, manifest_path]
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in sorted(files)),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/expansion_bootstrap")
    )
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--deletion-rate", type=float, default=PRIMARY_DELETION_RATE)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--block-lengths", type=int, nargs="+", default=list(DEFAULT_BLOCK_LENGTHS)
    )
    args = parser.parse_args()
    manifest = export_expansion_bootstrap(
        args.matrix_root,
        args.output_dir,
        target_coverage=args.target_coverage,
        deletion_rate=args.deletion_rate,
        block_lengths=args.block_lengths,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
