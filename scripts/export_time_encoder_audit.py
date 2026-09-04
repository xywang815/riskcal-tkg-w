"""Validate and summarize the prespecified time-encoder audit matrix."""

from argparse import ArgumentParser
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASETS = ("icews14", "icews05_15")
SCORERS = ("distmult", "tcomplex")
TIME_ENCODINGS = ("none", "linear", "bounded_fourier", "polynomial_fourier")
EXPECTED_RUNS = tuple(
    f"{dataset}_{scorer}_{encoding}"
    for dataset in DATASETS
    for scorer in SCORERS
    for encoding in TIME_ENCODINGS
)
DEFAULT_BLOCK_LENGTHS = (3, 7, 14, 21)
DEFAULT_ITERATIONS = 20_000
DEFAULT_BOOTSTRAP_SEED = 20260904


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
    matrix_root: Path,
    expected_runs: Iterable[str],
) -> tuple[str, pd.DataFrame, dict[str, dict[str, str]]]:
    progress_path = matrix_root / "matrix_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    commit = progress.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("matrix progress must record a full Git commit")
    records = progress.get("runs")
    if not isinstance(records, dict):
        raise ValueError("matrix progress runs field must be a mapping")
    frames: list[pd.DataFrame] = []
    inputs: dict[str, dict[str, str]] = {}
    for key in expected_runs:
        record = records.get(key)
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise ValueError(f"required time-encoder run is incomplete: {key}")
        root = _resolve_run_root(matrix_root, key, record)
        manifest_path = root / "run_manifest.json"
        window_path = root / "metrics" / "per_window.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"run manifest is incomplete: {root}")
        frame = pd.read_csv(window_path)
        frame.insert(0, "case", key)
        frames.append(frame)
        inputs[key] = {
            "run_root": str(root),
            "run_manifest_sha256": _sha256(manifest_path),
            "per_window_sha256": _sha256(window_path),
        }
    return commit, pd.concat(frames, ignore_index=True), inputs


def build_timestamp_contrasts(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "case",
        "dataset_mode",
        "model_name",
        "time_encoding",
        "negative_sampling",
        "seed",
        "deletion_rate",
        "timestamp",
        "query_count",
        "method",
        "coverage",
        "mean_size",
        "threshold",
        "score_global_min",
        "score_global_max",
        "score_global_range",
        "score_mean",
        "score_std",
        "mean_query_score_range",
        "true_score_mean",
        "true_margin_mean",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"per-window rows are missing columns: {missing}")
    rows = rows[np.isclose(rows["deletion_rate"], 0.0)].copy()
    identity = [
        "case",
        "dataset_mode",
        "model_name",
        "time_encoding",
        "negative_sampling",
        "seed",
        "timestamp",
        "query_count",
    ]
    methods = rows[rows["method"].isin(("static_margin", "rolling_margin"))]
    if methods.duplicated(identity + ["method"]).any():
        raise ValueError("duplicate method row for a seed/timestamp condition")
    wide = methods.pivot(
        index=identity,
        columns="method",
        values=("coverage", "mean_size", "threshold"),
    )
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    wide = wide.reset_index()
    expected_columns = {
        "static_margin_coverage",
        "rolling_margin_coverage",
        "static_margin_mean_size",
        "rolling_margin_mean_size",
        "static_margin_threshold",
        "rolling_margin_threshold",
    }
    if not expected_columns <= set(wide.columns):
        raise ValueError("static and rolling rows are not both present")
    diagnostic_columns = [
        "score_global_min",
        "score_global_max",
        "score_global_range",
        "score_mean",
        "score_std",
        "mean_query_score_range",
        "true_score_mean",
        "true_margin_mean",
    ]
    diagnostics = rows.groupby(identity, as_index=False)[diagnostic_columns].first()
    result = wide.merge(diagnostics, on=identity, validate="one_to_one")
    result["rolling_minus_static_coverage"] = (
        result["rolling_margin_coverage"] - result["static_margin_coverage"]
    )
    result["rolling_minus_static_mean_size"] = (
        result["rolling_margin_mean_size"] - result["static_margin_mean_size"]
    )
    return result.sort_values(identity).reset_index(drop=True)


def _weighted_mean(group: pd.DataFrame, column: str) -> float:
    return float(np.average(group[column], weights=group["query_count"]))


def aggregate_by_seed(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["case", "dataset_mode", "model_name", "time_encoding", "seed"]
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(keys, sort=True):
        ordered = group.sort_values("timestamp")
        quarter = max(1, len(ordered) // 4)
        early = ordered.iloc[:quarter]
        late = ordered.iloc[-quarter:]
        record = dict(zip(keys, key, strict=True))
        for column in (
            "static_margin_coverage",
            "rolling_margin_coverage",
            "rolling_minus_static_coverage",
            "static_margin_mean_size",
            "rolling_margin_mean_size",
            "rolling_minus_static_mean_size",
            "score_global_range",
            "score_std",
            "mean_query_score_range",
            "true_score_mean",
            "true_margin_mean",
        ):
            record[column] = _weighted_mean(ordered, column)
            record[f"early_{column}"] = _weighted_mean(early, column)
            record[f"late_{column}"] = _weighted_mean(late, column)
        for column in (
            "static_margin_threshold",
            "rolling_margin_threshold",
            "score_global_range",
            "score_std",
            "true_score_mean",
            "true_margin_mean",
        ):
            record[f"spearman_timestamp_{column}"] = float(
                ordered["timestamp"].corr(ordered[column], method="spearman")
            )
        record["timestamp_count"] = int(len(ordered))
        record["query_count"] = int(ordered["query_count"].sum())
        records.append(record)
    return pd.DataFrame(records)


def _prepare_bootstrap_matrix(
    frame: pd.DataFrame,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if frame.duplicated(["seed", "timestamp"]).any():
        raise ValueError("bootstrap frame has duplicate seed/timestamp rows")
    seeds = sorted(int(value) for value in frame["seed"].unique())
    timestamp_sets = [
        set(group["timestamp"].astype(int))
        for _, group in frame.groupby("seed", sort=True)
    ]
    common = sorted(set.intersection(*timestamp_sets))
    if not common:
        raise ValueError("bootstrap frame has no timestamp shared by every seed")
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for seed in seeds:
        ordered = frame[frame["seed"] == seed].set_index("timestamp").reindex(common)
        values.append(ordered[value_column].to_numpy(dtype=float))
        weights.append(ordered["query_count"].to_numpy(dtype=float))
    return np.stack(values), np.stack(weights), len(seeds), len(common)


def bootstrap_statistic(
    frame: pd.DataFrame,
    value_column: str,
    *,
    block_length: int,
    iterations: int,
    bootstrap_seed: int,
    chunk_size: int = 128,
) -> dict[str, Any]:
    values, weights, seed_count, timestamp_count = _prepare_bootstrap_matrix(
        frame,
        value_column,
    )
    observed_by_seed = (values * weights).sum(axis=1) / weights.sum(axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    samples = np.empty(iterations, dtype=float)
    block_count = math.ceil(timestamp_count / block_length)
    offsets = np.arange(block_length, dtype=np.int64)
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        count = stop - start
        sampled_seeds = rng.integers(0, seed_count, size=(count, seed_count))
        starts = rng.integers(0, timestamp_count, size=(count, block_count))
        positions = ((starts[:, :, None] + offsets) % timestamp_count).reshape(
            count,
            -1,
        )[:, :timestamp_count]
        selected_values = values[sampled_seeds[:, :, None], positions[:, None, :]]
        selected_weights = weights[sampled_seeds[:, :, None], positions[:, None, :]]
        per_seed = (selected_values * selected_weights).sum(axis=2) / (
            selected_weights.sum(axis=2)
        )
        samples[start:stop] = per_seed.mean(axis=1)
    return {
        "observed": float(observed_by_seed.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "pvalue_positive": float((np.sum(samples <= 0.0) + 1) / (iterations + 1)),
        "seed_count": seed_count,
        "timestamp_count": timestamp_count,
    }


def export_time_encoder_audit(
    matrix_root: Path,
    output_root: Path,
    *,
    expected_runs: Iterable[str] = EXPECTED_RUNS,
    block_lengths: tuple[int, ...] = DEFAULT_BLOCK_LENGTHS,
    iterations: int = DEFAULT_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    matrix_root = matrix_root.resolve()
    output_root = output_root.resolve()
    expected_runs = tuple(expected_runs)
    commit, raw_rows, inputs = _load_complete_runs(matrix_root, expected_runs)
    by_timestamp = build_timestamp_contrasts(raw_rows)
    by_seed = aggregate_by_seed(by_timestamp)
    bootstrap_rows: list[dict[str, Any]] = []
    for case_index, (case, group) in enumerate(by_timestamp.groupby("case", sort=True)):
        metadata = group.iloc[0]
        for block_length in block_lengths:
            result = bootstrap_statistic(
                group,
                "rolling_minus_static_coverage",
                block_length=block_length,
                iterations=iterations,
                bootstrap_seed=bootstrap_seed + 1000 * case_index + block_length,
            )
            bootstrap_rows.append(
                {
                    "case": case,
                    "dataset_mode": metadata["dataset_mode"],
                    "model_name": metadata["model_name"],
                    "time_encoding": metadata["time_encoding"],
                    "statistic": "rolling_minus_static_coverage",
                    "block_length": block_length,
                    **result,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    primary_bootstrap = bootstrap[bootstrap["block_length"] == 7].drop(
        columns=["dataset_mode", "model_name", "time_encoding", "statistic"]
    )
    summary = (
        by_seed.groupby(
            ["case", "dataset_mode", "model_name", "time_encoding"],
            as_index=False,
        )
        .mean(numeric_only=True)
        .drop(columns=["seed"])
        .merge(primary_bootstrap, on="case", suffixes=("", "_bootstrap"))
    )
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "time_encoder_audit_by_timestamp.csv": by_timestamp,
        "time_encoder_audit_by_seed.csv": by_seed,
        "time_encoder_audit_bootstrap.csv": bootstrap,
        "time_encoder_audit_summary.csv": summary,
    }
    for name, frame in outputs.items():
        _write_csv(frame, output_root / name)
    output_hashes = {name: _sha256(output_root / name) for name in outputs}
    manifest = {
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "expected_runs": list(expected_runs),
        "primary_deletion_rate": 0.0,
        "primary_block_length": 7,
        "block_lengths": list(block_lengths),
        "iterations": iterations,
        "bootstrap_seed": bootstrap_seed,
        "inputs": inputs,
        "outputs": output_hashes,
    }
    manifest_path = output_root / "time_encoder_audit_manifest.json"
    _write_json(manifest, manifest_path)
    checksums = {
        **output_hashes,
        manifest_path.name: _sha256(manifest_path),
    }
    (output_root / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--block-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_BLOCK_LENGTHS),
    )
    args = parser.parse_args()
    manifest = export_time_encoder_audit(
        args.matrix_root,
        args.output_root,
        block_lengths=tuple(args.block_lengths),
        iterations=args.iterations,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
