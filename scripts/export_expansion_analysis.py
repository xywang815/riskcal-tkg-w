"""Audit and summarize the frozen E1-E6 expansion matrix.

The exporter keeps analysis separate from training and writes only result-facing
CSV/JSON artifacts. It refuses incomplete matrices, missing provenance, and
inconsistent query multiplicities.
"""

from argparse import ArgumentParser
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

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
QUERY_KEY = (
    "seed",
    "deletion_rate",
    "method",
    "prediction_side",
    "timestamp",
    "subject_id",
    "relation_id",
)
CONDITION_METRICS = (
    "coverage",
    "mean_size",
    "mrr",
    "frequency_mrr",
)
KGCP_BASELINES = (
    "static_margin",
    "kgcp_negscore_static",
    "kgcp_minmax_static",
    "kgcp_softmax_static",
)


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


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_array = values.to_numpy(dtype=float)
    weight_array = weights.to_numpy(dtype=float)
    if len(value_array) == 0 or np.any(weight_array <= 0):
        raise ValueError("weighted summaries require positive nonempty weights")
    return float(np.average(value_array, weights=weight_array))


def _single_value(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"expected one {column}, found {values.tolist()}")
    return str(values[0])


def summarize_conditions(rows: pd.DataFrame, run_name: str) -> pd.DataFrame:
    required = {
        "seed",
        "deletion_rate",
        "method",
        "timestamp",
        "query_count",
        "dataset_mode",
        "model_name",
        "negative_sampling",
        *CONDITION_METRICS,
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"per-window metrics are missing columns: {missing}")
    if rows.empty or (rows["query_count"] <= 0).any():
        raise ValueError("per-window metrics require positive query counts")
    metadata = {
        "run": run_name,
        "dataset_mode": _single_value(rows, "dataset_mode"),
        "model_name": _single_value(rows, "model_name"),
        "negative_sampling": _single_value(rows, "negative_sampling"),
    }
    records: list[dict[str, Any]] = []
    for (seed, deletion_rate, method), frame in rows.groupby(
        ["seed", "deletion_rate", "method"], sort=True
    ):
        record: dict[str, Any] = {
            **metadata,
            "seed": int(seed),
            "deletion_rate": float(deletion_rate),
            "method": str(method),
            "timestamp_count": int(frame["timestamp"].nunique()),
            "query_count": int(frame["query_count"].sum()),
            "timestamp_macro_coverage": float(frame["coverage"].mean()),
        }
        for metric in CONDITION_METRICS:
            record[metric] = _weighted_mean(frame[metric], frame["query_count"])
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["run", "seed", "deletion_rate", "method"], kind="stable"
    ).reset_index(drop=True)


def aggregate_conditions(rows: pd.DataFrame, target_coverage: float) -> pd.DataFrame:
    keys = [
        "run",
        "dataset_mode",
        "model_name",
        "negative_sampling",
        "deletion_rate",
        "method",
    ]
    metrics = ["coverage", "timestamp_macro_coverage", "mean_size", "mrr"]
    records: list[dict[str, Any]] = []
    for values, frame in rows.groupby(keys, sort=True):
        record = dict(zip(keys, values, strict=True))
        record["seed_count"] = int(frame["seed"].nunique())
        for metric in metrics:
            record[f"{metric}_mean"] = float(frame[metric].mean())
            record[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        record["coverage_abs_error_mean"] = float(
            np.abs(frame["coverage"] - target_coverage).mean()
        )
        record["undercoverage_mean"] = float(
            np.maximum(target_coverage - frame["coverage"], 0.0).mean()
        )
        records.append(record)
    return pd.DataFrame(records)


def build_deletion_effects(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "run",
        "dataset_mode",
        "model_name",
        "negative_sampling",
        "seed",
        "method",
    ]
    records: list[dict[str, Any]] = []
    for values, frame in rows.groupby(keys, sort=True):
        baseline = frame[np.isclose(frame["deletion_rate"], 0.0)]
        if len(baseline) != 1:
            raise ValueError(f"deletion baseline is not unique for {values}")
        base = baseline.iloc[0]
        for _, current in frame[~np.isclose(frame["deletion_rate"], 0.0)].iterrows():
            record = dict(zip(keys, values, strict=True))
            record["deletion_rate"] = float(current["deletion_rate"])
            for metric in ("coverage", "mean_size", "mrr"):
                record[f"{metric}_change_vs_delete0"] = float(
                    current[metric] - base[metric]
                )
            records.append(record)
    return pd.DataFrame(records)


def build_method_contrasts(
    rows: pd.DataFrame, target_coverage: float
) -> pd.DataFrame:
    condition_keys = [
        "run",
        "dataset_mode",
        "model_name",
        "negative_sampling",
        "seed",
        "deletion_rate",
    ]
    records: list[dict[str, Any]] = []
    for values, frame in rows.groupby(condition_keys, sort=True):
        by_method = frame.set_index("method")
        if "rolling_margin" not in by_method.index:
            continue
        rolling = by_method.loc["rolling_margin"]
        for baseline_name in KGCP_BASELINES:
            if baseline_name not in by_method.index:
                continue
            baseline = by_method.loc[baseline_name]
            record = dict(zip(condition_keys, values, strict=True))
            record["baseline"] = baseline_name
            record["coverage_gain"] = float(
                rolling["coverage"] - baseline["coverage"]
            )
            record["undercoverage_reduction"] = float(
                max(target_coverage - baseline["coverage"], 0.0)
                - max(target_coverage - rolling["coverage"], 0.0)
            )
            record["absolute_error_reduction"] = float(
                abs(baseline["coverage"] - target_coverage)
                - abs(rolling["coverage"] - target_coverage)
            )
            record["mean_size_change"] = float(
                rolling["mean_size"] - baseline["mean_size"]
            )
            record["mean_size_ratio"] = float(
                rolling["mean_size"] / baseline["mean_size"]
            )
            records.append(record)
    return pd.DataFrame(records)


def build_sampling_contrasts(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_mode", "model_name", "seed", "deletion_rate", "method"]
    records: list[dict[str, Any]] = []
    for values, frame in rows.groupby(keys, sort=True):
        by_sampling = frame.set_index("negative_sampling")
        if not {"filtered", "uniform"}.issubset(by_sampling.index):
            continue
        filtered = by_sampling.loc["filtered"]
        uniform = by_sampling.loc["uniform"]
        record = dict(zip(keys, values, strict=True))
        record["filtered_run"] = str(filtered["run"])
        record["uniform_run"] = str(uniform["run"])
        for metric in ("coverage", "mean_size", "mrr"):
            record[f"{metric}_filtered_minus_uniform"] = float(
                filtered[metric] - uniform[metric]
            )
        records.append(record)
    return pd.DataFrame(records)


def summarize_multi_answer(rows: pd.DataFrame, run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        *QUERY_KEY,
        "true_object_id",
        "answer_count",
        "is_multi_answer",
        "set_size",
        "covered",
        "dataset_mode",
        "model_name",
        "negative_sampling",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"per-query metrics are missing columns: {missing}")
    metadata = {
        "run": run_name,
        "dataset_mode": _single_value(rows, "dataset_mode"),
        "model_name": _single_value(rows, "model_name"),
        "negative_sampling": _single_value(rows, "negative_sampling"),
    }
    normalized = rows.copy()
    normalized["covered"] = normalized["covered"].astype(bool)
    grouped = normalized.groupby(list(QUERY_KEY), as_index=False, sort=False).agg(
        distinct_answer_count=("true_object_id", "nunique"),
        recorded_answer_count=("answer_count", "first"),
        answer_count_nunique=("answer_count", "nunique"),
        recorded_multi_answer=("is_multi_answer", "first"),
        covered_answer_count=("covered", "sum"),
        full_set_covered=("covered", "all"),
        partial_answer_recall=("covered", "mean"),
        set_size=("set_size", "first"),
        set_size_nunique=("set_size", "nunique"),
    )
    invalid = grouped[
        (grouped["answer_count_nunique"] != 1)
        | (grouped["set_size_nunique"] != 1)
        | (grouped["recorded_answer_count"] != grouped["distinct_answer_count"])
        | (
            grouped["recorded_multi_answer"].astype(bool)
            != (grouped["recorded_answer_count"] > 1)
        )
    ]
    if len(invalid):
        raise ValueError(
            "per-query answer multiplicity is inconsistent for "
            f"{len(invalid)} query groups"
        )
    grouped["answer_group"] = np.where(
        grouped["recorded_answer_count"] > 1, "multi", "single"
    )
    records: list[dict[str, Any]] = []
    for (seed, deletion_rate, method, answer_group), frame in grouped.groupby(
        ["seed", "deletion_rate", "method", "answer_group"], sort=True
    ):
        records.append(
            {
                **metadata,
                "seed": int(seed),
                "deletion_rate": float(deletion_rate),
                "method": str(method),
                "answer_group": str(answer_group),
                "query_count": int(len(frame)),
                "mean_answer_count": float(frame["recorded_answer_count"].mean()),
                "full_set_coverage": float(frame["full_set_covered"].mean()),
                "partial_answer_recall": float(frame["partial_answer_recall"].mean()),
                "mean_size": float(frame["set_size"].mean()),
            }
        )
    by_group = pd.DataFrame(records)
    degradation_records: list[dict[str, Any]] = []
    pair_keys = [
        "run",
        "dataset_mode",
        "model_name",
        "negative_sampling",
        "seed",
        "deletion_rate",
        "method",
    ]
    for values, frame in by_group.groupby(pair_keys, sort=True):
        indexed = frame.set_index("answer_group")
        if not {"single", "multi"}.issubset(indexed.index):
            continue
        single = indexed.loc["single"]
        multi = indexed.loc["multi"]
        record = dict(zip(pair_keys, values, strict=True))
        record["multi_query_count"] = int(multi["query_count"])
        record["full_set_coverage_multi_minus_single"] = float(
            multi["full_set_coverage"] - single["full_set_coverage"]
        )
        record["partial_recall_multi_minus_single"] = float(
            multi["partial_answer_recall"] - single["partial_answer_recall"]
        )
        record["mean_size_multi_minus_single"] = float(
            multi["mean_size"] - single["mean_size"]
        )
        degradation_records.append(record)
    return by_group, pd.DataFrame(degradation_records)


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


def _aggregate_seed_rows(rows: pd.DataFrame, value_columns: Iterable[str]) -> pd.DataFrame:
    group_columns = [
        column
        for column in rows.columns
        if column not in {"seed", *value_columns}
    ]
    records: list[dict[str, Any]] = []
    for values, frame in rows.groupby(group_columns, dropna=False, sort=True):
        record = dict(zip(group_columns, values, strict=True))
        record["seed_count"] = int(frame["seed"].nunique())
        for column in value_columns:
            record[f"{column}_mean"] = float(frame[column].mean())
            record[f"{column}_sd"] = float(frame[column].std(ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def export_expansion_analysis(
    matrix_root: Path,
    output_dir: Path,
    *,
    expected_runs: Iterable[str] = EXPECTED_RUNS,
    target_coverage: float = 0.90,
) -> dict[str, Any]:
    matrix_root = matrix_root.resolve()
    output_dir = output_dir.resolve()
    progress_path = matrix_root / "matrix_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    commit = progress.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("matrix progress must record a full Git commit")
    run_records = progress.get("runs")
    if not isinstance(run_records, dict):
        raise ValueError("matrix progress runs field must be a mapping")

    condition_frames: list[pd.DataFrame] = []
    multi_frames: list[pd.DataFrame] = []
    degradation_frames: list[pd.DataFrame] = []
    inputs: dict[str, dict[str, str]] = {}
    for key in expected_runs:
        record = run_records.get(key)
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise ValueError(f"required expansion run is incomplete: {key}")
        run_root = _resolve_run_root(matrix_root, key, record)
        manifest_path = run_root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"run manifest is incomplete: {run_root}")
        window_path = run_root / "metrics" / "per_window.csv"
        query_path = run_root / "metrics" / "per_query.parquet"
        inputs[key] = {
            "run_root": str(run_root),
            "run_manifest_sha256": _sha256(manifest_path),
            "per_window_sha256": _sha256(window_path),
            "per_query_sha256": _sha256(query_path),
        }
        condition_frames.append(
            summarize_conditions(pd.read_csv(window_path), key)
        )
        query_columns = [
            *QUERY_KEY,
            "true_object_id",
            "answer_count",
            "is_multi_answer",
            "set_size",
            "covered",
            "dataset_mode",
            "model_name",
            "negative_sampling",
        ]
        by_group, degradation = summarize_multi_answer(
            pd.read_parquet(query_path, columns=query_columns), key
        )
        multi_frames.append(by_group)
        degradation_frames.append(degradation)

    conditions = pd.concat(condition_frames, ignore_index=True)
    condition_aggregate = aggregate_conditions(conditions, target_coverage)
    deletion_effects = build_deletion_effects(conditions)
    method_contrasts = build_method_contrasts(conditions, target_coverage)
    sampling_contrasts = build_sampling_contrasts(conditions)
    multi_answer = pd.concat(multi_frames, ignore_index=True)
    multi_degradation = pd.concat(degradation_frames, ignore_index=True)
    multi_aggregate = _aggregate_seed_rows(
        multi_answer,
        (
            "query_count",
            "mean_answer_count",
            "full_set_coverage",
            "partial_answer_recall",
            "mean_size",
        ),
    )
    degradation_aggregate = _aggregate_seed_rows(
        multi_degradation,
        (
            "multi_query_count",
            "full_set_coverage_multi_minus_single",
            "partial_recall_multi_minus_single",
            "mean_size_multi_minus_single",
        ),
    )

    outputs = {
        "condition_by_seed.csv": conditions,
        "condition_aggregate.csv": condition_aggregate,
        "deletion_effects_by_seed.csv": deletion_effects,
        "method_contrasts_by_seed.csv": method_contrasts,
        "sampling_contrasts_by_seed.csv": sampling_contrasts,
        "multi_answer_by_seed.csv": multi_answer,
        "multi_answer_aggregate.csv": multi_aggregate,
        "multi_answer_degradation_by_seed.csv": multi_degradation,
        "multi_answer_degradation_aggregate.csv": degradation_aggregate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for name, frame in outputs.items():
        path = output_dir / name
        _write_csv(frame, path)
        output_hashes[name] = _sha256(path)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "expected_runs": list(expected_runs),
        "git_commit": commit,
        "inputs": inputs,
        "outputs": output_hashes,
        "target_coverage": target_coverage,
    }
    manifest_path = output_dir / "analysis_manifest.json"
    _write_json(manifest, manifest_path)
    checksum_paths = [*(output_dir / name for name in outputs), manifest_path]
    checksum_text = "".join(
        f"{_sha256(path)}  {path.name}\n" for path in sorted(checksum_paths)
    )
    (output_dir / "SHA256SUMS.txt").write_text(
        checksum_text, encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/expansion_analysis")
    )
    parser.add_argument("--target-coverage", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.0 < args.target_coverage < 1.0 or not math.isfinite(
        args.target_coverage
    ):
        raise SystemExit("target coverage must be finite and between 0 and 1")
    manifest = export_expansion_analysis(
        args.matrix_root,
        args.output_dir,
        target_coverage=args.target_coverage,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
