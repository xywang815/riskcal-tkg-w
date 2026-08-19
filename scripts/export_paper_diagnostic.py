"""Export the immutable local CPU run into paper-facing tables and figures.

The run itself is produced by ``scripts/run_pilot.py``.  This module is the
versioned transformation from that run directory to ``paper/data`` and
``paper/figures``.  It makes internal timestamp identifiers and original
dataset timestamp labels explicit so that paper plots cannot silently mix the
two coordinate systems.
"""

from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


METHOD_ORDER = ["rolling", "static", "top1", "weighted", "adaptive"]
PLOT_METHODS = ["static", "rolling", "weighted", "adaptive"]
METHOD_COLORS = {
    "static": "#9c755f",
    "rolling": "#4e79a7",
    "weighted": "#e15759",
    "adaptive": "#b07aa1",
}


def sha256_file(path: Path) -> str:
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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset_manifest(source: dict[str, Any]) -> dict[str, Any]:
    """Disambiguate internal timestamp IDs from original timestamp labels."""
    exported = dict(source)
    for split in ("train", "calibration", "test"):
        old_key = f"{split}_max_timestamp"
        if old_key not in exported:
            raise ValueError(f"dataset manifest is missing {old_key}")
        exported[f"{split}_max_timestamp_id"] = exported.pop(old_key)
        timestamp_range = exported.get(f"{split}_timestamp_range")
        if not timestamp_range or len(timestamp_range) != 2:
            raise ValueError(f"dataset manifest has no valid {split} timestamp range")
        exported[f"{split}_max_timestamp_raw"] = str(timestamp_range[1])
    exported["timestamp_coordinate_note"] = (
        "Fields ending in _timestamp_id use zero-based internal IDs; "
        "timestamp_range and fields ending in _timestamp_raw use original labels."
    )
    return exported


def _timestamp_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def build_timestamp_mapping(data_root: Path) -> dict[int, str]:
    """Build the same sorted raw-label to internal-ID mapping as the loader."""
    labels: set[str] = set()
    for name in ("train.txt", "valid.txt", "test.txt"):
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing timestamp source: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                columns = line.rstrip("\n").split("\t")
                if len(columns) != 4:
                    columns = line.split()
                if len(columns) != 4:
                    raise ValueError(f"{path}:{line_number} does not have four columns")
                labels.add(columns[3])
    ordered = sorted(labels, key=_timestamp_key)
    return {index: label for index, label in enumerate(ordered)}


def verify_source_files(data_root: Path, dataset_manifest: dict[str, Any]) -> None:
    """Reject timestamp mappings built from files other than the immutable run inputs."""
    records = dataset_manifest.get("source_files")
    if not isinstance(records, dict):
        raise ValueError("dataset manifest has no source_files records")
    for name in ("train.txt", "valid.txt", "test.txt"):
        path = data_root / name
        record = records.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise ValueError(f"{name} is missing from data root or manifest")
        expected_bytes = int(record.get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"{name} byte count does not match immutable dataset manifest"
            )
        expected_sha = str(record.get("sha256", ""))
        if sha256_file(path) != expected_sha:
            raise ValueError(f"{name} SHA-256 does not match immutable dataset manifest")


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    return float(np.average(frame[column], weights=frame["query_count"]))


def condition_summary(
    rows: pd.DataFrame, target: float, num_entities: int
) -> pd.DataFrame:
    """Produce one diagnostic record per seed, deletion rate, and method."""
    required = {
        "seed",
        "deletion_rate",
        "actual_deletion_rate",
        "method",
        "timestamp",
        "calibration_max_timestamp",
        "selected_half_life",
        "query_count",
        "coverage",
        "mean_size",
        "mrr",
        "frequency_mrr",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"per-window metrics are missing columns: {missing}")
    if num_entities <= 0 or (rows["query_count"] <= 0).any():
        raise ValueError("num_entities and query_count must be positive")
    records: list[dict[str, Any]] = []
    grouped = rows.groupby(["seed", "deletion_rate", "method"], sort=False)
    for (seed, deletion_rate, method), frame in grouped:
        records.append(
            {
                "seed": int(seed),
                "deletion_rate": float(deletion_rate),
                "actual_deletion_rate": float(frame["actual_deletion_rate"].iloc[0]),
                "method": str(method),
                "query_count": int(frame["query_count"].sum()),
                "coverage": _weighted_mean(frame, "coverage"),
                "macro_time_coverage": float(frame["coverage"].mean()),
                "positive_undercoverage": float(
                    np.maximum(target - frame["coverage"], 0.0).mean()
                ),
                "worst_timestamp_coverage": float(frame["coverage"].min()),
                "fraction_timestamps_below_target": float(
                    (frame["coverage"] < target).mean()
                ),
                "mean_size": _weighted_mean(frame, "mean_size"),
                "macro_time_mean_size": float(frame["mean_size"].mean()),
                "normalized_mean_size": _weighted_mean(frame, "mean_size")
                / num_entities,
                "mrr": _weighted_mean(frame, "mrr"),
                "frequency_mrr": _weighted_mean(frame, "frequency_mrr"),
                "selected_half_life": float(frame["selected_half_life"].iloc[0]),
            }
        )
    result = pd.DataFrame(records)
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result["_method_rank"] = result["method"].map(method_rank).fillna(len(method_rank))
    return result.sort_values(
        ["seed", "deletion_rate", "_method_rank"], kind="stable"
    ).drop(columns="_method_rank").reset_index(drop=True)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (deletion_rate, method), frame in summary.groupby(
        ["deletion_rate", "method"], sort=False
    ):
        records.append(
            {
                "deletion_rate": float(deletion_rate),
                "method": str(method),
                "micro_coverage_mean": float(frame["coverage"].mean()),
                "micro_coverage_sd": float(frame["coverage"].std(ddof=1)),
                "mean_size_mean": float(frame["mean_size"].mean()),
                "mean_size_sd": float(frame["mean_size"].std(ddof=1)),
                "mrr_mean": float(frame["mrr"].mean()),
                "mrr_sd": float(frame["mrr"].std(ddof=1)),
                "frequency_mrr_mean": float(frame["frequency_mrr"].mean()),
                "macro_coverage_mean": float(frame["macro_time_coverage"].mean()),
                "macro_coverage_sd": float(frame["macro_time_coverage"].std(ddof=1)),
                "positive_undercoverage_mean": float(
                    frame["positive_undercoverage"].mean()
                ),
                "fraction_below_target_mean": float(
                    frame["fraction_timestamps_below_target"].mean()
                ),
                "worst_window_mean": float(frame["worst_timestamp_coverage"].mean()),
                "normalized_size_mean": float(frame["normalized_mean_size"].mean()),
            }
        )
    result = pd.DataFrame(records)
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result["_method_rank"] = result["method"].map(method_rank).fillna(len(method_rank))
    return result.sort_values(
        ["deletion_rate", "_method_rank"], kind="stable"
    ).drop(columns="_method_rank").reset_index(drop=True)


def coverage_by_time(
    rows: pd.DataFrame, timestamp_mapping: dict[int, str]
) -> pd.DataFrame:
    selected = rows[rows["method"].isin(PLOT_METHODS)].copy()
    selected["timestamp_id"] = selected["timestamp"].astype(int)
    selected["timestamp_raw"] = selected["timestamp_id"].map(timestamp_mapping)
    if selected["timestamp_raw"].isna().any():
        missing = sorted(selected.loc[selected["timestamp_raw"].isna(), "timestamp_id"].unique())
        raise ValueError(f"raw timestamp labels are missing for IDs: {missing}")
    try:
        selected["timestamp_plot"] = pd.to_numeric(selected["timestamp_raw"])
    except (TypeError, ValueError):
        ordered_ids = sorted(selected["timestamp_id"].unique())
        positions = {value: index for index, value in enumerate(ordered_ids)}
        selected["timestamp_plot"] = selected["timestamp_id"].map(positions)
    result = (
        selected.groupby(
            ["deletion_rate", "method", "timestamp_id", "timestamp_raw", "timestamp_plot"],
            as_index=False,
            sort=True,
        )
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_sd=("coverage", "std"),
            coverage_min=("coverage", "min"),
            coverage_max=("coverage", "max"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["deletion_rate", "method", "timestamp_id"], kind="stable")
    )
    return result


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {"bbox_inches": "tight"}
    if path.suffix == ".png":
        options["dpi"] = 180
    figure.savefig(path, **options)


def plot_coverage_by_time(
    by_time: pd.DataFrame, figures: Path, target: float
) -> None:
    rates = sorted(float(value) for value in by_time["deletion_rate"].unique())
    figure, axes = plt.subplots(1, len(rates), figsize=(12.0, 4.4), sharey=True)
    if len(rates) == 1:
        axes = [axes]
    for axis, rate in zip(axes, rates, strict=True):
        rate_rows = by_time[by_time["deletion_rate"] == rate]
        for method in PLOT_METHODS:
            frame = rate_rows[rate_rows["method"] == method].sort_values("timestamp_id")
            x = frame["timestamp_plot"].to_numpy(dtype=float)
            mean = frame["coverage_mean"].to_numpy(dtype=float)
            lower = frame["coverage_min"].to_numpy(dtype=float)
            upper = frame["coverage_max"].to_numpy(dtype=float)
            axis.plot(x, mean, color=METHOD_COLORS[method], linewidth=2.0, label=method)
            axis.fill_between(
                x,
                lower,
                upper,
                color=METHOD_COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
        axis.axhline(target, color="#222222", linestyle="--", linewidth=1.2)
        axis.set_title(f"Training deletion {rate:.0%}")
        axis.set_xlabel("Original test timestamp label")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed-label coverage")
    axes[0].set_ylim(0.74, 1.005)
    handles = [
        plt.Line2D([], [], color=METHOD_COLORS[method], linewidth=2.0, label=method)
        for method in PLOT_METHODS
    ]
    handles.append(
        plt.Line2D([], [], color="#222222", linestyle="--", label=f"target {target:.2f}")
    )
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    figure.subplots_adjust(bottom=0.24, wspace=0.08)
    for suffix in ("png", "pdf"):
        _save_figure(figure, figures / f"local_cpu_coverage.{suffix}")
    plt.close(figure)
    for suffix in ("png", "pdf"):
        shutil.copyfile(
            figures / f"local_cpu_coverage.{suffix}",
            figures / f"local_cpu_coverage_by_time.{suffix}",
        )


def plot_coverage_size_tradeoff(
    aggregate: pd.DataFrame, figures: Path, target: float, num_entities: int
) -> None:
    selected = aggregate[aggregate["method"].isin(PLOT_METHODS)]
    figure, axis = plt.subplots(figsize=(7.2, 4.9))
    markers = {0.0: "o", 0.3: "s"}
    for row in selected.itertuples(index=False):
        axis.scatter(
            row.normalized_size_mean,
            row.micro_coverage_mean,
            s=95,
            marker=markers.get(float(row.deletion_rate), "D"),
            color=METHOD_COLORS[str(row.method)],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    axis.axhline(target, color="#222222", linestyle="--", linewidth=1.2)
    axis.set_xlabel(f"Mean set size / {num_entities:,} entities")
    axis.set_ylabel("Micro observed-label coverage")
    axis.grid(alpha=0.2)
    method_handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="black", label=method, markersize=8
        )
        for method in PLOT_METHODS
    ]
    rate_handles = [
        plt.Line2D(
            [], [], marker=markers.get(rate, "D"), linestyle="", color="#777777",
            label=f"{rate:.0%} deletion", markersize=7
        )
        for rate in sorted(float(value) for value in selected["deletion_rate"].unique())
    ]
    figure.legend(
        handles=[*method_handles, *rate_handles],
        loc="lower center",
        ncol=5,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.20)
    for suffix in ("png", "pdf"):
        _save_figure(figure, figures / f"local_cpu_coverage_size_tradeoff.{suffix}")
    plt.close(figure)


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def export_paper_diagnostic(
    run_root: Path,
    paper_root: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    paper_root = paper_root.resolve()
    project_root = paper_root.parent
    resolved_config = _read_json(run_root / "config.resolved.yaml")
    target = float(resolved_config["target_coverage"])
    if data_root is None:
        configured = Path(resolved_config["data_path"])
        data_root = configured if configured.is_absolute() else project_root / configured
    data_root = data_root.resolve()

    source_dataset = _read_json(run_root / "dataset_manifest.json")
    dataset = build_dataset_manifest(source_dataset)
    verify_source_files(data_root, source_dataset)
    timestamp_mapping = build_timestamp_mapping(data_root)
    if len(timestamp_mapping) != int(dataset["num_timestamps"]):
        raise ValueError("timestamp mapping does not match dataset manifest")

    rows = pd.read_csv(run_root / "metrics" / "per_window.csv")
    if not (rows["calibration_max_timestamp"] < rows["timestamp"]).all():
        raise ValueError("timestamp leakage detected in per-window metrics")
    summary = condition_summary(rows, target, int(dataset["num_entities"]))
    aggregate = aggregate_summary(summary)
    by_time = coverage_by_time(rows, timestamp_mapping)

    data_dir = paper_root / "data"
    figure_dir = paper_root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    _write_json(dataset, data_dir / "icews14_dataset_manifest.json")
    _write_csv(summary, data_dir / "local_cpu_diagnostic_summary.csv")
    _write_csv(
        summary[
            [
                "coverage",
                "deletion_rate",
                "frequency_mrr",
                "mean_size",
                "method",
                "mrr",
                "query_count",
                "seed",
                "normalized_mean_size",
            ]
        ].rename(columns={"normalized_mean_size": "normalized_size"}),
        data_dir / "local_cpu_diagnostic_conditions.csv",
    )
    _write_csv(
        summary[
            [
                "seed",
                "deletion_rate",
                "method",
                "macro_time_coverage",
                "positive_undercoverage",
                "fraction_timestamps_below_target",
                "worst_timestamp_coverage",
                "macro_time_mean_size",
            ]
        ].rename(
            columns={
                "macro_time_coverage": "macro_coverage",
                "positive_undercoverage": "mean_positive_undercoverage",
            }
        ),
        data_dir / "local_cpu_diagnostic_macro_time_by_seed.csv",
    )
    _write_csv(aggregate, data_dir / "local_cpu_diagnostic_aggregate.csv")
    _write_csv(
        summary[
            [
                "seed",
                "deletion_rate",
                "actual_deletion_rate",
                "method",
                "selected_half_life",
            ]
        ],
        data_dir / "local_cpu_diagnostic_half_lives.csv",
    )
    if "adaptive_half_life" in rows:
        decision_columns = [
            "seed",
            "deletion_rate",
            "timestamp",
            "adaptive_half_life",
            "adaptive_predicted_coverage",
            "adaptive_predicted_mean_size",
            "adaptive_coverage_feasible",
            "drift_relation_tv",
            "drift_score_gap_ks",
            "drift_novelty_rate",
            "drift_log_query_count",
        ]
        available_decision_columns = [
            column for column in decision_columns if column in rows.columns
        ]
        _write_csv(
            rows.loc[rows["method"] == "adaptive", available_decision_columns]
            .drop_duplicates()
            .sort_values(["seed", "deletion_rate", "timestamp"], kind="stable"),
            data_dir / "local_cpu_diagnostic_adaptive_decisions.csv",
        )
    _write_csv(by_time, data_dir / "local_cpu_diagnostic_by_time.csv")
    plot_coverage_by_time(by_time, figure_dir, target)
    plot_coverage_size_tradeoff(
        aggregate, figure_dir, target, int(dataset["num_entities"])
    )

    relative_outputs = [
        "paper/data/icews14_dataset_manifest.json",
        "paper/data/local_cpu_diagnostic_aggregate.csv",
        "paper/data/local_cpu_diagnostic_by_time.csv",
        "paper/data/local_cpu_diagnostic_conditions.csv",
        "paper/data/local_cpu_diagnostic_half_lives.csv",
        "paper/data/local_cpu_diagnostic_macro_time_by_seed.csv",
        "paper/data/local_cpu_diagnostic_summary.csv",
        "paper/figures/local_cpu_coverage.pdf",
        "paper/figures/local_cpu_coverage.png",
        "paper/figures/local_cpu_coverage_by_time.pdf",
        "paper/figures/local_cpu_coverage_by_time.png",
        "paper/figures/local_cpu_coverage_size_tradeoff.pdf",
        "paper/figures/local_cpu_coverage_size_tradeoff.png",
    ]
    adaptive_decisions = "paper/data/local_cpu_diagnostic_adaptive_decisions.csv"
    if (project_root / adaptive_decisions).is_file():
        relative_outputs.insert(1, adaptive_decisions)
    environment = _read_json(run_root / "environment.json")
    run_manifest = _read_json(run_root / "run_manifest.json")
    script_path = Path(__file__).resolve()
    invocation = (
        f'python scripts/export_paper_diagnostic.py --run-root "{run_root}" '
        "--paper-root paper"
    )
    manifest = {
        "condition_count": int(len(summary[["seed", "deletion_rate"]].drop_duplicates())),
        "config_sha256": hashlib.sha256(
            json.dumps(resolved_config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "dataset_sha256": dataset["sha256"],
        "duration_seconds": float(run_manifest["duration_seconds"]),
        "environment": environment,
        "environment_json_sha256": sha256_file(run_root / "environment.json"),
        "evidence_class": "local CPU diagnostic; non-confirmatory",
        "export": {
            "invocation": invocation,
            "repository_base_commit": _git_commit(project_root),
            "script_path": "scripts/export_paper_diagnostic.py",
            "script_sha256": sha256_file(script_path),
            "timestamp_coordinate": (
                "Figures use original dataset timestamp labels; source metrics retain "
                "zero-based internal timestamp IDs."
            ),
        },
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hardware_summary": {
            "gpu": environment.get("gpu"),
            "gpu_memory_bytes": environment.get("gpu_memory_bytes"),
            "logical_cpu_count": environment.get("cpu_count"),
            "ram_bytes": environment.get("ram_bytes"),
        },
        "notes": [
            "Three seeds and two deletion rates; full ICEWS14 data.",
            "16-dimensional continuous-time DistMult, 3 epochs, CPU only.",
            "Not admissible for confirmatory tables, abstract, conclusion, or paper-level empirical claims.",
        ],
        "outputs": {
            relative: sha256_file(project_root / relative) for relative in relative_outputs
        },
        "query_rows": int(
            pq.ParquetFile(run_root / "metrics" / "per_query.parquet").metadata.num_rows
        ),
        "resolved_config_file_sha256": sha256_file(run_root / "config.resolved.yaml"),
        "run_id": run_manifest["run_id"],
        "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
        "run_root": str(run_root),
        "software_summary": {
            key: environment.get(key)
            for key in ("cuda_available", "cuda_version", "os", "python", "torch")
        },
        "source_files": dataset.get("source_files", {}),
        "source_metrics": {
            "SUCCESS_GATE.json": sha256_file(run_root / "SUCCESS_GATE.json"),
            "per_query.parquet": sha256_file(run_root / "metrics" / "per_query.parquet"),
            "per_window.csv": sha256_file(run_root / "metrics" / "per_window.csv"),
            "summary.json": sha256_file(run_root / "metrics" / "summary.json"),
        },
        "window_rows": int(len(rows)),
    }
    _write_json(manifest, data_dir / "local_cpu_diagnostic_manifest.json")
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    manifest = export_paper_diagnostic(args.run_root, args.paper_root, args.data_root)
    print(json.dumps(manifest["export"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
