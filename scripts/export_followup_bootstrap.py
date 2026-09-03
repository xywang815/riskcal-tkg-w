"""Bootstrap the fixed primary contrasts from the follow-up calibration study."""

from argparse import ArgumentParser
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.export_expansion_bootstrap import bootstrap_statistic
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from export_expansion_bootstrap import bootstrap_statistic


DEFAULT_BLOCK_LENGTHS = (3, 7, 14, 21)
DEFAULT_ITERATIONS = 20_000
DEFAULT_BOOTSTRAP_SEED = 20260903
TARGET_COVERAGE = 0.90


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g", lineterminator="\n")
    temporary.replace(path)


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _paired_contrast(
    rows: pd.DataFrame,
    *,
    left_method: str,
    right_method: str,
    metric: str,
    weight_column: str | None,
    transform: str = "difference",
) -> pd.DataFrame:
    columns = ["seed", "timestamp", "method", metric]
    if weight_column is not None:
        columns.append(weight_column)
    selected = rows[rows["method"].isin((left_method, right_method))][columns]
    values = selected.pivot(
        index=["seed", "timestamp"], columns="method", values=metric
    )
    if not {left_method, right_method}.issubset(values.columns):
        raise ValueError(f"missing method pair: {left_method}, {right_method}")
    complete = values[[left_method, right_method]].dropna().reset_index()
    if complete.empty:
        raise ValueError(f"no complete observations for {metric}")
    if transform == "difference":
        contrast = complete[left_method] - complete[right_method]
    elif transform == "absolute_target_error_reduction":
        contrast = (
            np.abs(complete[right_method] - TARGET_COVERAGE)
            - np.abs(complete[left_method] - TARGET_COVERAGE)
        )
    else:
        raise ValueError(f"unknown contrast transform: {transform}")
    output = pd.DataFrame(
        {
            "seed": complete["seed"].astype(int),
            "timestamp": complete["timestamp"].astype(int),
            "value": contrast.astype(float),
        }
    )
    if weight_column is not None:
        weights = selected[selected["method"].eq(left_method)].set_index(
            ["seed", "timestamp"]
        )[weight_column]
        index = pd.MultiIndex.from_frame(output[["seed", "timestamp"]])
        output["weight"] = weights.reindex(index).to_numpy(dtype=float)
        if output["weight"].isna().any() or (output["weight"] <= 0).any():
            raise ValueError("contrast weights must be present and positive")
    return output


def _case_contrasts(rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows = rows.copy()
    gap_column = "multi_minus_single_full_set_coverage"
    if gap_column not in rows:
        required = {"multi_full_set_coverage", "single_full_set_coverage"}
        missing = sorted(required - set(rows.columns))
        if missing:
            raise ValueError(f"cannot derive multi-answer gap; missing {missing}")
        rows[gap_column] = (
            rows["multi_full_set_coverage"] - rows["single_full_set_coverage"]
        )
    contrasts: list[dict[str, Any]] = []

    def add(
        *,
        family: str,
        comparison: str,
        metric: str,
        input_metric: str | None = None,
        left: str,
        right: str,
        weight: str | None,
        transform: str = "difference",
    ) -> None:
        contrasts.append(
            {
                "family": family,
                "comparison": comparison,
                "metric": metric,
                "left_method": left,
                "right_method": right,
                "weighting": "timestamp_macro" if weight is None else weight,
                "frame": _paired_contrast(
                    rows,
                    left_method=left,
                    right_method=right,
                    metric=input_metric or metric,
                    weight_column=weight,
                    transform=transform,
                ),
            }
        )

    for metric, weight in (
        ("label_coverage", "label_count"),
        ("full_set_coverage", "query_count"),
        ("mean_set_size", "query_count"),
    ):
        add(
            family="history_policy",
            comparison="label_margin_rolling-minus-label_margin_static",
            metric=metric,
            left="label_margin_rolling",
            right="label_margin_static",
            weight=weight,
        )
    add(
        family="history_policy",
        comparison="label_margin_rolling-minus-label_margin_static",
        metric="absolute_target_error_reduction",
        input_metric="label_coverage",
        left="label_margin_rolling",
        right="label_margin_static",
        weight="label_count",
        transform="absolute_target_error_reduction",
    )

    for score in ("negscore", "minmax", "softmax"):
        for metric, weight in (
            ("label_coverage", "label_count"),
            ("mean_set_size", "query_count"),
        ):
            add(
                family="score_function",
                comparison=f"label_{score}_rolling-minus-label_margin_rolling",
                metric=metric,
                left=f"label_{score}_rolling",
                right="label_margin_rolling",
                weight=weight,
            )

    for metric, weight in (
        ("full_set_coverage", "query_count"),
        ("mean_set_size", "query_count"),
    ):
        add(
            family="query_objective",
            comparison="query_max_margin_rolling-minus-label_margin_rolling",
            metric=metric,
            left="query_max_margin_rolling",
            right="label_margin_rolling",
            weight=weight,
        )
    add(
        family="query_objective",
        comparison="query_max_margin_rolling-minus-label_margin_rolling",
        metric=gap_column,
        left="query_max_margin_rolling",
        right="label_margin_rolling",
        weight=None,
    )
    return contrasts


def export_followup_bootstrap(
    followup_root: Path,
    output_dir: Path,
    *,
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    iterations: int = DEFAULT_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    followup_root = followup_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if iterations <= 0 or not block_lengths or any(value <= 0 for value in block_lengths):
        raise ValueError("iterations and block lengths must be positive")
    source_manifest_path = followup_root / "followup_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise ValueError("follow-up manifest is not complete")
    rows_path = followup_root / "followup_by_timestamp.csv"
    rows = pd.read_csv(rows_path)
    if len(rows) != int(source_manifest.get("timestamp_row_count", -1)):
        raise ValueError("timestamp row count does not match source manifest")

    records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    sequence = 0
    for case, case_rows in rows.groupby("case", sort=True):
        metadata = {
            "case": case,
            "dataset_mode": str(case_rows["dataset_mode"].iloc[0]),
            "model_name": str(case_rows["model_name"].iloc[0]),
            "deletion_rate": float(case_rows["deletion_rate"].iloc[0]),
        }
        for contrast in _case_contrasts(case_rows):
            for block_length in block_lengths:
                result = bootstrap_statistic(
                    contrast["frame"],
                    "value",
                    weight_column=(
                        None
                        if contrast["weighting"] == "timestamp_macro"
                        else "weight"
                    ),
                    block_length=int(block_length),
                    iterations=iterations,
                    bootstrap_seed=bootstrap_seed + sequence,
                )
                sequence += 1
                record = {
                    **metadata,
                    "family": contrast["family"],
                    "comparison": contrast["comparison"],
                    "metric": contrast["metric"],
                    "left_method": contrast["left_method"],
                    "right_method": contrast["right_method"],
                    "observed": result["observed"],
                    "ci95_low": result["ci95"][0],
                    "ci95_high": result["ci95"][1],
                    "pvalue_positive": result["pvalue_positive"],
                    "pvalue_negative": result["pvalue_negative"],
                    "block_length": int(block_length),
                    "iterations": iterations,
                    "seed_count": result["seed_count"],
                    "timestamp_count": result["timestamp_count"],
                    "excluded_timestamp_count": result["excluded_timestamp_count"],
                    "weighting": contrast["weighting"],
                }
                records.append(record)
                for seed, value in result["observed_by_seed"].items():
                    seed_records.append(
                        {
                            **metadata,
                            "family": contrast["family"],
                            "comparison": contrast["comparison"],
                            "metric": contrast["metric"],
                            "block_length": int(block_length),
                            "seed": int(seed),
                            "observed": value,
                            "weighting": contrast["weighting"],
                        }
                    )

    summary = pd.DataFrame(records).sort_values(
        ["family", "case", "comparison", "metric", "block_length"],
        kind="stable",
    )
    by_seed = pd.DataFrame(seed_records).sort_values(
        ["family", "case", "comparison", "metric", "block_length", "seed"],
        kind="stable",
    )
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "followup_bootstrap_summary.csv"
    seed_path = output_dir / "followup_bootstrap_by_seed.csv"
    _write_csv(summary, summary_path)
    _write_csv(by_seed, seed_path)
    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        "block_lengths": [int(value) for value in block_lengths],
        "bootstrap_seed": bootstrap_seed,
        "case_count": int(rows["case"].nunique()),
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(repo_root),
        "iterations": iterations,
        "resampling_scheme": (
            "equal-seed circular moving-block bootstrap with one shared "
            "timestamp-block draw per replicate"
        ),
        "source_git_commit": source_manifest.get("git_commit"),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_rows_sha256": _sha256(rows_path),
        "statistics_count": int(len(summary)),
        "status": "complete",
        "target_coverage": TARGET_COVERAGE,
    }
    manifest_path = output_dir / "followup_bootstrap_manifest.json"
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
    parser.add_argument("--followup-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--block-lengths", type=int, nargs="+", default=list(DEFAULT_BLOCK_LENGTHS)
    )
    args = parser.parse_args()
    manifest = export_followup_bootstrap(
        args.followup_root,
        args.output_dir,
        block_lengths=args.block_lengths,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
