"""Export block-length sensitivity for timestamp-block bootstrap diagnostics."""

from argparse import ArgumentParser
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

try:
    from export_timestamp_block_bootstrap import (
        DEFAULT_BLOCK_LENGTH,
        DEFAULT_BOOTSTRAP_SEED,
        DEFAULT_ITERATIONS,
        _bootstrap_statistic,
        _mrr_frame,
        _rolling_static_frames,
    )
except ModuleNotFoundError:
    from scripts.export_timestamp_block_bootstrap import (
        DEFAULT_BLOCK_LENGTH,
        DEFAULT_BOOTSTRAP_SEED,
        DEFAULT_ITERATIONS,
        _bootstrap_statistic,
        _mrr_frame,
        _rolling_static_frames,
    )


DEFAULT_BLOCK_LENGTHS = (3, 7, 14, 21)


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


def export_timestamp_block_sensitivity(
    run_root: Path,
    paper_root: Path,
    *,
    block_lengths: tuple[int, ...] = DEFAULT_BLOCK_LENGTHS,
    deletion_rate: float | None = None,
    target: float = 0.90,
    iterations: int = DEFAULT_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rows = pd.read_csv(run_root / "metrics" / "per_window.csv")
    if deletion_rate is None:
        deletion_rate = float(rows["deletion_rate"].max())
    if not block_lengths:
        raise ValueError("block_lengths must be nonempty")
    if any(length <= 0 for length in block_lengths):
        raise ValueError("block_lengths must be positive")

    undercoverage, coverage_gain = _rolling_static_frames(rows, deletion_rate, target)
    mrr = _mrr_frame(rows, deletion_rate)
    statistic_inputs = (
        (
            "rolling_undercoverage_reduction_vs_static",
            undercoverage,
            "undercoverage_reduction",
            None,
            "timestamp_macro",
        ),
        (
            "rolling_micro_coverage_gain_vs_static",
            coverage_gain,
            "coverage_gain",
            "query_count",
            "query_count",
        ),
        (
            "scorer_mrr_gain_vs_frequency",
            mrr,
            "mrr_gain",
            "query_count",
            "query_count",
        ),
    )

    records: list[dict[str, Any]] = []
    manifest_statistics: dict[str, dict[str, Any]] = {}
    for block_index, block_length in enumerate(block_lengths):
        seed_base = (
            bootstrap_seed
            if block_length == DEFAULT_BLOCK_LENGTH
            else bootstrap_seed + 100 * (block_index + 1)
        )
        for stat_index, (name, frame, value_column, weight_column, weighting) in enumerate(
            statistic_inputs
        ):
            result = _bootstrap_statistic(
                frame,
                value_column,
                weight_column=weight_column,
                block_length=block_length,
                iterations=iterations,
                bootstrap_seed=seed_base + stat_index,
            )
            key = f"{name}__block{block_length}"
            manifest_statistics[key] = result
            records.append(
                {
                    "statistic": name,
                    "deletion_rate": deletion_rate,
                    "target": target,
                    "block_length": block_length,
                    "observed": result["observed"],
                    "ci95_low": result["ci95"][0],
                    "ci95_high": result["ci95"][1],
                    "pvalue_positive": result["pvalue_positive"],
                    "iterations": iterations,
                    "seed_count": result["seed_count"],
                    "timestamp_count": result["timestamp_count"],
                    "weighting": weighting,
                }
            )

    summary = pd.DataFrame(records)
    data_dir = paper_root / "data" / "final_confirmatory"
    _write_csv(summary, data_dir / "timestamp_block_sensitivity_summary.csv")
    manifest = {
        "block_lengths": list(block_lengths),
        "block_scheme": (
            "seed-resampled circular moving-block bootstrap with one shared "
            "timestamp-block draw applied to all sampled seeds in each replicate"
        ),
        "bootstrap_seed": bootstrap_seed,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "deletion_rate": deletion_rate,
        "iterations": iterations,
        "run_root": str(run_root),
        "statistics": manifest_statistics,
        "target": target,
    }
    _write_json(manifest, data_dir / "timestamp_block_sensitivity.json")
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument(
        "--block-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_BLOCK_LENGTHS),
    )
    parser.add_argument("--deletion-rate", type=float)
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    manifest = export_timestamp_block_sensitivity(
        args.run_root,
        args.paper_root,
        block_lengths=tuple(args.block_lengths),
        deletion_rate=args.deletion_rate,
        target=args.target,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(manifest["statistics"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
