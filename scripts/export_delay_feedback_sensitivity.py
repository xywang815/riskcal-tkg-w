"""Export delayed-feedback sensitivity diagnostics for rolling calibration.

This exporter reuses a completed confirmatory run and its checkpoints.  It does
not retrain the scorer.  The only changed protocol detail is when completed
test-timestamp labels are allowed to enter the rolling calibration pool.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from riskcal_tkg.calibration import (
    finite_sample_quantile,
    margin_nonconformity,
)
from riskcal_tkg.data import (
    QuadrupleTable,
    add_inverse_relations,
    split_calibration_roles,
    temporal_split,
)
from scripts.export_timestamp_block_bootstrap import _bootstrap_statistic
from scripts.export_window_ablation import (
    ScoreHistory,
    _evaluate_threshold,
    _load_condition_model,
    _score_all_objects,
    _score_batches,
    sha256_file,
)


DEFAULT_DELAYS = (0, 1, 3, 7, 14)
DEFAULT_RETENTIONS = (1.0, 0.9, 0.7)
DEFAULT_FEEDBACK_SEED = 20260904
DEFAULT_BOOTSTRAP_SEED = 20260818
DEFAULT_BLOCK_LENGTH = 7
DEFAULT_ITERATIONS = 20_000
METHOD_ORDER = ["static", "rolling"]


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


def _finite_or_inf(value: float) -> str:
    return "inf" if np.isposinf(float(value)) else f"{float(value):g}"


def _method_sort(frame: pd.DataFrame, extra_columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result = frame.copy()
    result["_method_rank"] = result["method"].map(method_rank).fillna(len(method_rank))
    columns = [
        "deletion_rate",
        "feedback_retention",
        "extra_delay_blocks",
        *(extra_columns or []),
        "_method_rank",
        "method",
    ]
    present = [column for column in columns if column in result.columns]
    return result.sort_values(present, kind="stable").drop(
        columns="_method_rank"
    ).reset_index(drop=True)


def _validate_delays(delays: tuple[int, ...]) -> tuple[int, ...]:
    if not delays:
        raise ValueError("delays must be nonempty")
    if any(delay < 0 for delay in delays):
        raise ValueError("delays must be nonnegative")
    return tuple(sorted(set(int(delay) for delay in delays)))


def _validate_retentions(retentions: tuple[float, ...]) -> tuple[float, ...]:
    if not retentions:
        raise ValueError("feedback retentions must be nonempty")
    if any(not 0.0 < retention <= 1.0 for retention in retentions):
        raise ValueError("feedback retentions must be in (0, 1]")
    return tuple(sorted(set(float(value) for value in retentions), reverse=True))


def release_due_feedback(
    history: ScoreHistory,
    pending: list[tuple[int, int, np.ndarray]],
    current_index: int,
) -> list[tuple[int, int, np.ndarray]]:
    """Move feedback batches whose release index has arrived into history."""
    remaining: list[tuple[int, int, np.ndarray]] = []
    for release_index, timestamp, margins in pending:
        if release_index <= current_index:
            history.add(timestamp, margins)
        else:
            remaining.append((release_index, timestamp, margins))
    return remaining


def summarize_delay_rows(rows: pd.DataFrame, target_coverage: float) -> pd.DataFrame:
    rows = rows.copy()
    if "feedback_retention" not in rows:
        rows["feedback_retention"] = 1.0
    required = {
        "seed",
        "deletion_rate",
        "extra_delay_blocks",
        "method",
        "query_count",
        "coverage",
        "mean_size",
        "median_size",
        "p90_size",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"delay rows are missing columns: {missing}")
    records: list[dict[str, Any]] = []
    for (seed, deletion_rate, delay, retention, method), frame in rows.groupby(
        [
            "seed",
            "deletion_rate",
            "extra_delay_blocks",
            "feedback_retention",
            "method",
        ],
        sort=False,
    ):
        weights = frame["query_count"].to_numpy(dtype=float)
        records.append(
            {
                "seed": int(seed),
                "deletion_rate": float(deletion_rate),
                "extra_delay_blocks": int(delay),
                "feedback_retention": float(retention),
                "method": str(method),
                "query_count": int(frame["query_count"].sum()),
                "coverage": float(np.average(frame["coverage"], weights=weights)),
                "macro_time_coverage": float(frame["coverage"].mean()),
                "positive_undercoverage": float(
                    np.maximum(target_coverage - frame["coverage"], 0.0).mean()
                ),
                "fraction_timestamps_below_target": float(
                    (frame["coverage"] < target_coverage).mean()
                ),
                "mean_size": float(np.average(frame["mean_size"], weights=weights)),
                "median_size": float(np.average(frame["median_size"], weights=weights)),
                "p90_size": float(np.average(frame["p90_size"], weights=weights)),
                "pool_score_count_mean": float(frame["pool_score_count"].mean()),
                "pool_span_blocks_mean": float(frame["pool_span_blocks"].mean()),
            }
        )
    return _method_sort(pd.DataFrame(records), ["seed"])


def aggregate_delay_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "coverage",
        "macro_time_coverage",
        "positive_undercoverage",
        "fraction_timestamps_below_target",
        "mean_size",
        "median_size",
        "p90_size",
        "pool_score_count_mean",
        "pool_span_blocks_mean",
    ]
    records: list[dict[str, Any]] = []
    for (deletion_rate, delay, retention, method), frame in summary.groupby(
        [
            "deletion_rate",
            "extra_delay_blocks",
            "feedback_retention",
            "method",
        ],
        sort=False,
    ):
        record: dict[str, Any] = {
            "deletion_rate": float(deletion_rate),
            "extra_delay_blocks": int(delay),
            "feedback_retention": float(retention),
            "method": str(method),
            "seed_count": int(frame["seed"].nunique()),
        }
        for metric in metrics:
            values = frame[metric].dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        records.append(record)
    return _method_sort(pd.DataFrame(records))


def build_delay_effect_frames(
    rows: pd.DataFrame,
    *,
    deletion_rate: float,
    target_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    if "feedback_retention" not in rows:
        rows["feedback_retention"] = 1.0
    selected = rows[
        (rows["deletion_rate"] == deletion_rate)
        & (rows["method"].isin(["static", "rolling"]))
    ].copy()
    required = {
        "seed",
        "timestamp",
        "extra_delay_blocks",
        "method",
        "coverage",
        "query_count",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"delay rows are missing columns: {missing}")
    pivot = selected.pivot_table(
        index=["seed", "timestamp", "extra_delay_blocks", "feedback_retention"],
        columns="method",
        values=["coverage", "query_count"],
        aggfunc="first",
    )
    if pivot.isna().any().any():
        raise ValueError("static and rolling rows must exist for every seed/timestamp/delay")
    base = pivot.reset_index()
    undercoverage = pd.DataFrame(
        {
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "extra_delay_blocks": base["extra_delay_blocks"].astype(int),
            "feedback_retention": base["feedback_retention"].astype(float),
            "undercoverage_reduction": np.maximum(
                target_coverage - base[("coverage", "static")], 0.0
            )
            - np.maximum(target_coverage - base[("coverage", "rolling")], 0.0),
        }
    )
    coverage_gain = pd.DataFrame(
        {
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "extra_delay_blocks": base["extra_delay_blocks"].astype(int),
            "feedback_retention": base["feedback_retention"].astype(float),
            "query_count": base[("query_count", "static")].astype(float),
            "coverage_gain": base[("coverage", "rolling")]
            - base[("coverage", "static")],
        }
    )
    return undercoverage, coverage_gain


def bootstrap_delay_effects(
    rows: pd.DataFrame,
    *,
    deletion_rate: float,
    target_coverage: float,
    block_length: int,
    iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    undercoverage, coverage_gain = build_delay_effect_frames(
        rows,
        deletion_rate=deletion_rate,
        target_coverage=target_coverage,
    )
    records: list[dict[str, Any]] = []
    policies = sorted(
        {
            (int(row.extra_delay_blocks), float(row.feedback_retention))
            for row in undercoverage.itertuples()
        }
    )
    for index, (delay, retention) in enumerate(policies):
        under_mask = (
            (undercoverage["extra_delay_blocks"] == delay)
            & np.isclose(undercoverage["feedback_retention"], retention)
        )
        gain_mask = (
            (coverage_gain["extra_delay_blocks"] == delay)
            & np.isclose(coverage_gain["feedback_retention"], retention)
        )
        delay_under = undercoverage[under_mask]
        delay_gain = coverage_gain[gain_mask]
        for offset, (name, frame, value_column, weight_column, weighting) in enumerate(
            (
                (
                    "rolling_undercoverage_reduction_vs_static",
                    delay_under,
                    "undercoverage_reduction",
                    None,
                    "timestamp_macro",
                ),
                (
                    "rolling_micro_coverage_gain_vs_static",
                    delay_gain,
                    "coverage_gain",
                    "query_count",
                    "query_count",
                ),
            )
        ):
            result = _bootstrap_statistic(
                frame,
                value_column,
                weight_column=weight_column,
                block_length=block_length,
                iterations=iterations,
                bootstrap_seed=bootstrap_seed + 10 * index + offset,
            )
            records.append(
                {
                    "statistic": name,
                    "deletion_rate": float(deletion_rate),
                    "target": float(target_coverage),
                    "extra_delay_blocks": int(delay),
                    "feedback_retention": float(retention),
                    "observed": result["observed"],
                    "ci95_low": result["ci95"][0],
                    "ci95_high": result["ci95"][1],
                    "pvalue_positive": result["pvalue_positive"],
                    "iterations": int(iterations),
                    "block_length": int(block_length),
                    "seed_count": result["seed_count"],
                    "timestamp_count": result["timestamp_count"],
                    "weighting": weighting,
                }
            )
    return pd.DataFrame(records).sort_values(
        ["feedback_retention", "extra_delay_blocks", "statistic"], kind="stable"
    ).reset_index(drop=True)


def _evaluate_condition(
    *,
    run_root: Path,
    table: QuadrupleTable,
    split: Any,
    seed: int,
    deletion_rate: float,
    embedding_dim: int,
    batch_size: int,
    rolling_window: int,
    target_coverage: float,
    delays: tuple[int, ...],
    retentions: tuple[float, ...],
    feedback_seed: int,
    device: Any,
) -> list[dict[str, Any]]:
    relation_count = len(table.relation_to_id)
    roles = split_calibration_roles(split.calibration)
    model = _load_condition_model(
        run_root,
        seed=seed,
        deletion_rate=deletion_rate,
        table=table,
        relation_count=relation_count,
        embedding_dim=embedding_dim,
        device=device,
    )
    initial_batches, initial_margins = _score_batches(
        model,
        roles.final_calibration,
        relation_count,
        batch_size,
        device,
    )
    initial_scores = np.concatenate(initial_margins)
    static_threshold = finite_sample_quantile(
        initial_scores,
        alpha=1.0 - target_coverage,
    )

    policies = tuple(
        (delay, retention) for delay in delays for retention in retentions
    )
    histories = {policy: ScoreHistory() for policy in policies}
    pending = {policy: [] for policy in policies}
    feedback_rngs = {
        policy: np.random.default_rng(
            feedback_seed
            + 100_000 * int(seed)
            + 1_000 * int(policy[0])
            + int(round(100 * policy[1]))
        )
        for policy in policies
    }
    for batch, margins in zip(initial_batches, initial_margins, strict=True):
        for history in histories.values():
            history.add(batch.timestamp, margins)

    rows: list[dict[str, Any]] = []
    test_timestamps = [int(value) for value in np.unique(split.test.timestamps)]
    for timestamp_index, timestamp in enumerate(test_timestamps):
        for policy in policies:
            pending[policy] = release_due_feedback(
                histories[policy],
                pending[policy],
                timestamp_index,
            )

        raw_facts = split.test.values[split.test.timestamps == timestamp]
        facts = add_inverse_relations(raw_facts, relation_count)
        scores = _score_all_objects(model, facts, batch_size, device)
        labels = facts[:, 2].copy()
        query_count = int(len(labels))
        current_margins = margin_nonconformity(scores, labels)

        for delay, retention in policies:
            policy = (delay, retention)
            static_pool_scores, static_pool_timestamps = histories[
                policy
            ].values_before(timestamp)
            rows.append(
                {
                    **_evaluate_threshold(
                        seed=seed,
                        deletion_rate=deletion_rate,
                        method="static",
                        timestamp=timestamp,
                        scores=scores,
                        labels=labels,
                        threshold=static_threshold,
                        query_count=query_count,
                        pool_score_count=len(static_pool_scores),
                        pool_span_blocks=int(timestamp - static_pool_timestamps.min()),
                    ),
                    "extra_delay_blocks": int(delay),
                    "feedback_retention": float(retention),
                }
            )
            pool_scores, pool_timestamps = histories[policy].values_before(
                timestamp,
                max_count=rolling_window,
            )
            rows.append(
                {
                    **_evaluate_threshold(
                        seed=seed,
                        deletion_rate=deletion_rate,
                        method="rolling",
                        timestamp=timestamp,
                        scores=scores,
                        labels=labels,
                        threshold=finite_sample_quantile(
                            pool_scores,
                            alpha=1.0 - target_coverage,
                        ),
                        query_count=query_count,
                        pool_score_count=len(pool_scores),
                        pool_span_blocks=int(timestamp - pool_timestamps.min()),
                    ),
                    "extra_delay_blocks": int(delay),
                    "feedback_retention": float(retention),
                }
            )
            keep = feedback_rngs[policy].random(len(current_margins)) < retention
            retained_margins = current_margins[keep]
            if len(retained_margins):
                pending[policy].append(
                    (timestamp_index + delay + 1, timestamp, retained_margins)
                )

    model.to("cpu")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass
    return rows


def export_delay_feedback_sensitivity(
    run_root: Path,
    paper_root: Path,
    *,
    data_root: Path | None = None,
    output_name: str = "final_confirmatory",
    delays: tuple[int, ...] = DEFAULT_DELAYS,
    retentions: tuple[float, ...] = DEFAULT_RETENTIONS,
    feedback_seed: int = DEFAULT_FEEDBACK_SEED,
    effect_deletion_rate: float | None = None,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    iterations: int = DEFAULT_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    device_name: str = "auto",
) -> dict[str, Any]:
    import torch

    from riskcal_tkg.config import load_config
    from riskcal_tkg.data import load_configured_table

    run_root = run_root.resolve()
    paper_root = paper_root.resolve()
    delays = _validate_delays(delays)
    retentions = _validate_retentions(retentions)
    config = load_config(run_root / "config.resolved.yaml")
    if data_root is not None:
        config = type(config)(
            **{
                **config.__dict__,
                "data_path": data_root,
            }
        )
    table = load_configured_table(config)
    split = temporal_split(
        table,
        train_fraction=config.train_fraction,
        calibration_fraction=config.calibration_fraction,
    )
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name in {"cpu", "cuda"}:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device(device_name)
    else:
        raise ValueError("device must be auto, cpu, or cuda")

    seeds = config.seeds or (config.seed,)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for deletion_rate in config.deletion_rates:
            rows.extend(
                _evaluate_condition(
                    run_root=run_root,
                    table=table,
                    split=split,
                    seed=int(seed),
                    deletion_rate=float(deletion_rate),
                    embedding_dim=config.embedding_dim,
                    batch_size=config.batch_size,
                    rolling_window=config.rolling_window,
                    target_coverage=config.target_coverage,
                    delays=delays,
                    retentions=retentions,
                    feedback_seed=feedback_seed,
                    device=device,
                )
            )

    data_dir = paper_root / "data" / output_name
    row_frame = pd.DataFrame(rows)
    by_seed = summarize_delay_rows(row_frame, config.target_coverage)
    summary = aggregate_delay_summary(by_seed)
    if effect_deletion_rate is None:
        effect_deletion_rate = float(max(config.deletion_rates))
    effects = bootstrap_delay_effects(
        row_frame,
        deletion_rate=float(effect_deletion_rate),
        target_coverage=config.target_coverage,
        block_length=block_length,
        iterations=iterations,
        bootstrap_seed=bootstrap_seed,
    )

    _write_csv(row_frame, data_dir / "delay_feedback_by_timestamp.csv")
    _write_csv(by_seed, data_dir / "delay_feedback_by_seed.csv")
    _write_csv(summary, data_dir / "delay_feedback_summary.csv")
    _write_csv(effects, data_dir / "delay_feedback_effects.csv")

    outputs = {
        "delay_feedback_by_timestamp.csv": sha256_file(
            data_dir / "delay_feedback_by_timestamp.csv"
        ),
        "delay_feedback_by_seed.csv": sha256_file(
            data_dir / "delay_feedback_by_seed.csv"
        ),
        "delay_feedback_summary.csv": sha256_file(
            data_dir / "delay_feedback_summary.csv"
        ),
        "delay_feedback_effects.csv": sha256_file(
            data_dir / "delay_feedback_effects.csv"
        ),
    }
    manifest = {
        "block_length": int(block_length),
        "bootstrap_seed": int(bootstrap_seed),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "definition": {
            "extra_delay_blocks": (
                "Additional completed test-timestamp batches withheld before a "
                "test batch can update the rolling calibration pool.  Delay 0 "
                "matches the ordinary batch-completion prequential update."
            ),
            "rolling": (
                f"Equal-weight conformal threshold over the most recent "
                f"{config.rolling_window} revealed nonconformity scores."
            ),
            "static": (
                "Fixed empirical threshold from the final initial calibration "
                "interval; repeated across delay values only for paired summaries."
            ),
        },
        "delays": list(delays),
        "feedback_retentions": list(retentions),
        "feedback_seed": int(feedback_seed),
        "device": str(device),
        "effect_deletion_rate": float(effect_deletion_rate),
        "iterations": int(iterations),
        "output_name": output_name,
        "outputs": outputs,
        "run_root": str(run_root),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_coverage": float(config.target_coverage),
    }
    _write_json(manifest, data_dir / "delay_feedback_manifest.json")
    manifest["outputs"]["delay_feedback_manifest.json"] = sha256_file(
        data_dir / "delay_feedback_manifest.json"
    )
    _write_json(manifest, data_dir / "delay_feedback_manifest.json")
    return manifest


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one integer is required")
    return parsed


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one number is required")
    return parsed


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-name", default="final_confirmatory")
    parser.add_argument(
        "--delays",
        default="0,1,3,7,14",
        help="Comma-separated additional feedback delays in test timestamp blocks.",
    )
    parser.add_argument(
        "--retentions",
        default="1.0,0.9,0.7",
        help="Comma-separated fractions of completed test feedback retained.",
    )
    parser.add_argument("--feedback-seed", type=int, default=DEFAULT_FEEDBACK_SEED)
    parser.add_argument("--effect-deletion-rate", type=float)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    manifest = export_delay_feedback_sensitivity(
        args.run_root,
        args.paper_root,
        data_root=args.data_root,
        output_name=args.output_name,
        delays=_parse_int_tuple(args.delays),
        retentions=_parse_float_tuple(args.retentions),
        feedback_seed=args.feedback_seed,
        effect_deletion_rate=args.effect_deletion_rate,
        block_length=args.block_length,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
        device_name=args.device,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
