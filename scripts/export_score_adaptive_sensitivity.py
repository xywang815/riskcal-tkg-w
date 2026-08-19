"""Export target/tolerance sensitivity for score-adaptive shortlists.

This exporter reuses a completed confirmatory run and its checkpoints.  For
each seed/deletion condition it scores the calibration and test streams once,
then evaluates a grid of target coverages and selector tolerances on the same
cached score batches.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from riskcal_tkg.calibration import (  # noqa: E402
    finite_sample_quantile,
    margin_nonconformity,
    prediction_set_mask,
)
from riskcal_tkg.data import (  # noqa: E402
    split_calibration_roles,
    temporal_split,
)
from scripts.export_query_level_diagnostics import sha256_file  # noqa: E402
from scripts.export_score_adaptive_shortlist import (  # noqa: E402
    DEFAULT_BLOCK_LENGTH,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_ITERATIONS,
    DEFAULT_RAPS_K,
    DEFAULT_RAPS_LAMBDAS,
    MassCandidate,
    _candidate_threshold,
    _evaluate_candidate_on_batches,
    _filter_numbers,
    _rolling_threshold,
    _table_batches,
    build_mass_candidates,
    mass_nonconformity_from_cumulative,
    mass_prediction_mask_from_cumulative,
    select_mass_candidate,
)
from scripts.export_shortlist_calibration import (  # noqa: E402
    _summarize_mask,
    topk_prediction_mask,
    true_label_ranks,
)
from scripts.export_timestamp_block_bootstrap import _bootstrap_statistic  # noqa: E402
from scripts.export_window_ablation import (  # noqa: E402
    ScoreHistory,
    _load_condition_model,
)


DEFAULT_TARGET_COVERAGES = (0.88, 0.90, 0.92)
DEFAULT_SELECTION_TOLERANCES = (0.0, 0.01, 0.02, 0.03)
SENSITIVITY_METHODS = ("margin_rolling", "rank_rolling", "adaptive_mass_rolling")


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


def _parse_float_tuple(value: str | None) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    parsed = tuple(float(part) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one float is required")
    return parsed


def _parse_int_tuple(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    parsed = tuple(int(part) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one integer is required")
    return parsed


def _validate_probability_grid(name: str, values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    parsed = tuple(float(value) for value in values)
    if any(value <= 0.0 or value >= 1.0 for value in parsed):
        raise ValueError(f"{name} values must be inside (0, 1)")
    return tuple(sorted(set(parsed)))


def _validate_tolerances(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        raise ValueError("selection tolerances must not be empty")
    parsed = tuple(float(value) for value in values)
    if any(value < 0.0 or value >= 1.0 for value in parsed):
        raise ValueError("selection tolerances must be in [0, 1)")
    return tuple(sorted(set(parsed)))


def summarize_sensitivity_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "target_coverage",
        "selection_tolerance",
        "seed",
        "deletion_rate",
        "method",
        "timestamp",
        "label_row_count",
        "unique_query_count",
        "observed_label_coverage",
        "full_set_coverage",
        "partial_answer_recall",
        "mean_size",
        "median_size",
        "p90_size",
        "full_vocabulary_set_rate",
        "pool_score_count",
        "pool_span_blocks",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"sensitivity rows are missing columns: {missing}")
    records: list[dict[str, Any]] = []
    groups = [
        "target_coverage",
        "selection_tolerance",
        "seed",
        "deletion_rate",
        "method",
    ]
    for key, frame in rows.groupby(groups, sort=False):
        target, tolerance, seed, deletion_rate, method = key
        query_weights = frame["unique_query_count"].to_numpy(dtype=float)
        label_weights = frame["label_row_count"].to_numpy(dtype=float)
        target_float = float(target)
        records.append(
            {
                "target_coverage": target_float,
                "selection_tolerance": float(tolerance),
                "seed": int(seed),
                "deletion_rate": float(deletion_rate),
                "method": str(method),
                "timestamp_count": int(frame["timestamp"].nunique()),
                "label_row_count": int(frame["label_row_count"].sum()),
                "unique_query_count": int(frame["unique_query_count"].sum()),
                "observed_label_coverage": float(
                    np.average(frame["observed_label_coverage"], weights=label_weights)
                ),
                "full_set_coverage": float(
                    np.average(frame["full_set_coverage"], weights=query_weights)
                ),
                "macro_time_full_set_coverage": float(frame["full_set_coverage"].mean()),
                "partial_answer_recall": float(
                    np.average(frame["partial_answer_recall"], weights=query_weights)
                ),
                "positive_label_undercoverage": float(
                    np.maximum(target_float - frame["observed_label_coverage"], 0.0).mean()
                ),
                "positive_full_set_undercoverage": float(
                    np.maximum(target_float - frame["full_set_coverage"], 0.0).mean()
                ),
                "fraction_timestamps_label_below_target": float(
                    (frame["observed_label_coverage"] < target_float).mean()
                ),
                "fraction_timestamps_full_set_below_target": float(
                    (frame["full_set_coverage"] < target_float).mean()
                ),
                "mean_size": float(np.average(frame["mean_size"], weights=query_weights)),
                "median_size": float(
                    np.average(frame["median_size"], weights=query_weights)
                ),
                "p90_size": float(np.average(frame["p90_size"], weights=query_weights)),
                "full_vocabulary_set_rate": float(
                    np.average(frame["full_vocabulary_set_rate"], weights=query_weights)
                ),
                "pool_score_count_mean": float(frame["pool_score_count"].mean()),
                "pool_span_blocks_mean": float(frame["pool_span_blocks"].mean()),
            }
        )
    return _sort_sensitivity(pd.DataFrame(records), ["seed"])


def aggregate_sensitivity_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "observed_label_coverage",
        "full_set_coverage",
        "macro_time_full_set_coverage",
        "partial_answer_recall",
        "positive_label_undercoverage",
        "positive_full_set_undercoverage",
        "fraction_timestamps_label_below_target",
        "fraction_timestamps_full_set_below_target",
        "mean_size",
        "median_size",
        "p90_size",
        "full_vocabulary_set_rate",
        "pool_score_count_mean",
        "pool_span_blocks_mean",
    ]
    records: list[dict[str, Any]] = []
    groups = ["target_coverage", "selection_tolerance", "deletion_rate", "method"]
    for key, frame in summary.groupby(groups, sort=False):
        target, tolerance, deletion_rate, method = key
        record: dict[str, Any] = {
            "target_coverage": float(target),
            "selection_tolerance": float(tolerance),
            "deletion_rate": float(deletion_rate),
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
    return _sort_sensitivity(pd.DataFrame(records))


def _sort_sensitivity(
    frame: pd.DataFrame,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "method" in result.columns:
        rank = {method: index for index, method in enumerate(SENSITIVITY_METHODS)}
        result["_method_rank"] = result["method"].map(rank).fillna(len(rank))
    columns = [
        "target_coverage",
        "selection_tolerance",
        "deletion_rate",
        *(extra_columns or []),
        "_method_rank",
        "method",
    ]
    present = [column for column in columns if column in result.columns]
    sorted_frame = result.sort_values(present, kind="stable").reset_index(drop=True)
    return sorted_frame.drop(
        columns="_method_rank",
        errors="ignore",
    )


def build_sensitivity_effects(
    rows: pd.DataFrame,
    *,
    target_coverage: float,
    selection_tolerance: float,
    deletion_rate: float,
    baseline_method: str,
    candidate_method: str,
) -> pd.DataFrame:
    selected = rows[
        (rows["target_coverage"] == float(target_coverage))
        & (rows["selection_tolerance"] == float(selection_tolerance))
        & (rows["deletion_rate"] == float(deletion_rate))
        & (rows["method"].isin([baseline_method, candidate_method]))
    ].copy()
    required = {
        "seed",
        "timestamp",
        "method",
        "unique_query_count",
        "label_row_count",
        "observed_label_coverage",
        "full_set_coverage",
        "mean_size",
        "median_size",
        "p90_size",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"effect rows are missing columns: {missing}")
    pivot = selected.pivot_table(
        index=["seed", "timestamp"],
        columns="method",
        values=[
            "unique_query_count",
            "label_row_count",
            "observed_label_coverage",
            "full_set_coverage",
            "mean_size",
            "median_size",
            "p90_size",
        ],
        aggfunc="first",
    )
    if pivot.empty or pivot.isna().any().any():
        raise ValueError("baseline and candidate rows must exist for every timestamp")
    base = pivot.reset_index()
    baseline_size = base[("mean_size", baseline_method)].astype(float)
    candidate_size = base[("mean_size", candidate_method)].astype(float)
    baseline_median = base[("median_size", baseline_method)].astype(float)
    candidate_median = base[("median_size", candidate_method)].astype(float)
    baseline_p90 = base[("p90_size", baseline_method)].astype(float)
    candidate_p90 = base[("p90_size", candidate_method)].astype(float)
    return pd.DataFrame(
        {
            "target_coverage": float(target_coverage),
            "selection_tolerance": float(selection_tolerance),
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "deletion_rate": float(deletion_rate),
            "baseline_method": baseline_method,
            "candidate_method": candidate_method,
            "unique_query_count": base[("unique_query_count", baseline_method)].astype(
                float
            ),
            "label_row_count": base[("label_row_count", baseline_method)].astype(float),
            "mean_size_reduction": baseline_size - candidate_size,
            "relative_mean_size_reduction": np.where(
                baseline_size > 0.0,
                (baseline_size - candidate_size) / baseline_size,
                np.nan,
            ),
            "median_size_reduction": baseline_median - candidate_median,
            "p90_size_reduction": baseline_p90 - candidate_p90,
            "observed_label_coverage_delta": (
                base[("observed_label_coverage", candidate_method)].astype(float)
                - base[("observed_label_coverage", baseline_method)].astype(float)
            ),
            "full_set_coverage_delta": (
                base[("full_set_coverage", candidate_method)].astype(float)
                - base[("full_set_coverage", baseline_method)].astype(float)
            ),
        }
    )


def bootstrap_sensitivity_effects(
    rows: pd.DataFrame,
    *,
    target_coverages: tuple[float, ...],
    selection_tolerances: tuple[float, ...],
    deletion_rate: float,
    comparisons: tuple[tuple[str, str], ...],
    block_length: int,
    iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    specs = [
        ("mean_size_reduction", "unique_query_count", "positive"),
        ("relative_mean_size_reduction", "unique_query_count", "positive"),
        ("median_size_reduction", "unique_query_count", "positive"),
        ("p90_size_reduction", "unique_query_count", "positive"),
        ("observed_label_coverage_delta", "label_row_count", "not_negative"),
        ("full_set_coverage_delta", "unique_query_count", "not_negative"),
    ]
    offset = 0
    for target in target_coverages:
        for tolerance in selection_tolerances:
            for baseline, candidate in comparisons:
                effects = build_sensitivity_effects(
                    rows,
                    target_coverage=target,
                    selection_tolerance=tolerance,
                    deletion_rate=deletion_rate,
                    baseline_method=baseline,
                    candidate_method=candidate,
                )
                for value_column, weight_column, direction in specs:
                    result = _bootstrap_statistic(
                        effects,
                        value_column,
                        weight_column=weight_column,
                        block_length=block_length,
                        iterations=iterations,
                        bootstrap_seed=bootstrap_seed + offset,
                    )
                    offset += 1
                    records.append(
                        {
                            "target_coverage": float(target),
                            "selection_tolerance": float(tolerance),
                            "statistic": (
                                f"{candidate}_{value_column}_vs_{baseline}"
                            ),
                            "deletion_rate": float(deletion_rate),
                            "baseline_method": baseline,
                            "candidate_method": candidate,
                            "observed": result["observed"],
                            "ci95_low": result["ci95"][0],
                            "ci95_high": result["ci95"][1],
                            "pvalue_positive": result["pvalue_positive"],
                            "iterations": int(iterations),
                            "block_length": int(block_length),
                            "seed_count": result["seed_count"],
                            "timestamp_count": result["timestamp_count"],
                            "direction": direction,
                        }
                    )
    return _sort_sensitivity(pd.DataFrame(records))


def build_success_table(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    target_coverages: tuple[float, ...],
    selection_tolerances: tuple[float, ...],
    deletion_rate: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target in target_coverages:
        for tolerance in selection_tolerances:
            adaptive = summary[
                (summary["target_coverage"] == float(target))
                & (summary["selection_tolerance"] == float(tolerance))
                & (summary["deletion_rate"] == float(deletion_rate))
                & (summary["method"] == "adaptive_mass_rolling")
            ]
            if len(adaptive) != 1:
                continue
            row = adaptive.iloc[0]

            def effect_row(value_column: str, baseline: str) -> pd.Series | None:
                statistic = f"adaptive_mass_rolling_{value_column}_vs_{baseline}"
                matches = effects[
                    (effects["target_coverage"] == float(target))
                    & (effects["selection_tolerance"] == float(tolerance))
                    & (effects["statistic"] == statistic)
                ]
                return None if len(matches) != 1 else matches.iloc[0]

            margin_size = effect_row("relative_mean_size_reduction", "margin_rolling")
            margin_label = effect_row("observed_label_coverage_delta", "margin_rolling")
            margin_full = effect_row("full_set_coverage_delta", "margin_rolling")
            rank_size = effect_row("relative_mean_size_reduction", "rank_rolling")
            rank_p90 = effect_row("p90_size_reduction", "rank_rolling")
            label_coverage_ok = (
                float(row["observed_label_coverage_mean"]) >= float(target) - 0.02
            )
            supported_vs_margin = all(
                [
                    label_coverage_ok,
                    margin_size is not None and float(margin_size["ci95_low"]) > 0.0,
                    margin_label is not None and float(margin_label["ci95_low"]) >= -0.02,
                    margin_full is not None and float(margin_full["ci95_low"]) >= -0.05,
                ]
            )
            records.append(
                {
                    "target_coverage": float(target),
                    "selection_tolerance": float(tolerance),
                    "deletion_rate": float(deletion_rate),
                    "adaptive_observed_label_coverage_mean": float(
                        row["observed_label_coverage_mean"]
                    ),
                    "adaptive_full_set_coverage_mean": float(
                        row["full_set_coverage_mean"]
                    ),
                    "adaptive_mean_size_mean": float(row["mean_size_mean"]),
                    "adaptive_median_size_mean": float(row["median_size_mean"]),
                    "adaptive_p90_size_mean": float(row["p90_size_mean"]),
                    "label_coverage_at_least_target_minus_0p02": bool(
                        label_coverage_ok
                    ),
                    "relative_mean_size_reduction_vs_margin_ci_low": (
                        float(margin_size["ci95_low"])
                        if margin_size is not None
                        else np.nan
                    ),
                    "observed_label_delta_vs_margin_ci_low": (
                        float(margin_label["ci95_low"])
                        if margin_label is not None
                        else np.nan
                    ),
                    "full_set_delta_vs_margin_ci_low": (
                        float(margin_full["ci95_low"])
                        if margin_full is not None
                        else np.nan
                    ),
                    "relative_mean_size_reduction_vs_rank_ci_low": (
                        float(rank_size["ci95_low"]) if rank_size is not None else np.nan
                    ),
                    "p90_size_reduction_vs_rank_ci_low": (
                        float(rank_p90["ci95_low"]) if rank_p90 is not None else np.nan
                    ),
                    "supported_vs_margin": bool(supported_vs_margin),
                    "upgrade_over_rank_mean_supported": bool(
                        supported_vs_margin
                        and rank_size is not None
                        and float(rank_size["ci95_low"]) > 0.0
                    ),
                    "tail_size_not_worse_than_rank": bool(
                        rank_p90 is not None and float(rank_p90["ci95_low"]) >= 0.0
                    ),
                }
            )
    return _sort_sensitivity(pd.DataFrame(records))


def build_paper_table(
    summary: pd.DataFrame,
    *,
    deletion_rate: float,
    methods: tuple[str, ...] = SENSITIVITY_METHODS,
) -> pd.DataFrame:
    selected = summary[
        (summary["deletion_rate"] == float(deletion_rate))
        & summary["method"].isin(methods)
    ].copy()
    columns = [
        "target_coverage",
        "selection_tolerance",
        "deletion_rate",
        "method",
        "seed_count",
        "observed_label_coverage_mean",
        "full_set_coverage_mean",
        "partial_answer_recall_mean",
        "mean_size_mean",
        "median_size_mean",
        "p90_size_mean",
        "full_vocabulary_set_rate_mean",
    ]
    return _sort_sensitivity(
        selected[[column for column in columns if column in selected.columns]]
    )


def build_selection_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    required = {
        "target_coverage",
        "selection_tolerance",
        "candidate",
        "k_reg",
        "penalty",
        "selected",
        "observed_label_coverage",
        "full_set_coverage",
        "mean_size",
        "p90_size",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"selection rows are missing columns: {missing}")
    selected = rows[rows["selected"]].copy()
    if selected.empty:
        return pd.DataFrame()
    summary = (
        selected.groupby(
            [
                "target_coverage",
                "selection_tolerance",
                "candidate",
                "k_reg",
                "penalty",
            ],
            as_index=False,
        )
        .agg(
            condition_count=("selected", "count"),
            selection_observed_label_coverage_mean=(
                "observed_label_coverage",
                "mean",
            ),
            selection_full_set_coverage_mean=("full_set_coverage", "mean"),
            selection_mean_size_mean=("mean_size", "mean"),
            selection_p90_size_mean=("p90_size", "mean"),
        )
        .sort_values(
            [
                "target_coverage",
                "selection_tolerance",
                "condition_count",
                "candidate",
            ],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    totals = (
        selected.groupby(["target_coverage", "selection_tolerance"], as_index=False)
        .agg(total_condition_count=("selected", "count"))
    )
    summary = summary.merge(
        totals,
        on=["target_coverage", "selection_tolerance"],
        how="left",
        validate="many_to_one",
    )
    summary["condition_fraction"] = (
        summary["condition_count"] / summary["total_condition_count"]
    )
    summary = summary.drop(columns="total_condition_count")
    return summary


def _prepare_condition_batches(
    *,
    run_root: Path,
    table: Any,
    split: Any,
    seed: int,
    deletion_rate: float,
    embedding_dim: int,
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
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
    try:
        return {
            "tuning": _table_batches(
                model=model,
                values=roles.calibrator_tuning.values,
                relation_count=relation_count,
                batch_size=batch_size,
                device=device,
            ),
            "validation": _table_batches(
                model=model,
                values=roles.selector_validation.values,
                relation_count=relation_count,
                batch_size=batch_size,
                device=device,
            ),
            "final": _table_batches(
                model=model,
                values=roles.final_calibration.values,
                relation_count=relation_count,
                batch_size=batch_size,
                device=device,
            ),
            "test": _table_batches(
                model=model,
                values=split.test.values,
                relation_count=relation_count,
                batch_size=batch_size,
                device=device,
            ),
        }
    finally:
        import torch

        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _select_candidates_for_targets(
    *,
    tuning_batches: list[dict[str, Any]],
    validation_batches: list[dict[str, Any]],
    candidates: tuple[MassCandidate, ...],
    target_coverages: tuple[float, ...],
    selection_tolerances: tuple[float, ...],
    seed: int,
    deletion_rate: float,
    num_entities: int,
) -> tuple[dict[tuple[float, float], MassCandidate], list[dict[str, Any]]]:
    by_name = {candidate.name: candidate for candidate in candidates}
    selected: dict[tuple[float, float], MassCandidate] = {}
    output_rows: list[dict[str, Any]] = []
    for target in target_coverages:
        candidate_rows = []
        for candidate in candidates:
            threshold = _candidate_threshold(tuning_batches, candidate, target)
            candidate_rows.append(
                _evaluate_candidate_on_batches(
                    batches=validation_batches,
                    candidate=candidate,
                    threshold=threshold,
                    seed=seed,
                    deletion_rate=deletion_rate,
                    target_coverage=target,
                    num_entities=num_entities,
                )
            )
        candidate_frame = pd.DataFrame(candidate_rows)
        for tolerance in selection_tolerances:
            decision = select_mass_candidate(
                candidate_frame,
                target_coverage=target,
                coverage_tolerance=tolerance,
            )
            selected_candidate = by_name[decision["candidate"]]
            selected[(float(target), float(tolerance))] = selected_candidate
            for row in candidate_rows:
                output_rows.append(
                    {
                        **row,
                        "target_coverage": float(target),
                        "selection_tolerance": float(tolerance),
                        "selected": row["candidate"] == selected_candidate.name,
                        **{
                            key: value
                            for key, value in decision.items()
                            if key.startswith("selection_")
                        },
                    }
                )
    return selected, output_rows


def _evaluate_condition_grid(
    *,
    batches: dict[str, list[dict[str, Any]]],
    candidates: tuple[MassCandidate, ...],
    selected_candidates: dict[tuple[float, float], MassCandidate],
    target_coverages: tuple[float, ...],
    selection_tolerances: tuple[float, ...],
    seed: int,
    deletion_rate: float,
    rolling_window: int,
    num_entities: int,
) -> list[dict[str, Any]]:
    margin_history = ScoreHistory()
    rank_history = ScoreHistory()
    mass_histories = {candidate.name: ScoreHistory() for candidate in candidates}
    for batch in batches["final"]:
        margin_history.add(
            batch["timestamp"],
            margin_nonconformity(batch["scores"], batch["labels"]),
        )
        rank_history.add(
            batch["timestamp"],
            true_label_ranks(batch["scores"], batch["labels"]),
        )
        for candidate in candidates:
            mass_histories[candidate.name].add(
                batch["timestamp"],
                mass_nonconformity_from_cumulative(
                    batch["cumulative"],
                    batch["positions"],
                    candidate,
                ),
            )

    rows: list[dict[str, Any]] = []
    for batch in batches["test"]:
        timestamp = int(batch["timestamp"])
        threshold_cache: dict[tuple[str, float], tuple[float, int, int]] = {}

        def threshold_for(kind: str, target: float, candidate: MassCandidate | None = None):
            key = (kind if candidate is None else f"{kind}:{candidate.name}", target)
            if key not in threshold_cache:
                if kind == "margin":
                    threshold_cache[key] = _rolling_threshold(
                        margin_history, timestamp, target, rolling_window
                    )
                elif kind == "rank":
                    threshold_cache[key] = _rolling_threshold(
                        rank_history, timestamp, target, rolling_window
                    )
                elif kind == "mass" and candidate is not None:
                    threshold_cache[key] = _rolling_threshold(
                        mass_histories[candidate.name], timestamp, target, rolling_window
                    )
                else:
                    raise ValueError(f"unknown threshold kind: {kind}")
            return threshold_cache[key]

        for target in target_coverages:
            margin_threshold, margin_count, margin_span = threshold_for("margin", target)
            rank_threshold, rank_count, rank_span = threshold_for("rank", target)
            margin_mask = prediction_set_mask(batch["scores"], margin_threshold)
            rank_mask = topk_prediction_mask(batch["scores"], rank_threshold)
            for tolerance in selection_tolerances:
                selected_candidate = selected_candidates[(float(target), float(tolerance))]
                mass_threshold, mass_count, mass_span = threshold_for(
                    "mass", target, selected_candidate
                )
                method_specs = [
                    (
                        "margin_rolling",
                        margin_mask,
                        margin_threshold,
                        margin_count,
                        margin_span,
                        "score_margin",
                    ),
                    (
                        "rank_rolling",
                        rank_mask,
                        rank_threshold,
                        rank_count,
                        rank_span,
                        "rank",
                    ),
                    (
                        "adaptive_mass_rolling",
                        mass_prediction_mask_from_cumulative(
                            batch["order"],
                            batch["cumulative"],
                            candidate=selected_candidate,
                            threshold=mass_threshold,
                        ),
                        mass_threshold,
                        mass_count,
                        mass_span,
                        "mass",
                    ),
                ]
                for method, mask, threshold, pool_count, pool_span, units in method_specs:
                    row = _summarize_mask(
                        seed=seed,
                        deletion_rate=deletion_rate,
                        method=method,
                        timestamp=timestamp,
                        facts=batch["facts"],
                        prediction_sides=batch["prediction_sides"],
                        labels=batch["labels"],
                        ranks=batch["ranks"],
                        mask=mask,
                        threshold=threshold,
                        pool_score_count=pool_count,
                        pool_span_blocks=pool_span,
                        threshold_units=units,
                        num_entities=num_entities,
                    )
                    row["target_coverage"] = float(target)
                    row["selection_tolerance"] = float(tolerance)
                    row["selection_candidate"] = selected_candidate.name
                    row["selection_candidate_k_reg"] = int(selected_candidate.k_reg)
                    row["selection_candidate_penalty"] = float(selected_candidate.penalty)
                    rows.append(row)

        margin_history.add(
            timestamp,
            margin_nonconformity(batch["scores"], batch["labels"]),
        )
        rank_history.add(timestamp, true_label_ranks(batch["scores"], batch["labels"]))
        for candidate in candidates:
            mass_histories[candidate.name].add(
                timestamp,
                mass_nonconformity_from_cumulative(
                    batch["cumulative"],
                    batch["positions"],
                    candidate,
                ),
            )
    return rows


def export_score_adaptive_sensitivity(
    run_root: Path,
    paper_root: Path,
    *,
    data_root: Path | None = None,
    output_name: str = "final_confirmatory",
    seeds: tuple[int, ...] | None = None,
    deletion_rates: tuple[float, ...] | None = None,
    effect_deletion_rate: float | None = None,
    target_coverages: tuple[float, ...] = DEFAULT_TARGET_COVERAGES,
    selection_tolerances: tuple[float, ...] = DEFAULT_SELECTION_TOLERANCES,
    raps_k_values: tuple[int, ...] = DEFAULT_RAPS_K,
    raps_penalties: tuple[float, ...] = DEFAULT_RAPS_LAMBDAS,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    iterations: int = DEFAULT_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED + 31,
    device_name: str = "auto",
) -> dict[str, Any]:
    import torch

    from riskcal_tkg.config import load_config
    from riskcal_tkg.data import load_configured_table

    target_coverages = _validate_probability_grid("target coverages", target_coverages)
    selection_tolerances = _validate_tolerances(selection_tolerances)
    run_root = run_root.resolve()
    paper_root = paper_root.resolve()
    config = load_config(run_root / "config.resolved.yaml")
    if data_root is not None:
        config = type(config)(**{**config.__dict__, "data_path": data_root})
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

    candidates = build_mass_candidates(
        k_values=raps_k_values,
        penalties=raps_penalties,
    )
    selected_seeds = tuple(
        int(seed) for seed in _filter_numbers(config.seeds or (config.seed,), seeds)
    )
    selected_rates = tuple(
        float(rate) for rate in _filter_numbers(config.deletion_rates, deletion_rates)
    )
    if not selected_seeds:
        raise ValueError("no seeds selected")
    if not selected_rates:
        raise ValueError("no deletion rates selected")
    if effect_deletion_rate is None:
        effect_deletion_rate = float(max(selected_rates))

    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        for deletion_rate in selected_rates:
            print(
                "[score-adaptive-sensitivity] scoring "
                f"seed={seed} deletion_rate={deletion_rate:g} "
                f"targets={target_coverages} tolerances={selection_tolerances}",
                flush=True,
            )
            batches = _prepare_condition_batches(
                run_root=run_root,
                table=table,
                split=split,
                seed=seed,
                deletion_rate=deletion_rate,
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                device=device,
            )
            selected_candidates, current_selection_rows = _select_candidates_for_targets(
                tuning_batches=batches["tuning"],
                validation_batches=batches["validation"],
                candidates=candidates,
                target_coverages=target_coverages,
                selection_tolerances=selection_tolerances,
                seed=seed,
                deletion_rate=deletion_rate,
                num_entities=len(table.entity_to_id),
            )
            current_rows = _evaluate_condition_grid(
                batches=batches,
                candidates=candidates,
                selected_candidates=selected_candidates,
                target_coverages=target_coverages,
                selection_tolerances=selection_tolerances,
                seed=seed,
                deletion_rate=deletion_rate,
                rolling_window=config.rolling_window,
                num_entities=len(table.entity_to_id),
            )
            rows.extend(current_rows)
            selection_rows.extend(current_selection_rows)
            print(
                "[score-adaptive-sensitivity] completed "
                f"seed={seed} deletion_rate={deletion_rate:g}; "
                f"test_rows={len(current_rows)} selection_rows={len(current_selection_rows)}",
                flush=True,
            )

    data_dir = paper_root / "data" / output_name
    row_frame = pd.DataFrame(rows)
    by_seed = summarize_sensitivity_rows(row_frame)
    summary = aggregate_sensitivity_summary(by_seed)
    effects = bootstrap_sensitivity_effects(
        row_frame,
        target_coverages=target_coverages,
        selection_tolerances=selection_tolerances,
        deletion_rate=float(effect_deletion_rate),
        comparisons=(
            ("margin_rolling", "adaptive_mass_rolling"),
            ("rank_rolling", "adaptive_mass_rolling"),
        ),
        block_length=block_length,
        iterations=iterations,
        bootstrap_seed=bootstrap_seed,
    )
    print(
        "[score-adaptive-sensitivity] bootstrap completed "
        f"for deletion_rate={float(effect_deletion_rate):g}",
        flush=True,
    )
    success_table = build_success_table(
        summary,
        effects,
        target_coverages=target_coverages,
        selection_tolerances=selection_tolerances,
        deletion_rate=float(effect_deletion_rate),
    )
    paper_table = build_paper_table(summary, deletion_rate=float(effect_deletion_rate))
    selection_frame = pd.DataFrame(selection_rows)
    selection_summary = build_selection_summary(selection_frame)
    output_paths = {
        "score_adaptive_sensitivity_by_timestamp.csv": data_dir
        / "score_adaptive_sensitivity_by_timestamp.csv",
        "score_adaptive_sensitivity_by_seed.csv": data_dir
        / "score_adaptive_sensitivity_by_seed.csv",
        "score_adaptive_sensitivity_summary.csv": data_dir
        / "score_adaptive_sensitivity_summary.csv",
        "score_adaptive_sensitivity_effects.csv": data_dir
        / "score_adaptive_sensitivity_effects.csv",
        "score_adaptive_sensitivity_paper_table.csv": data_dir
        / "score_adaptive_sensitivity_paper_table.csv",
        "score_adaptive_sensitivity_selection_by_condition.csv": data_dir
        / "score_adaptive_sensitivity_selection_by_condition.csv",
        "score_adaptive_sensitivity_selection_summary.csv": data_dir
        / "score_adaptive_sensitivity_selection_summary.csv",
        "score_adaptive_sensitivity_success_table.csv": data_dir
        / "score_adaptive_sensitivity_success_table.csv",
    }
    _write_csv(_sort_sensitivity(row_frame), output_paths["score_adaptive_sensitivity_by_timestamp.csv"])
    _write_csv(by_seed, output_paths["score_adaptive_sensitivity_by_seed.csv"])
    _write_csv(summary, output_paths["score_adaptive_sensitivity_summary.csv"])
    _write_csv(effects, output_paths["score_adaptive_sensitivity_effects.csv"])
    _write_csv(paper_table, output_paths["score_adaptive_sensitivity_paper_table.csv"])
    _write_csv(selection_frame, output_paths["score_adaptive_sensitivity_selection_by_condition.csv"])
    _write_csv(selection_summary, output_paths["score_adaptive_sensitivity_selection_summary.csv"])
    _write_csv(success_table, output_paths["score_adaptive_sensitivity_success_table.csv"])
    outputs = {name: sha256_file(path) for name, path in output_paths.items()}
    manifest = {
        "block_length": int(block_length),
        "bootstrap_seed": int(bootstrap_seed),
        "candidate_count": len(candidates),
        "condition_count": len(selected_seeds) * len(selected_rates),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "definition": {
            "target_coverage": (
                "Nominal conformal target used for margin, rank, and mass thresholds."
            ),
            "selection_tolerance": (
                "Allowed validation observed-label coverage shortfall below the target "
                "when selecting the smallest APS/RAPS candidate."
            ),
            "adaptive_mass_rolling": (
                "APS/RAPS candidate selected on calibration-only validation batches, "
                "then evaluated with a rolling threshold over strictly past scores."
            ),
        },
        "deletion_rates": list(selected_rates),
        "device": str(device),
        "effect_deletion_rate": float(effect_deletion_rate),
        "iterations": int(iterations),
        "methods": list(SENSITIVITY_METHODS),
        "output_name": output_name,
        "outputs": outputs,
        "run_root": str(run_root),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seeds": list(selected_seeds),
        "selection_tolerances": list(selection_tolerances),
        "target_coverages": list(target_coverages),
    }
    manifest_path = data_dir / "score_adaptive_sensitivity_manifest.json"
    _write_json(manifest, manifest_path)
    manifest["outputs"]["score_adaptive_sensitivity_manifest.json"] = sha256_file(
        manifest_path
    )
    _write_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-name", default="final_confirmatory")
    parser.add_argument("--seeds", help="Optional comma-separated seed subset.")
    parser.add_argument(
        "--deletion-rates",
        help="Optional comma-separated deletion-rate subset.",
    )
    parser.add_argument("--effect-deletion-rate", type=float)
    parser.add_argument(
        "--target-coverages",
        default=",".join(f"{value:g}" for value in DEFAULT_TARGET_COVERAGES),
    )
    parser.add_argument(
        "--selection-tolerances",
        default=",".join(f"{value:g}" for value in DEFAULT_SELECTION_TOLERANCES),
    )
    parser.add_argument(
        "--raps-k-values",
        default=",".join(str(value) for value in DEFAULT_RAPS_K),
    )
    parser.add_argument(
        "--raps-penalties",
        default=",".join(f"{value:g}" for value in DEFAULT_RAPS_LAMBDAS),
    )
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED + 31)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    export_score_adaptive_sensitivity(
        args.run_root,
        args.paper_root,
        data_root=args.data_root,
        output_name=args.output_name,
        seeds=_parse_int_tuple(args.seeds),
        deletion_rates=_parse_float_tuple(args.deletion_rates),
        effect_deletion_rate=args.effect_deletion_rate,
        target_coverages=_parse_float_tuple(args.target_coverages)
        or DEFAULT_TARGET_COVERAGES,
        selection_tolerances=_parse_float_tuple(args.selection_tolerances)
        or DEFAULT_SELECTION_TOLERANCES,
        raps_k_values=_parse_int_tuple(args.raps_k_values) or DEFAULT_RAPS_K,
        raps_penalties=_parse_float_tuple(args.raps_penalties) or DEFAULT_RAPS_LAMBDAS,
        block_length=args.block_length,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
