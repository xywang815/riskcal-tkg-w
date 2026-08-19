"""Export score-adaptive APS/RAPS shortlist diagnostics.

This exporter reuses a completed confirmatory run and its checkpoints.  It does
not retrain the scorer.  APS/RAPS candidates are selected on calibration-only
validation batches, then evaluated prequentially on the test stream.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
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
    add_inverse_relations,
    split_calibration_roles,
    temporal_split,
)
from scripts.export_query_level_diagnostics import sha256_file  # noqa: E402
from scripts.export_shortlist_calibration import (  # noqa: E402
    _summarize_mask,
    aggregate_shortlist_summary,
    summarize_shortlist_rows,
    topk_prediction_mask,
    true_label_ranks,
)
from scripts.export_timestamp_block_bootstrap import _bootstrap_statistic  # noqa: E402
from scripts.export_window_ablation import (  # noqa: E402
    ScoreHistory,
    _load_condition_model,
    _score_all_objects,
)


DEFAULT_BOOTSTRAP_SEED = 20260818
DEFAULT_BLOCK_LENGTH = 7
DEFAULT_ITERATIONS = 20_000
DEFAULT_RAPS_K = (50, 100, 250, 500, 1000)
DEFAULT_RAPS_LAMBDAS = (0.0001, 0.0005, 0.001, 0.002)
METHOD_ORDER = [
    "margin_rolling",
    "rank_rolling",
    "aps_rolling",
    "adaptive_mass_rolling",
]


@dataclass(frozen=True)
class MassCandidate:
    name: str
    k_reg: int
    penalty: float


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


def _candidate_name(k_reg: int, penalty: float) -> str:
    if penalty == 0.0:
        return "aps"
    label = f"{penalty:g}".replace(".", "p").replace("-", "m")
    return f"raps_k{k_reg}_lam{label}"


def build_mass_candidates(
    *,
    k_values: tuple[int, ...] = DEFAULT_RAPS_K,
    penalties: tuple[float, ...] = DEFAULT_RAPS_LAMBDAS,
) -> tuple[MassCandidate, ...]:
    if any(value <= 0 for value in k_values):
        raise ValueError("RAPS k values must be positive")
    if any(value < 0.0 or not math.isfinite(value) for value in penalties):
        raise ValueError("RAPS penalties must be finite and nonnegative")
    candidates = [MassCandidate("aps", 0, 0.0)]
    for k_reg in sorted(set(int(value) for value in k_values)):
        for penalty in sorted(set(float(value) for value in penalties)):
            if penalty > 0.0:
                candidates.append(
                    MassCandidate(_candidate_name(k_reg, penalty), k_reg, penalty)
                )
    return tuple(candidates)


def _stable_sorted_softmax(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("scores must have shape (n, classes)")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    order = np.argsort(-values, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(values, order, axis=1)
    shifted = sorted_scores - sorted_scores[:, :1]
    exp_values = np.exp(np.clip(shifted, -745.0, 0.0))
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probabilities, axis=1)
    return order, probabilities, cumulative


def true_positions_from_order(order: np.ndarray, labels: np.ndarray) -> np.ndarray:
    order_array = np.asarray(order, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if order_array.ndim != 2 or label_array.shape != (len(order_array),):
        raise ValueError("order and labels have incompatible shapes")
    if label_array.min() < 0 or label_array.max() >= order_array.shape[1]:
        raise ValueError("labels contain an out-of-range entity ID")
    inverse = np.empty_like(order_array)
    inverse[np.arange(len(order_array))[:, None], order_array] = np.arange(
        order_array.shape[1]
    )
    return inverse[np.arange(len(order_array)), label_array]


def mass_nonconformity_from_cumulative(
    cumulative: np.ndarray,
    true_positions: np.ndarray,
    candidate: MassCandidate,
) -> np.ndarray:
    positions = np.asarray(true_positions, dtype=np.int64).reshape(-1)
    if cumulative.ndim != 2 or positions.shape != (len(cumulative),):
        raise ValueError("cumulative probabilities and positions have incompatible shapes")
    if len(positions) and (positions.min() < 0 or positions.max() >= cumulative.shape[1]):
        raise ValueError("true positions are out of range")
    ranks = positions.astype(float) + 1.0
    penalties = candidate.penalty * np.maximum(ranks - float(candidate.k_reg), 0.0)
    return cumulative[np.arange(len(cumulative)), positions] + penalties


def mass_prediction_mask_from_cumulative(
    order: np.ndarray,
    cumulative: np.ndarray,
    *,
    candidate: MassCandidate,
    threshold: float,
) -> np.ndarray:
    if order.ndim != 2 or cumulative.shape != order.shape:
        raise ValueError("order and cumulative probabilities have incompatible shapes")
    if not math.isfinite(float(threshold)) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")
    ranks = np.arange(1, order.shape[1] + 1, dtype=float)
    scores = cumulative + candidate.penalty * np.maximum(
        ranks.reshape(1, -1) - float(candidate.k_reg),
        0.0,
    )
    sorted_mask = scores <= float(threshold) + 1e-12
    mask = np.zeros(order.shape, dtype=bool)
    mask[np.arange(len(order))[:, None], order] = sorted_mask
    return mask


def mass_nonconformity(scores: np.ndarray, labels: np.ndarray, candidate: MassCandidate) -> np.ndarray:
    order, _, cumulative = _stable_sorted_softmax(scores)
    positions = true_positions_from_order(order, labels)
    return mass_nonconformity_from_cumulative(cumulative, positions, candidate)


def mass_prediction_mask(
    scores: np.ndarray,
    *,
    candidate: MassCandidate,
    threshold: float,
) -> np.ndarray:
    order, _, cumulative = _stable_sorted_softmax(scores)
    return mass_prediction_mask_from_cumulative(
        order,
        cumulative,
        candidate=candidate,
        threshold=threshold,
    )


def _method_sort(frame: pd.DataFrame, extra_columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result = frame.copy()
    result["_method_rank"] = result["method"].map(rank).fillna(len(rank))
    columns = [
        "deletion_rate",
        *(extra_columns or []),
        "_method_rank",
        "method",
    ]
    present = [column for column in columns if column in result.columns]
    return result.sort_values(present, kind="stable").drop(
        columns="_method_rank"
    ).reset_index(drop=True)


def _table_batches(
    *,
    model: Any,
    values: np.ndarray,
    relation_count: int,
    batch_size: int,
    device: Any,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    timestamps = np.unique(values[:, 3])
    for timestamp in timestamps:
        raw_facts = values[values[:, 3] == timestamp]
        facts = add_inverse_relations(raw_facts, relation_count)
        prediction_sides = np.asarray(
            ["object"] * len(raw_facts) + ["subject"] * len(raw_facts)
        )
        scores = _score_all_objects(model, facts, batch_size, device)
        labels = facts[:, 2].copy()
        ranks = true_label_ranks(scores, labels)
        order, _, cumulative = _stable_sorted_softmax(scores)
        positions = true_positions_from_order(order, labels)
        batches.append(
            {
                "timestamp": int(timestamp),
                "facts": facts,
                "prediction_sides": prediction_sides,
                "scores": scores,
                "labels": labels,
                "ranks": ranks,
                "order": order,
                "cumulative": cumulative,
                "positions": positions,
            }
        )
    return batches


def _candidate_threshold(
    batches: list[dict[str, Any]],
    candidate: MassCandidate,
    target_coverage: float,
) -> float:
    scores = [
        mass_nonconformity_from_cumulative(
            batch["cumulative"],
            batch["positions"],
            candidate,
        )
        for batch in batches
    ]
    return finite_sample_quantile(
        np.concatenate(scores),
        alpha=1.0 - target_coverage,
    )


def _evaluate_candidate_on_batches(
    *,
    batches: list[dict[str, Any]],
    candidate: MassCandidate,
    threshold: float,
    seed: int,
    deletion_rate: float,
    target_coverage: float,
    num_entities: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        mask = mass_prediction_mask_from_cumulative(
            batch["order"],
            batch["cumulative"],
            candidate=candidate,
            threshold=threshold,
        )
        rows.append(
            _summarize_mask(
                seed=seed,
                deletion_rate=deletion_rate,
                method=candidate.name,
                timestamp=batch["timestamp"],
                facts=batch["facts"],
                prediction_sides=batch["prediction_sides"],
                labels=batch["labels"],
                ranks=batch["ranks"],
                mask=mask,
                threshold=threshold,
                pool_score_count=0,
                pool_span_blocks=0,
                threshold_units="mass",
                num_entities=num_entities,
            )
        )
    summary = summarize_shortlist_rows(pd.DataFrame(rows), target_coverage).iloc[0]
    return {
        "seed": int(seed),
        "deletion_rate": float(deletion_rate),
        "candidate": candidate.name,
        "k_reg": int(candidate.k_reg),
        "penalty": float(candidate.penalty),
        "threshold": float(threshold),
        "observed_label_coverage": float(summary["observed_label_coverage"]),
        "full_set_coverage": float(summary["full_set_coverage"]),
        "partial_answer_recall": float(summary["partial_answer_recall"]),
        "mean_size": float(summary["mean_size"]),
        "median_size": float(summary["median_size"]),
        "p90_size": float(summary["p90_size"]),
        "full_vocabulary_set_rate": float(summary["full_vocabulary_set_rate"]),
    }


def select_mass_candidate(
    rows: pd.DataFrame,
    *,
    target_coverage: float,
    coverage_tolerance: float,
) -> dict[str, Any]:
    required = {
        "candidate",
        "observed_label_coverage",
        "full_set_coverage",
        "mean_size",
        "p90_size",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"selection rows are missing columns: {missing}")
    if rows.empty:
        raise ValueError("selection rows are empty")
    feasible = rows[
        rows["observed_label_coverage"] >= float(target_coverage) - coverage_tolerance
    ].copy()
    if feasible.empty:
        ordered = rows.sort_values(
            ["observed_label_coverage", "mean_size", "full_set_coverage"],
            ascending=[False, True, False],
            kind="stable",
        )
        chosen = ordered.iloc[0]
        feasible_flag = False
    else:
        ordered = feasible.sort_values(
            ["mean_size", "p90_size", "full_set_coverage", "observed_label_coverage"],
            ascending=[True, True, False, False],
            kind="stable",
        )
        chosen = ordered.iloc[0]
        feasible_flag = True
    return {
        "candidate": str(chosen["candidate"]),
        "selection_feasible": bool(feasible_flag),
        "selection_target_coverage": float(target_coverage),
        "selection_coverage_tolerance": float(coverage_tolerance),
        "selection_observed_label_coverage": float(chosen["observed_label_coverage"]),
        "selection_full_set_coverage": float(chosen["full_set_coverage"]),
        "selection_mean_size": float(chosen["mean_size"]),
        "selection_p90_size": float(chosen["p90_size"]),
    }


def _selection_summary(rows: pd.DataFrame) -> pd.DataFrame:
    selected = rows[rows["selected"]].copy()
    if selected.empty:
        return pd.DataFrame()
    summary = (
        selected.groupby(["candidate", "k_reg", "penalty"], as_index=False)
        .agg(
            condition_count=("seed", "count"),
            selection_observed_label_coverage_mean=(
                "observed_label_coverage",
                "mean",
            ),
            selection_full_set_coverage_mean=("full_set_coverage", "mean"),
            selection_mean_size_mean=("mean_size", "mean"),
            selection_p90_size_mean=("p90_size", "mean"),
        )
        .sort_values(["condition_count", "candidate"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["condition_fraction"] = summary["condition_count"] / int(len(selected))
    return summary


def _filter_numbers(
    values: tuple[int | float, ...],
    requested: tuple[int | float, ...] | None,
) -> tuple[int | float, ...]:
    if requested is None:
        return values
    wanted = {float(value) for value in requested}
    return tuple(value for value in values if float(value) in wanted)


def _rolling_threshold(history: ScoreHistory, timestamp: int, target_coverage: float, window: int) -> tuple[float, int, int]:
    values, timestamps = history.values_before(timestamp, max_count=window)
    return (
        finite_sample_quantile(values, alpha=1.0 - target_coverage),
        int(len(values)),
        int(timestamp - timestamps.min()),
    )


def _evaluate_condition(
    *,
    run_root: Path,
    table: Any,
    split: Any,
    seed: int,
    deletion_rate: float,
    embedding_dim: int,
    batch_size: int,
    rolling_window: int,
    target_coverage: float,
    selection_tolerance: float,
    candidates: tuple[MassCandidate, ...],
    device: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

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
    tuning_batches = _table_batches(
        model=model,
        values=roles.calibrator_tuning.values,
        relation_count=relation_count,
        batch_size=batch_size,
        device=device,
    )
    validation_batches = _table_batches(
        model=model,
        values=roles.selector_validation.values,
        relation_count=relation_count,
        batch_size=batch_size,
        device=device,
    )
    selection_rows: list[dict[str, Any]] = []
    by_name = {candidate.name: candidate for candidate in candidates}
    for candidate in candidates:
        threshold = _candidate_threshold(tuning_batches, candidate, target_coverage)
        selection_rows.append(
            _evaluate_candidate_on_batches(
                batches=validation_batches,
                candidate=candidate,
                threshold=threshold,
                seed=seed,
                deletion_rate=deletion_rate,
                target_coverage=target_coverage,
                num_entities=len(table.entity_to_id),
            )
        )
    decision = select_mass_candidate(
        pd.DataFrame(selection_rows),
        target_coverage=target_coverage,
        coverage_tolerance=selection_tolerance,
    )
    selected_candidate = by_name[decision["candidate"]]
    selection_rows = [
        {
            **row,
            "selected": row["candidate"] == selected_candidate.name,
            **{
                key: value
                for key, value in decision.items()
                if key.startswith("selection_")
            },
        }
        for row in selection_rows
    ]

    margin_history = ScoreHistory()
    rank_history = ScoreHistory()
    mass_histories = {candidate.name: ScoreHistory() for candidate in candidates}
    final_batches = _table_batches(
        model=model,
        values=roles.final_calibration.values,
        relation_count=relation_count,
        batch_size=batch_size,
        device=device,
    )
    for batch in final_batches:
        margin_history.add(batch["timestamp"], margin_nonconformity(batch["scores"], batch["labels"]))
        rank_history.add(batch["timestamp"], true_label_ranks(batch["scores"], batch["labels"]))
        for candidate in candidates:
            mass_histories[candidate.name].add(
                batch["timestamp"],
                mass_nonconformity_from_cumulative(
                    batch["cumulative"],
                    batch["positions"],
                    candidate,
                ),
            )

    test_batches = _table_batches(
        model=model,
        values=split.test.values,
        relation_count=relation_count,
        batch_size=batch_size,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    aps = by_name["aps"]
    for batch in test_batches:
        timestamp = int(batch["timestamp"])
        margin_threshold, margin_count, margin_span = _rolling_threshold(
            margin_history,
            timestamp,
            target_coverage,
            rolling_window,
        )
        rank_threshold, rank_count, rank_span = _rolling_threshold(
            rank_history,
            timestamp,
            target_coverage,
            rolling_window,
        )
        aps_threshold, aps_count, aps_span = _rolling_threshold(
            mass_histories[aps.name],
            timestamp,
            target_coverage,
            rolling_window,
        )
        selected_threshold, selected_count, selected_span = _rolling_threshold(
            mass_histories[selected_candidate.name],
            timestamp,
            target_coverage,
            rolling_window,
        )
        method_specs = [
            (
                "margin_rolling",
                prediction_set_mask(batch["scores"], margin_threshold),
                margin_threshold,
                margin_count,
                margin_span,
                "score_margin",
                None,
            ),
            (
                "rank_rolling",
                topk_prediction_mask(batch["scores"], rank_threshold),
                rank_threshold,
                rank_count,
                rank_span,
                "rank",
                None,
            ),
            (
                "aps_rolling",
                mass_prediction_mask_from_cumulative(
                    batch["order"],
                    batch["cumulative"],
                    candidate=aps,
                    threshold=aps_threshold,
                ),
                aps_threshold,
                aps_count,
                aps_span,
                "mass",
                aps,
            ),
            (
                "adaptive_mass_rolling",
                mass_prediction_mask_from_cumulative(
                    batch["order"],
                    batch["cumulative"],
                    candidate=selected_candidate,
                    threshold=selected_threshold,
                ),
                selected_threshold,
                selected_count,
                selected_span,
                "mass",
                selected_candidate,
            ),
        ]
        for method, mask, threshold, pool_count, pool_span, units, candidate in method_specs:
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
                num_entities=len(table.entity_to_id),
            )
            if candidate is not None:
                row["mass_candidate"] = candidate.name
                row["mass_candidate_k_reg"] = int(candidate.k_reg)
                row["mass_candidate_penalty"] = float(candidate.penalty)
            if method == "adaptive_mass_rolling":
                row["selection_feasible"] = decision["selection_feasible"]
                row["selection_candidate"] = selected_candidate.name
                row["selection_observed_label_coverage"] = decision[
                    "selection_observed_label_coverage"
                ]
                row["selection_mean_size"] = decision["selection_mean_size"]
            rows.append(row)

        margin_history.add(timestamp, margin_nonconformity(batch["scores"], batch["labels"]))
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

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, selection_rows


def build_effect_frames(
    rows: pd.DataFrame,
    *,
    deletion_rate: float,
    baseline_method: str,
    candidate_method: str,
) -> pd.DataFrame:
    selected = rows[
        (rows["deletion_rate"] == float(deletion_rate))
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
            "p90_size",
        ],
        aggfunc="first",
    )
    if pivot.isna().any().any():
        raise ValueError("baseline and candidate rows must exist for every timestamp")
    base = pivot.reset_index()
    baseline_size = base[("mean_size", baseline_method)].astype(float)
    candidate_size = base[("mean_size", candidate_method)].astype(float)
    baseline_p90 = base[("p90_size", baseline_method)].astype(float)
    candidate_p90 = base[("p90_size", candidate_method)].astype(float)
    return pd.DataFrame(
        {
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "deletion_rate": float(deletion_rate),
            "baseline_method": baseline_method,
            "candidate_method": candidate_method,
            "unique_query_count": base[("unique_query_count", baseline_method)].astype(float),
            "label_row_count": base[("label_row_count", baseline_method)].astype(float),
            "mean_size_reduction": baseline_size - candidate_size,
            "relative_mean_size_reduction": np.where(
                baseline_size > 0,
                (baseline_size - candidate_size) / baseline_size,
                np.nan,
            ),
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


def bootstrap_effects(
    rows: pd.DataFrame,
    *,
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
        ("p90_size_reduction", "unique_query_count", "positive"),
        ("observed_label_coverage_delta", "label_row_count", "not_negative"),
        ("full_set_coverage_delta", "unique_query_count", "not_negative"),
    ]
    offset = 0
    for baseline, candidate in comparisons:
        effects = build_effect_frames(
            rows,
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
                    "statistic": f"{candidate}_{value_column}_vs_{baseline}",
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
    return pd.DataFrame(records)


def _success_gate(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    target_coverage: float,
    deletion_rate: float,
) -> dict[str, Any]:
    selected = summary[
        (summary["deletion_rate"] == float(deletion_rate))
        & (summary["method"] == "adaptive_mass_rolling")
    ]
    if len(selected) != 1:
        return {
            "status": "not_evaluated",
            "reason": "adaptive_mass_rolling summary row missing",
            "supported_vs_margin": False,
            "upgrade_over_rank_supported": False,
        }
    row = selected.iloc[0]

    def effect_row(value_column: str, baseline: str) -> pd.Series | None:
        statistic = f"adaptive_mass_rolling_{value_column}_vs_{baseline}"
        matches = effects[effects["statistic"] == statistic]
        return None if len(matches) != 1 else matches.iloc[0]

    margin_size = effect_row("relative_mean_size_reduction", "margin_rolling")
    margin_label = effect_row("observed_label_coverage_delta", "margin_rolling")
    margin_full = effect_row("full_set_coverage_delta", "margin_rolling")
    rank_size = effect_row("relative_mean_size_reduction", "rank_rolling")
    label_coverage_ok = (
        float(row["observed_label_coverage_mean"]) >= float(target_coverage) - 0.02
    )
    supported_vs_margin = all(
        [
            label_coverage_ok,
            margin_size is not None and float(margin_size["ci95_low"]) > 0.0,
            margin_label is not None and float(margin_label["ci95_low"]) >= -0.02,
            margin_full is not None and float(margin_full["ci95_low"]) >= -0.05,
        ]
    )
    upgrade_over_rank_supported = bool(
        supported_vs_margin
        and rank_size is not None
        and float(rank_size["ci95_low"]) > 0.0
    )
    return {
        "status": "evaluated",
        "supported_vs_margin": bool(supported_vs_margin),
        "upgrade_over_rank_supported": bool(upgrade_over_rank_supported),
        "target_coverage": float(target_coverage),
        "effect_deletion_rate": float(deletion_rate),
        "adaptive_observed_label_coverage_mean": float(
            row["observed_label_coverage_mean"]
        ),
        "adaptive_full_set_coverage_mean": float(row["full_set_coverage_mean"]),
        "adaptive_mean_size_mean": float(row["mean_size_mean"]),
        "criteria": {
            "label_coverage_at_least_target_minus_0p02": bool(label_coverage_ok),
            "relative_mean_size_reduction_vs_margin_ci_low_positive": bool(
                margin_size is not None and float(margin_size["ci95_low"]) > 0.0
            ),
            "label_coverage_delta_vs_margin_ci_low_at_least_minus_0p02": bool(
                margin_label is not None and float(margin_label["ci95_low"]) >= -0.02
            ),
            "full_set_coverage_delta_vs_margin_ci_low_at_least_minus_0p05": bool(
                margin_full is not None and float(margin_full["ci95_low"]) >= -0.05
            ),
            "relative_mean_size_reduction_vs_rank_ci_low_positive": bool(
                rank_size is not None and float(rank_size["ci95_low"]) > 0.0
            ),
        },
    }


def build_paper_table(
    summary: pd.DataFrame,
    *,
    deletion_rate: float,
    methods: tuple[str, ...] = (
        "margin_rolling",
        "rank_rolling",
        "aps_rolling",
        "adaptive_mass_rolling",
    ),
) -> pd.DataFrame:
    selected = summary[
        (summary["deletion_rate"] == float(deletion_rate))
        & summary["method"].isin(methods)
    ].copy()
    columns = [
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
        "positive_label_undercoverage_mean",
        "positive_full_set_undercoverage_mean",
    ]
    return _method_sort(
        selected[[column for column in columns if column in selected.columns]]
    )


def export_score_adaptive_shortlist(
    run_root: Path,
    paper_root: Path,
    *,
    data_root: Path | None = None,
    output_name: str = "final_confirmatory",
    seeds: tuple[int, ...] | None = None,
    deletion_rates: tuple[float, ...] | None = None,
    effect_deletion_rate: float | None = None,
    raps_k_values: tuple[int, ...] = DEFAULT_RAPS_K,
    raps_penalties: tuple[float, ...] = DEFAULT_RAPS_LAMBDAS,
    selection_tolerance: float = 0.02,
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

    candidates = build_mass_candidates(
        k_values=raps_k_values,
        penalties=raps_penalties,
    )
    selected_seeds = tuple(int(seed) for seed in _filter_numbers(config.seeds or (config.seed,), seeds))
    selected_rates = tuple(
        float(rate) for rate in _filter_numbers(config.deletion_rates, deletion_rates)
    )
    if not selected_seeds:
        raise ValueError("no seeds selected")
    if not selected_rates:
        raise ValueError("no deletion rates selected")

    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        for deletion_rate in selected_rates:
            print(
                "[score-adaptive] scoring "
                f"seed={seed} deletion_rate={deletion_rate:g} "
                f"with {len(candidates)} APS/RAPS candidates",
                flush=True,
            )
            current_rows, current_selection = _evaluate_condition(
                run_root=run_root,
                table=table,
                split=split,
                seed=seed,
                deletion_rate=deletion_rate,
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                rolling_window=config.rolling_window,
                target_coverage=config.target_coverage,
                selection_tolerance=selection_tolerance,
                candidates=candidates,
                device=device,
            )
            rows.extend(current_rows)
            selection_rows.extend(current_selection)
            print(
                "[score-adaptive] completed "
                f"seed={seed} deletion_rate={deletion_rate:g}; "
                f"test_rows={len(current_rows)} selection_rows={len(current_selection)}",
                flush=True,
            )

    data_dir = paper_root / "data" / output_name
    row_frame = pd.DataFrame(rows)
    by_seed = summarize_shortlist_rows(row_frame, config.target_coverage)
    summary = aggregate_shortlist_summary(by_seed)
    if effect_deletion_rate is None:
        effect_deletion_rate = float(max(selected_rates))
    effects = bootstrap_effects(
        row_frame,
        deletion_rate=float(effect_deletion_rate),
        comparisons=(
            ("margin_rolling", "aps_rolling"),
            ("margin_rolling", "adaptive_mass_rolling"),
            ("rank_rolling", "adaptive_mass_rolling"),
        ),
        block_length=block_length,
        iterations=iterations,
        bootstrap_seed=bootstrap_seed,
    )
    print(
        "[score-adaptive] bootstrap completed "
        f"for deletion_rate={float(effect_deletion_rate):g}",
        flush=True,
    )
    selection_frame = pd.DataFrame(selection_rows)
    selection_summary = _selection_summary(selection_frame)
    paper_table = build_paper_table(
        summary,
        deletion_rate=float(effect_deletion_rate),
    )
    success_gate = _success_gate(
        summary,
        effects,
        target_coverage=config.target_coverage,
        deletion_rate=float(effect_deletion_rate),
    )

    output_paths = {
        "score_adaptive_shortlist_by_timestamp.csv": data_dir
        / "score_adaptive_shortlist_by_timestamp.csv",
        "score_adaptive_shortlist_by_seed.csv": data_dir
        / "score_adaptive_shortlist_by_seed.csv",
        "score_adaptive_shortlist_summary.csv": data_dir
        / "score_adaptive_shortlist_summary.csv",
        "score_adaptive_shortlist_paper_table.csv": data_dir
        / "score_adaptive_shortlist_paper_table.csv",
        "score_adaptive_shortlist_effects.csv": data_dir
        / "score_adaptive_shortlist_effects.csv",
        "score_adaptive_shortlist_selection_by_condition.csv": data_dir
        / "score_adaptive_shortlist_selection_by_condition.csv",
        "score_adaptive_shortlist_selection_summary.csv": data_dir
        / "score_adaptive_shortlist_selection_summary.csv",
        "score_adaptive_shortlist_success_gate.json": data_dir
        / "score_adaptive_shortlist_success_gate.json",
    }
    _write_csv(_method_sort(row_frame), output_paths["score_adaptive_shortlist_by_timestamp.csv"])
    _write_csv(by_seed, output_paths["score_adaptive_shortlist_by_seed.csv"])
    _write_csv(summary, output_paths["score_adaptive_shortlist_summary.csv"])
    _write_csv(paper_table, output_paths["score_adaptive_shortlist_paper_table.csv"])
    _write_csv(effects, output_paths["score_adaptive_shortlist_effects.csv"])
    _write_csv(selection_frame, output_paths["score_adaptive_shortlist_selection_by_condition.csv"])
    _write_csv(selection_summary, output_paths["score_adaptive_shortlist_selection_summary.csv"])
    _write_json(success_gate, output_paths["score_adaptive_shortlist_success_gate.json"])
    outputs = {name: sha256_file(path) for name, path in output_paths.items()}
    manifest = {
        "block_length": int(block_length),
        "bootstrap_seed": int(bootstrap_seed),
        "candidate_count": len(candidates),
        "candidates": [
            {
                "name": candidate.name,
                "k_reg": int(candidate.k_reg),
                "penalty": float(candidate.penalty),
            }
            for candidate in candidates
        ],
        "condition_count": len(selected_seeds) * len(selected_rates),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "definition": {
            "aps_rolling": (
                "Rolling conformal set using cumulative softmax probability mass "
                "as the nonconformity score."
            ),
            "adaptive_mass_rolling": (
                "APS/RAPS candidate selected on calibration-only validation batches, "
                "then evaluated with a rolling threshold over strictly past scores."
            ),
            "raps": (
                "APS cumulative mass plus lambda times the amount by which a label "
                "rank exceeds k_reg."
            ),
        },
        "deletion_rates": list(selected_rates),
        "device": str(device),
        "effect_deletion_rate": float(effect_deletion_rate),
        "iterations": int(iterations),
        "methods": METHOD_ORDER,
        "output_name": output_name,
        "outputs": outputs,
        "run_root": str(run_root),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seeds": list(selected_seeds),
        "selection_tolerance": float(selection_tolerance),
        "success_gate": success_gate,
        "target_coverage": float(config.target_coverage),
    }
    manifest_path = data_dir / "score_adaptive_shortlist_manifest.json"
    _write_json(manifest, manifest_path)
    manifest["outputs"]["score_adaptive_shortlist_manifest.json"] = sha256_file(
        manifest_path
    )
    _write_json(manifest, manifest_path)
    return manifest


def _parse_int_tuple(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    parsed = tuple(int(part) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one integer is required")
    return parsed


def _parse_float_tuple(value: str | None) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    parsed = tuple(float(part) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one float is required")
    return parsed


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
        "--raps-k-values",
        default=",".join(str(value) for value in DEFAULT_RAPS_K),
    )
    parser.add_argument(
        "--raps-penalties",
        default=",".join(f"{value:g}" for value in DEFAULT_RAPS_LAMBDAS),
    )
    parser.add_argument("--selection-tolerance", type=float, default=0.02)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    manifest = export_score_adaptive_shortlist(
        args.run_root,
        args.paper_root,
        data_root=args.data_root,
        output_name=args.output_name,
        seeds=_parse_int_tuple(args.seeds),
        deletion_rates=_parse_float_tuple(args.deletion_rates),
        effect_deletion_rate=args.effect_deletion_rate,
        raps_k_values=_parse_int_tuple(args.raps_k_values) or DEFAULT_RAPS_K,
        raps_penalties=_parse_float_tuple(args.raps_penalties) or DEFAULT_RAPS_LAMBDAS,
        selection_tolerance=args.selection_tolerance,
        block_length=args.block_length,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
        device_name=args.device,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
