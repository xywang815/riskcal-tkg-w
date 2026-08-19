"""Export coverage-preserving shortlist calibration diagnostics.

This exporter reuses a completed confirmatory run and its checkpoints.  It does
not retrain the scorer.  The experiment compares the current margin-threshold
sets with rank-threshold conformal sets that return a top-k shortlist selected
from strictly past calibration scores.
"""

from __future__ import annotations

from argparse import ArgumentParser
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
from scripts.export_query_level_diagnostics import (  # noqa: E402
    build_query_level_rows,
    sha256_file,
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
METHOD_ORDER = [
    "margin_static",
    "margin_rolling",
    "rank_static",
    "rank_expanding",
    "rank_rolling",
]


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


def _rate_label(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


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


def _validate_scores_and_labels(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or label_array.shape != (len(values),):
        raise ValueError("scores and labels have incompatible shapes")
    if len(values) == 0:
        raise ValueError("scores must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if label_array.min() < 0 or label_array.max() >= values.shape[1]:
        raise ValueError("labels contain an out-of-range entity ID")
    return values, label_array


def descending_score_order(scores: np.ndarray) -> np.ndarray:
    """Return entity indices sorted by descending score with stable ID tie breaks."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("scores must have shape (n, classes)")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return np.argsort(-values, axis=1, kind="stable")


def true_label_ranks(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One-based unfiltered rank of each true label under the model scores."""
    values, label_array = _validate_scores_and_labels(scores, labels)
    order = descending_score_order(values)
    inverse = np.empty_like(order)
    inverse[np.arange(len(values))[:, None], order] = np.arange(1, values.shape[1] + 1)
    return inverse[np.arange(len(values)), label_array].astype(float)


def topk_prediction_mask(scores: np.ndarray, k: int | float) -> np.ndarray:
    """Boolean top-k prediction-set mask."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("scores must have shape (n, classes)")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if math.isnan(float(k)):
        raise ValueError("k must not be NaN")
    k_int = int(math.ceil(float(k)))
    if k_int <= 0:
        return np.zeros(values.shape, dtype=bool)
    if k_int >= values.shape[1]:
        return np.ones(values.shape, dtype=bool)
    order = descending_score_order(values)
    mask = np.zeros(values.shape, dtype=bool)
    mask[np.arange(len(values))[:, None], order[:, :k_int]] = True
    return mask


def _summarize_mask(
    *,
    seed: int,
    deletion_rate: float,
    method: str,
    timestamp: int,
    facts: np.ndarray,
    prediction_sides: np.ndarray,
    labels: np.ndarray,
    ranks: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    pool_score_count: int,
    pool_span_blocks: int,
    threshold_units: str,
    num_entities: int,
) -> dict[str, Any]:
    sizes = mask.sum(axis=1)
    covered = mask[np.arange(len(mask)), labels]
    top1_correct = ranks == 1

    label_rows = pd.DataFrame(
        {
            "seed": int(seed),
            "deletion_rate": float(deletion_rate),
            "method": str(method),
            "prediction_side": prediction_sides.astype(str),
            "timestamp": int(timestamp),
            "subject_id": facts[:, 0].astype(int),
            "relation_id": facts[:, 1].astype(int),
            "true_object_id": facts[:, 2].astype(int),
            "rank": ranks.astype(int),
            "frequency_rank": ranks.astype(int),
            "set_size": sizes.astype(int),
            "covered": covered.astype(bool),
            "top1_correct": top1_correct,
        }
    )
    queries = build_query_level_rows(label_rows)
    label_count = int(len(label_rows))
    unique_query_count = int(len(queries))
    covered_label_count = int(label_rows["covered"].sum())
    full_set_covered_count = int(queries["full_set_covered"].sum())
    partial_recall_sum = float(queries["partial_answer_recall"].sum())
    query_sizes = queries["set_size"].to_numpy(dtype=float)
    return {
        "seed": int(seed),
        "deletion_rate": float(deletion_rate),
        "method": str(method),
        "timestamp": int(timestamp),
        "threshold": float(threshold),
        "threshold_units": threshold_units,
        "pool_score_count": int(pool_score_count),
        "pool_span_blocks": int(pool_span_blocks),
        "label_row_count": label_count,
        "unique_query_count": unique_query_count,
        "observed_label_coverage": float(covered_label_count / label_count),
        "full_set_coverage": float(full_set_covered_count / unique_query_count),
        "partial_answer_recall": float(partial_recall_sum / unique_query_count),
        "mean_size": float(np.mean(query_sizes)),
        "median_size": float(np.median(query_sizes)),
        "p90_size": float(np.quantile(query_sizes, 0.9)),
        "singleton_rate": float((query_sizes == 1).mean()),
        "full_vocabulary_set_rate": float((query_sizes >= num_entities).mean()),
    }


def summarize_shortlist_rows(rows: pd.DataFrame, target_coverage: float) -> pd.DataFrame:
    required = {
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
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"shortlist rows are missing columns: {missing}")
    records: list[dict[str, Any]] = []
    for (seed, deletion_rate, method), frame in rows.groupby(
        ["seed", "deletion_rate", "method"], sort=False
    ):
        query_weights = frame["unique_query_count"].to_numpy(dtype=float)
        label_weights = frame["label_row_count"].to_numpy(dtype=float)
        records.append(
            {
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
                    np.maximum(target_coverage - frame["observed_label_coverage"], 0.0).mean()
                ),
                "positive_full_set_undercoverage": float(
                    np.maximum(target_coverage - frame["full_set_coverage"], 0.0).mean()
                ),
                "fraction_timestamps_label_below_target": float(
                    (frame["observed_label_coverage"] < target_coverage).mean()
                ),
                "fraction_timestamps_full_set_below_target": float(
                    (frame["full_set_coverage"] < target_coverage).mean()
                ),
                "mean_size": float(np.average(frame["mean_size"], weights=query_weights)),
                "median_size": float(np.average(frame["median_size"], weights=query_weights)),
                "p90_size": float(np.average(frame["p90_size"], weights=query_weights)),
                "full_vocabulary_set_rate": float(
                    np.average(frame["full_vocabulary_set_rate"], weights=query_weights)
                ),
                "pool_score_count_mean": float(frame["pool_score_count"].mean()),
                "pool_span_blocks_mean": float(frame["pool_span_blocks"].mean()),
            }
        )
    return _method_sort(pd.DataFrame(records), ["seed"])


def aggregate_shortlist_summary(summary: pd.DataFrame) -> pd.DataFrame:
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
    for (deletion_rate, method), frame in summary.groupby(
        ["deletion_rate", "method"], sort=False
    ):
        record: dict[str, Any] = {
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
    return _method_sort(pd.DataFrame(records))


def build_paper_table(
    summary: pd.DataFrame,
    *,
    deletion_rate: float,
    methods: tuple[str, ...] = ("margin_rolling", "rank_rolling"),
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


def build_shortlist_effect_frames(
    rows: pd.DataFrame,
    *,
    deletion_rate: float,
    baseline_method: str = "margin_rolling",
    shortlist_method: str = "rank_rolling",
) -> pd.DataFrame:
    selected = rows[
        (rows["deletion_rate"] == float(deletion_rate))
        & (rows["method"].isin([baseline_method, shortlist_method]))
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
        raise ValueError("baseline and shortlist rows must exist for every timestamp")
    base = pivot.reset_index()
    baseline_size = base[("mean_size", baseline_method)].astype(float)
    shortlist_size = base[("mean_size", shortlist_method)].astype(float)
    baseline_p90 = base[("p90_size", baseline_method)].astype(float)
    shortlist_p90 = base[("p90_size", shortlist_method)].astype(float)
    return pd.DataFrame(
        {
            "seed": base["seed"].astype(int),
            "timestamp": base["timestamp"].astype(int),
            "deletion_rate": float(deletion_rate),
            "unique_query_count": base[("unique_query_count", baseline_method)].astype(float),
            "label_row_count": base[("label_row_count", baseline_method)].astype(float),
            "mean_size_reduction": baseline_size - shortlist_size,
            "relative_mean_size_reduction": np.where(
                baseline_size > 0,
                (baseline_size - shortlist_size) / baseline_size,
                np.nan,
            ),
            "p90_size_reduction": baseline_p90 - shortlist_p90,
            "observed_label_coverage_delta": (
                base[("observed_label_coverage", shortlist_method)].astype(float)
                - base[("observed_label_coverage", baseline_method)].astype(float)
            ),
            "full_set_coverage_delta": (
                base[("full_set_coverage", shortlist_method)].astype(float)
                - base[("full_set_coverage", baseline_method)].astype(float)
            ),
        }
    )


def bootstrap_shortlist_effects(
    rows: pd.DataFrame,
    *,
    deletion_rate: float,
    block_length: int,
    iterations: int,
    bootstrap_seed: int,
    baseline_method: str = "margin_rolling",
    shortlist_method: str = "rank_rolling",
) -> pd.DataFrame:
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    effects = build_shortlist_effect_frames(
        rows,
        deletion_rate=deletion_rate,
        baseline_method=baseline_method,
        shortlist_method=shortlist_method,
    )
    specs = [
        (
            "rank_rolling_mean_size_reduction_vs_margin_rolling",
            "mean_size_reduction",
            "unique_query_count",
            "positive",
        ),
        (
            "rank_rolling_relative_mean_size_reduction_vs_margin_rolling",
            "relative_mean_size_reduction",
            "unique_query_count",
            "positive",
        ),
        (
            "rank_rolling_p90_size_reduction_vs_margin_rolling",
            "p90_size_reduction",
            "unique_query_count",
            "positive",
        ),
        (
            "rank_rolling_label_coverage_delta_vs_margin_rolling",
            "observed_label_coverage_delta",
            "label_row_count",
            "not_negative",
        ),
        (
            "rank_rolling_full_set_coverage_delta_vs_margin_rolling",
            "full_set_coverage_delta",
            "unique_query_count",
            "not_negative",
        ),
    ]
    records: list[dict[str, Any]] = []
    for offset, (name, value_column, weight_column, direction) in enumerate(specs):
        result = _bootstrap_statistic(
            effects,
            value_column,
            weight_column=weight_column,
            block_length=block_length,
            iterations=iterations,
            bootstrap_seed=bootstrap_seed + offset,
        )
        records.append(
            {
                "statistic": name,
                "deletion_rate": float(deletion_rate),
                "observed": result["observed"],
                "ci95_low": result["ci95"][0],
                "ci95_high": result["ci95"][1],
                "pvalue_positive": result["pvalue_positive"],
                "iterations": int(iterations),
                "block_length": int(block_length),
                "seed_count": result["seed_count"],
                "timestamp_count": result["timestamp_count"],
                "direction": direction,
                "baseline_method": baseline_method,
                "shortlist_method": shortlist_method,
            }
        )
    return pd.DataFrame(records)


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
    device: Any,
) -> list[dict[str, Any]]:
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

    margin_history = ScoreHistory()
    rank_history = ScoreHistory()
    static_margins: list[np.ndarray] = []
    static_ranks: list[np.ndarray] = []
    for timestamp in np.unique(roles.final_calibration.timestamps):
        raw_facts = roles.final_calibration.values[
            roles.final_calibration.timestamps == timestamp
        ]
        facts = add_inverse_relations(raw_facts, relation_count)
        scores = _score_all_objects(model, facts, batch_size, device)
        labels = facts[:, 2].copy()
        margins = margin_nonconformity(scores, labels)
        ranks = true_label_ranks(scores, labels)
        timestamp_int = int(timestamp)
        margin_history.add(timestamp_int, margins)
        rank_history.add(timestamp_int, ranks)
        static_margins.append(margins)
        static_ranks.append(ranks)

    static_margin_threshold = finite_sample_quantile(
        np.concatenate(static_margins),
        alpha=1.0 - target_coverage,
    )
    static_rank_threshold = finite_sample_quantile(
        np.concatenate(static_ranks),
        alpha=1.0 - target_coverage,
    )

    rows: list[dict[str, Any]] = []
    for timestamp in np.unique(split.test.timestamps):
        raw_facts = split.test.values[split.test.timestamps == timestamp]
        facts = add_inverse_relations(raw_facts, relation_count)
        prediction_sides = np.asarray(
            ["object"] * len(raw_facts) + ["subject"] * len(raw_facts)
        )
        scores = _score_all_objects(model, facts, batch_size, device)
        labels = facts[:, 2].copy()
        ranks = true_label_ranks(scores, labels)
        margins = margin_nonconformity(scores, labels)
        timestamp_int = int(timestamp)

        expanding_margins, expanding_margin_timestamps = margin_history.values_before(
            timestamp_int
        )
        expanding_ranks, expanding_rank_timestamps = rank_history.values_before(
            timestamp_int
        )
        rolling_margins, rolling_margin_timestamps = margin_history.values_before(
            timestamp_int,
            max_count=rolling_window,
        )
        rolling_ranks, rolling_rank_timestamps = rank_history.values_before(
            timestamp_int,
            max_count=rolling_window,
        )
        method_specs = [
            (
                "margin_static",
                prediction_set_mask(scores, static_margin_threshold),
                static_margin_threshold,
                len(expanding_margins),
                int(timestamp_int - expanding_margin_timestamps.min()),
                "score_margin",
            ),
            (
                "margin_rolling",
                prediction_set_mask(
                    scores,
                    finite_sample_quantile(
                        rolling_margins,
                        alpha=1.0 - target_coverage,
                    ),
                ),
                finite_sample_quantile(
                    rolling_margins,
                    alpha=1.0 - target_coverage,
                ),
                len(rolling_margins),
                int(timestamp_int - rolling_margin_timestamps.min()),
                "score_margin",
            ),
            (
                "rank_static",
                topk_prediction_mask(scores, static_rank_threshold),
                static_rank_threshold,
                len(expanding_ranks),
                int(timestamp_int - expanding_rank_timestamps.min()),
                "rank",
            ),
            (
                "rank_expanding",
                topk_prediction_mask(
                    scores,
                    finite_sample_quantile(
                        expanding_ranks,
                        alpha=1.0 - target_coverage,
                    ),
                ),
                finite_sample_quantile(
                    expanding_ranks,
                    alpha=1.0 - target_coverage,
                ),
                len(expanding_ranks),
                int(timestamp_int - expanding_rank_timestamps.min()),
                "rank",
            ),
            (
                "rank_rolling",
                topk_prediction_mask(
                    scores,
                    finite_sample_quantile(
                        rolling_ranks,
                        alpha=1.0 - target_coverage,
                    ),
                ),
                finite_sample_quantile(
                    rolling_ranks,
                    alpha=1.0 - target_coverage,
                ),
                len(rolling_ranks),
                int(timestamp_int - rolling_rank_timestamps.min()),
                "rank",
            ),
        ]
        for method, mask, threshold, pool_count, pool_span, threshold_units in method_specs:
            rows.append(
                _summarize_mask(
                    seed=seed,
                    deletion_rate=deletion_rate,
                    method=method,
                    timestamp=timestamp_int,
                    facts=facts,
                    prediction_sides=prediction_sides,
                    labels=labels,
                    ranks=ranks,
                    mask=mask,
                    threshold=float(threshold),
                    pool_score_count=pool_count,
                    pool_span_blocks=pool_span,
                    threshold_units=threshold_units,
                    num_entities=len(table.entity_to_id),
                )
            )

        margin_history.add(timestamp_int, margins)
        rank_history.add(timestamp_int, ranks)

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _filter_numbers(values: tuple[int | float, ...], requested: tuple[int | float, ...] | None) -> tuple[int | float, ...]:
    if requested is None:
        return values
    wanted = {float(value) for value in requested}
    return tuple(value for value in values if float(value) in wanted)


def export_shortlist_calibration(
    run_root: Path,
    paper_root: Path,
    *,
    data_root: Path | None = None,
    output_name: str = "final_confirmatory",
    seeds: tuple[int, ...] | None = None,
    deletion_rates: tuple[float, ...] | None = None,
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

    selected_seeds = tuple(int(seed) for seed in _filter_numbers(config.seeds or (config.seed,), seeds))
    selected_rates = tuple(
        float(rate) for rate in _filter_numbers(config.deletion_rates, deletion_rates)
    )
    if not selected_seeds:
        raise ValueError("no seeds selected")
    if not selected_rates:
        raise ValueError("no deletion rates selected")
    rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        for deletion_rate in selected_rates:
            rows.extend(
                _evaluate_condition(
                    run_root=run_root,
                    table=table,
                    split=split,
                    seed=seed,
                    deletion_rate=deletion_rate,
                    embedding_dim=config.embedding_dim,
                    batch_size=config.batch_size,
                    rolling_window=config.rolling_window,
                    target_coverage=config.target_coverage,
                    device=device,
                )
            )

    data_dir = paper_root / "data" / output_name
    row_frame = pd.DataFrame(rows)
    by_seed = summarize_shortlist_rows(row_frame, config.target_coverage)
    summary = aggregate_shortlist_summary(by_seed)
    if effect_deletion_rate is None:
        effect_deletion_rate = float(max(selected_rates))
    paper_table = build_paper_table(summary, deletion_rate=float(effect_deletion_rate))
    effects = bootstrap_shortlist_effects(
        row_frame,
        deletion_rate=float(effect_deletion_rate),
        block_length=block_length,
        iterations=iterations,
        bootstrap_seed=bootstrap_seed,
    )
    success_gate = _success_gate(
        summary,
        effects,
        target_coverage=config.target_coverage,
        deletion_rate=float(effect_deletion_rate),
    )

    output_paths = {
        "shortlist_calibration_by_timestamp.csv": data_dir
        / "shortlist_calibration_by_timestamp.csv",
        "shortlist_calibration_by_seed.csv": data_dir
        / "shortlist_calibration_by_seed.csv",
        "shortlist_calibration_summary.csv": data_dir
        / "shortlist_calibration_summary.csv",
        "shortlist_calibration_paper_table.csv": data_dir
        / "shortlist_calibration_paper_table.csv",
        "shortlist_calibration_effects.csv": data_dir
        / "shortlist_calibration_effects.csv",
        "shortlist_calibration_success_gate.json": data_dir
        / "shortlist_calibration_success_gate.json",
    }
    _write_csv(row_frame, output_paths["shortlist_calibration_by_timestamp.csv"])
    _write_csv(by_seed, output_paths["shortlist_calibration_by_seed.csv"])
    _write_csv(summary, output_paths["shortlist_calibration_summary.csv"])
    _write_csv(paper_table, output_paths["shortlist_calibration_paper_table.csv"])
    _write_csv(effects, output_paths["shortlist_calibration_effects.csv"])
    _write_json(
        success_gate,
        output_paths["shortlist_calibration_success_gate.json"],
    )
    outputs = {name: sha256_file(path) for name, path in output_paths.items()}
    manifest = {
        "block_length": int(block_length),
        "bootstrap_seed": int(bootstrap_seed),
        "condition_count": len(selected_seeds) * len(selected_rates),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "definition": {
            "margin_static": (
                "Original score-margin conformal threshold fixed from final "
                "initial calibration scores."
            ),
            "margin_rolling": (
                f"Original score-margin conformal threshold over the most recent "
                f"{config.rolling_window} strictly past nonconformity scores."
            ),
            "rank_static": (
                "Top-k prediction set where k is the conformal quantile of true-label "
                "ranks on final initial calibration scores."
            ),
            "rank_expanding": (
                "Top-k prediction set where k is the conformal quantile of all "
                "strictly past true-label ranks."
            ),
            "rank_rolling": (
                f"Top-k prediction set where k is the conformal quantile of the "
                f"most recent {config.rolling_window} strictly past true-label ranks."
            ),
            "observed_label_coverage": (
                "Marginal coverage over observed subject/object prediction labels."
            ),
            "full_set_coverage": (
                "Fraction of unique temporal queries for which every observed "
                "answer label is included."
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
        "success_gate": success_gate,
        "target_coverage": float(config.target_coverage),
    }
    manifest_path = data_dir / "shortlist_calibration_manifest.json"
    _write_json(manifest, manifest_path)
    manifest["outputs"]["shortlist_calibration_manifest.json"] = sha256_file(
        manifest_path
    )
    _write_json(manifest, manifest_path)
    return manifest


def _success_gate(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    target_coverage: float,
    deletion_rate: float,
) -> dict[str, Any]:
    selected = summary[
        (summary["deletion_rate"] == float(deletion_rate))
        & (summary["method"] == "rank_rolling")
    ]
    if len(selected) != 1:
        return {
            "status": "not_evaluated",
            "reason": "rank_rolling summary row missing",
            "supported": False,
        }
    row = selected.iloc[0]
    size_effect = effects[
        effects["statistic"]
        == "rank_rolling_relative_mean_size_reduction_vs_margin_rolling"
    ]
    label_delta = effects[
        effects["statistic"] == "rank_rolling_label_coverage_delta_vs_margin_rolling"
    ]
    full_delta = effects[
        effects["statistic"] == "rank_rolling_full_set_coverage_delta_vs_margin_rolling"
    ]
    label_coverage_ok = (
        float(row["observed_label_coverage_mean"]) >= float(target_coverage) - 0.02
    )
    size_reduction_supported = (
        len(size_effect) == 1 and float(size_effect.iloc[0]["ci95_low"]) > 0.0
    )
    label_delta_not_materially_negative = (
        len(label_delta) == 1 and float(label_delta.iloc[0]["ci95_low"]) >= -0.02
    )
    full_delta_not_materially_negative = (
        len(full_delta) == 1 and float(full_delta.iloc[0]["ci95_low"]) >= -0.05
    )
    supported = all(
        [
            label_coverage_ok,
            size_reduction_supported,
            label_delta_not_materially_negative,
            full_delta_not_materially_negative,
        ]
    )
    return {
        "status": "evaluated",
        "supported": bool(supported),
        "target_coverage": float(target_coverage),
        "effect_deletion_rate": float(deletion_rate),
        "rank_rolling_observed_label_coverage_mean": float(
            row["observed_label_coverage_mean"]
        ),
        "rank_rolling_full_set_coverage_mean": float(row["full_set_coverage_mean"]),
        "rank_rolling_mean_size_mean": float(row["mean_size_mean"]),
        "criteria": {
            "label_coverage_at_least_target_minus_0p02": bool(label_coverage_ok),
            "relative_mean_size_reduction_ci_low_positive": bool(
                size_reduction_supported
            ),
            "label_coverage_delta_ci_low_at_least_minus_0p02": bool(
                label_delta_not_materially_negative
            ),
            "full_set_coverage_delta_ci_low_at_least_minus_0p05": bool(
                full_delta_not_materially_negative
            ),
        },
    }


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
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    manifest = export_shortlist_calibration(
        args.run_root,
        args.paper_root,
        data_root=args.data_root,
        output_name=args.output_name,
        seeds=_parse_int_tuple(args.seeds),
        deletion_rates=_parse_float_tuple(args.deletion_rates),
        effect_deletion_rate=args.effect_deletion_rate,
        block_length=args.block_length,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
        device_name=args.device,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
