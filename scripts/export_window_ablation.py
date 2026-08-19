"""Export reviewer-facing rolling-window and half-life ablations.

This script reuses a completed confirmatory run and its checkpoints.  It does
not retrain the scorer.  Instead, it re-scores the calibration and test
timestamps for each completed condition, then evaluates alternative calibration
histories that directly answer the reviewer concern that a score-count window
of 1000 may cover too few timestamps to activate 7/14/30-day half-lives.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskcal_tkg.calibration import (
    CalibrationBatch,
    finite_sample_quantile,
    margin_nonconformity,
    prediction_set_mask,
    select_half_life_with_validation,
    weighted_quantile,
)
from riskcal_tkg.data import (
    QuadrupleTable,
    add_inverse_relations,
    split_calibration_roles,
    temporal_split,
)
from riskcal_tkg.metrics import coverage_and_size


DEFAULT_COUNT_WINDOWS = (250, 500, 1000, 2000)
DEFAULT_TIME_WINDOWS = (3, 7, 14, 30)
METHOD_ORDER = [
    "static",
    "expanding",
    "rolling_count_250",
    "rolling_count_500",
    "rolling_count_1000",
    "rolling_count_2000",
    "weighted_count_250",
    "weighted_count_500",
    "weighted_count_1000",
    "weighted_count_2000",
    "time_window_3",
    "time_window_7",
    "time_window_14",
    "time_window_30",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate_label(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _finite_or_inf(value: float) -> str:
    return "inf" if math.isinf(float(value)) else f"{float(value):g}"


@dataclass
class ScoreHistory:
    """Strictly past nonconformity-score history."""

    _timestamps: list[np.ndarray] = field(default_factory=list)
    _scores: list[np.ndarray] = field(default_factory=list)

    def add(self, timestamp: int, scores: np.ndarray) -> None:
        values = np.asarray(scores, dtype=float).reshape(-1)
        if len(values) == 0:
            raise ValueError("scores must not be empty")
        if not np.isfinite(values).all():
            raise ValueError("scores must be finite")
        self._timestamps.append(np.full(len(values), int(timestamp), dtype=np.int64))
        self._scores.append(values.copy())

    def values_before(
        self,
        timestamp: int,
        *,
        max_count: int | None = None,
        lookback_blocks: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._scores:
            raise ValueError("score history is empty")
        if max_count is not None and max_count <= 0:
            raise ValueError("max_count must be positive")
        if lookback_blocks is not None and lookback_blocks <= 0:
            raise ValueError("lookback_blocks must be positive")

        timestamps = np.concatenate(self._timestamps)
        scores = np.concatenate(self._scores)
        mask = timestamps < int(timestamp)
        if lookback_blocks is not None:
            mask &= timestamps >= int(timestamp) - int(lookback_blocks)
        timestamps = timestamps[mask]
        scores = scores[mask]
        if len(scores) == 0:
            raise ValueError("no strictly earlier scores are available")
        if max_count is not None and len(scores) > max_count:
            timestamps = timestamps[-max_count:]
            scores = scores[-max_count:]
        return scores, timestamps


def effective_sample_size(timestamps: np.ndarray, timestamp: int, half_life: float) -> float:
    if len(timestamps) == 0:
        raise ValueError("timestamps must not be empty")
    if math.isinf(float(half_life)):
        weights = np.ones(len(timestamps), dtype=float)
    else:
        if half_life <= 0 or math.isnan(float(half_life)):
            raise ValueError("half_life must be positive")
        ages = int(timestamp) - np.asarray(timestamps, dtype=float)
        weights = np.exp2(-ages / float(half_life))
    return float(weights.sum() ** 2 / np.square(weights).sum())


def weighted_threshold_from_history(
    scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    timestamp: int,
    target_coverage: float,
    half_life: float,
) -> float:
    if math.isinf(float(half_life)):
        weights = np.ones(len(scores), dtype=float)
    else:
        ages = int(timestamp) - np.asarray(timestamps, dtype=float)
        weights = np.exp2(-ages / float(half_life))
    return weighted_quantile(scores, weights, target_coverage)


def _method_sort(frame: pd.DataFrame) -> pd.DataFrame:
    rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result = frame.copy()
    result["_method_rank"] = result["method"].map(rank).fillna(len(rank))
    return result.sort_values(
        ["deletion_rate", "_method_rank", "method"], kind="stable"
    ).drop(columns="_method_rank").reset_index(drop=True)


def summarize_method_rows(rows: pd.DataFrame, target_coverage: float) -> pd.DataFrame:
    required = {
        "seed",
        "deletion_rate",
        "method",
        "query_count",
        "coverage",
        "mean_size",
        "median_size",
        "p90_size",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"method rows are missing columns: {missing}")
    records: list[dict[str, Any]] = []
    for (seed, deletion_rate, method), frame in rows.groupby(
        ["seed", "deletion_rate", "method"], sort=False
    ):
        weights = frame["query_count"].to_numpy(dtype=float)
        records.append(
            {
                "seed": int(seed),
                "deletion_rate": float(deletion_rate),
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
            }
        )
    return _method_sort(pd.DataFrame(records))


def aggregate_method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "coverage",
        "macro_time_coverage",
        "positive_undercoverage",
        "fraction_timestamps_below_target",
        "mean_size",
        "median_size",
        "p90_size",
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


def build_interaction_effects(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[
        summary["method"].isin(["static", "rolling_count_1000", "expanding"])
    ]
    pivot = selected.pivot_table(
        index=["seed", "deletion_rate"],
        columns="method",
        values="positive_undercoverage",
        aggfunc="first",
    ).reset_index()
    if {"static", "rolling_count_1000"} <= set(pivot.columns):
        pivot["rolling1000_undercoverage_reduction_vs_static"] = (
            pivot["static"] - pivot["rolling_count_1000"]
        )
    if {"static", "expanding"} <= set(pivot.columns):
        pivot["expanding_undercoverage_reduction_vs_static"] = (
            pivot["static"] - pivot["expanding"]
        )
    if "rolling1000_undercoverage_reduction_vs_static" not in pivot:
        return pivot
    base = pivot[pivot["deletion_rate"] == 0.0][
        ["seed", "rolling1000_undercoverage_reduction_vs_static"]
    ].rename(
        columns={
            "rolling1000_undercoverage_reduction_vs_static": "rolling_gain_delete0"
        }
    )
    merged = pivot.merge(base, on="seed", how="left")
    merged["rolling_gain_interaction_vs_delete0"] = (
        merged["rolling1000_undercoverage_reduction_vs_static"]
        - merged["rolling_gain_delete0"]
    )
    return merged


def summarize_pool_rows(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (window_label, half_life), frame in rows.groupby(
        ["window_label", "half_life"], sort=False
    ):
        records.append(
            {
                "window_label": str(window_label),
                "half_life": str(half_life),
                "timestamp_rows": int(len(frame)),
                "score_count_mean": float(frame["score_count"].mean()),
                "score_count_median": float(frame["score_count"].median()),
                "span_blocks_mean": float(frame["span_blocks"].mean()),
                "span_blocks_median": float(frame["span_blocks"].median()),
                "span_blocks_min": float(frame["span_blocks"].min()),
                "span_blocks_max": float(frame["span_blocks"].max()),
                "unique_timestamp_count_median": float(
                    frame["unique_timestamp_count"].median()
                ),
                "neff_mean": float(frame["effective_sample_size"].mean()),
                "neff_median": float(frame["effective_sample_size"].median()),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["window_label", "half_life"], kind="stable"
    ).reset_index(drop=True)


def summarize_selection_rows(rows: pd.DataFrame) -> pd.DataFrame:
    selected = rows[rows["selected"]].copy()
    if selected.empty:
        return pd.DataFrame()
    counts = (
        selected.groupby(["window_label", "selected_half_life"], as_index=False)
        .agg(condition_count=("seed", "count"))
        .sort_values(["window_label", "selected_half_life"], kind="stable")
    )
    total = counts.groupby("window_label")["condition_count"].transform("sum")
    counts["condition_fraction"] = counts["condition_count"] / total
    return counts


def _score_all_objects(
    model: Any,
    values: np.ndarray,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import torch

    batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size][:, [0, 1, 3]],
                dtype=torch.long,
                device=device,
            )
            batches.append(model.score_all_objects(batch).detach().cpu().numpy())
    if not batches:
        raise ValueError("cannot score an empty fact array")
    return np.concatenate(batches, axis=0)


def _score_batches(
    model: Any,
    table: QuadrupleTable,
    relation_count: int,
    batch_size: int,
    device: Any,
) -> tuple[list[CalibrationBatch], list[np.ndarray]]:
    batches: list[CalibrationBatch] = []
    margins: list[np.ndarray] = []
    for timestamp in np.unique(table.timestamps):
        raw_facts = table.values[table.timestamps == timestamp]
        facts = add_inverse_relations(raw_facts, relation_count)
        scores = _score_all_objects(model, facts, batch_size, device)
        labels = facts[:, 2].copy()
        current_margins = margin_nonconformity(scores, labels)
        margins.append(current_margins)
        batches.append(
            CalibrationBatch(
                timestamp=int(timestamp),
                scores=scores,
                true_ids=labels,
                subjects=facts[:, 0].copy(),
                relations=facts[:, 1].copy(),
            )
        )
    return batches, margins


def _load_condition_model(
    run_root: Path,
    *,
    seed: int,
    deletion_rate: float,
    table: QuadrupleTable,
    relation_count: int,
    embedding_dim: int,
    device: Any,
) -> Any:
    from riskcal_tkg.artifacts import load_verified_checkpoint
    from riskcal_tkg.model import TemporalDistMult

    label = f"seed{seed}_delete{_rate_label(deletion_rate)}"
    marker = _read_json(run_root / "conditions" / f"{label}.complete.json")
    checkpoint_path = run_root / "checkpoints" / f"{label}.ckpt"
    payload = load_verified_checkpoint(
        checkpoint_path,
        config_sha256=str(marker["config_sha256"]),
        dataset_sha256=str(marker["dataset_sha256"]),
        deletion_mask_sha256=str(marker["deletion_mask_sha256"]),
    )
    model = TemporalDistMult(
        len(table.entity_to_id),
        2 * relation_count,
        len(table.timestamp_to_id),
        embedding_dim,
        time_scale=int(float(payload["time_scale"])),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)  # type: ignore[arg-type]
    model.eval()
    return model


def _evaluate_threshold(
    *,
    seed: int,
    deletion_rate: float,
    method: str,
    timestamp: int,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    query_count: int,
    pool_score_count: int,
    pool_span_blocks: int,
    selected_half_life: float | None = None,
) -> dict[str, Any]:
    mask = prediction_set_mask(scores, threshold)
    calibrated = coverage_and_size(mask, labels)
    return {
        "seed": int(seed),
        "deletion_rate": float(deletion_rate),
        "method": method,
        "timestamp": int(timestamp),
        "query_count": int(query_count),
        "threshold": float(threshold),
        "pool_score_count": int(pool_score_count),
        "pool_span_blocks": int(pool_span_blocks),
        "selected_half_life": selected_half_life,
        "coverage": calibrated.coverage,
        "mean_size": calibrated.mean_size,
        "median_size": calibrated.median_size,
        "p90_size": calibrated.p90_size,
    }


def _selection_diagnostics(
    *,
    seed: int,
    deletion_rate: float,
    tuning_batches: list[CalibrationBatch],
    validation_batches: list[CalibrationBatch],
    half_lives: tuple[float, ...],
    target_coverage: float,
    count_windows: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    rows: list[dict[str, Any]] = []
    selected_by_window: dict[int, float] = {}
    for window in count_windows:
        selection = select_half_life_with_validation(
            tuning_batches,
            validation_batches,
            candidates=half_lives,
            target_coverage=target_coverage,
            max_size=window,
        )
        selected_by_window[window] = selection.selected_half_life
        for evaluation in selection.evaluations:
            rows.append(
                {
                    "seed": int(seed),
                    "deletion_rate": float(deletion_rate),
                    "window_label": f"count_{window}",
                    "half_life": _finite_or_inf(float(evaluation["half_life"])),
                    "selected_half_life": _finite_or_inf(selection.selected_half_life),
                    "selected": bool(
                        float(evaluation["half_life"]) == selection.selected_half_life
                    ),
                    "validation_coverage": float(evaluation["coverage"]),
                    "validation_mean_size": float(evaluation["mean_size"]),
                    "validation_query_count": int(evaluation["query_count"]),
                }
            )
    return rows, selected_by_window


def _pool_diagnostic_records(
    *,
    seed: int,
    deletion_rate: float,
    timestamp: int,
    history: ScoreHistory,
    half_lives: tuple[float, ...],
    count_windows: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, int | None]] = [
        (f"count_{window}", window) for window in count_windows
    ]
    specs.append(("all_history", None))
    for window_label, max_count in specs:
        scores, timestamps = history.values_before(timestamp, max_count=max_count)
        span_blocks = int(timestamp - timestamps.min())
        unique_count = int(np.unique(timestamps).size)
        for half_life in half_lives:
            rows.append(
                {
                    "seed": int(seed),
                    "deletion_rate": float(deletion_rate),
                    "timestamp": int(timestamp),
                    "window_label": window_label,
                    "half_life": _finite_or_inf(float(half_life)),
                    "score_count": int(len(scores)),
                    "span_blocks": span_blocks,
                    "unique_timestamp_count": unique_count,
                    "effective_sample_size": effective_sample_size(
                        timestamps,
                        timestamp,
                        half_life,
                    ),
                }
            )
    return rows


def _evaluate_condition(
    *,
    run_root: Path,
    table: QuadrupleTable,
    split: Any,
    seed: int,
    deletion_rate: float,
    embedding_dim: int,
    batch_size: int,
    target_coverage: float,
    half_lives: tuple[float, ...],
    count_windows: tuple[int, ...],
    time_windows: tuple[int, ...],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
    tuning_batches, _ = _score_batches(
        model,
        roles.calibrator_tuning,
        relation_count,
        batch_size,
        device,
    )
    validation_batches, _ = _score_batches(
        model,
        roles.selector_validation,
        relation_count,
        batch_size,
        device,
    )
    initial_batches, initial_margins = _score_batches(
        model,
        roles.final_calibration,
        relation_count,
        batch_size,
        device,
    )
    selection_rows, selected_by_window = _selection_diagnostics(
        seed=seed,
        deletion_rate=deletion_rate,
        tuning_batches=tuning_batches,
        validation_batches=validation_batches,
        half_lives=half_lives,
        target_coverage=target_coverage,
        count_windows=count_windows,
    )
    static_threshold = finite_sample_quantile(
        np.concatenate(initial_margins),
        alpha=1.0 - target_coverage,
    )
    history = ScoreHistory()
    for batch, margins in zip(initial_batches, initial_margins, strict=True):
        history.add(batch.timestamp, margins)

    method_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    for timestamp in np.unique(split.test.timestamps):
        raw_facts = split.test.values[split.test.timestamps == timestamp]
        facts = add_inverse_relations(raw_facts, relation_count)
        scores = _score_all_objects(model, facts, batch_size, device)
        labels = facts[:, 2].copy()
        timestamp_int = int(timestamp)
        query_count = int(len(labels))

        expanding_scores, expanding_timestamps = history.values_before(timestamp_int)
        pool_rows.extend(
            _pool_diagnostic_records(
                seed=seed,
                deletion_rate=deletion_rate,
                timestamp=timestamp_int,
                history=history,
                half_lives=half_lives,
                count_windows=count_windows,
            )
        )
        method_rows.append(
            _evaluate_threshold(
                seed=seed,
                deletion_rate=deletion_rate,
                method="static",
                timestamp=timestamp_int,
                scores=scores,
                labels=labels,
                threshold=static_threshold,
                query_count=query_count,
                pool_score_count=len(expanding_scores),
                pool_span_blocks=int(timestamp_int - expanding_timestamps.min()),
            )
        )
        method_rows.append(
            _evaluate_threshold(
                seed=seed,
                deletion_rate=deletion_rate,
                method="expanding",
                timestamp=timestamp_int,
                scores=scores,
                labels=labels,
                threshold=finite_sample_quantile(
                    expanding_scores,
                    alpha=1.0 - target_coverage,
                ),
                query_count=query_count,
                pool_score_count=len(expanding_scores),
                pool_span_blocks=int(timestamp_int - expanding_timestamps.min()),
            )
        )
        for window in count_windows:
            pool_scores, pool_timestamps = history.values_before(
                timestamp_int,
                max_count=window,
            )
            span = int(timestamp_int - pool_timestamps.min())
            method_rows.append(
                _evaluate_threshold(
                    seed=seed,
                    deletion_rate=deletion_rate,
                    method=f"rolling_count_{window}",
                    timestamp=timestamp_int,
                    scores=scores,
                    labels=labels,
                    threshold=finite_sample_quantile(
                        pool_scores,
                        alpha=1.0 - target_coverage,
                    ),
                    query_count=query_count,
                    pool_score_count=len(pool_scores),
                    pool_span_blocks=span,
                )
            )
            selected_half_life = selected_by_window[window]
            method_rows.append(
                _evaluate_threshold(
                    seed=seed,
                    deletion_rate=deletion_rate,
                    method=f"weighted_count_{window}",
                    timestamp=timestamp_int,
                    scores=scores,
                    labels=labels,
                    threshold=weighted_threshold_from_history(
                        pool_scores,
                        pool_timestamps,
                        timestamp=timestamp_int,
                        target_coverage=target_coverage,
                        half_life=selected_half_life,
                    ),
                    query_count=query_count,
                    pool_score_count=len(pool_scores),
                    pool_span_blocks=span,
                    selected_half_life=selected_half_life,
                )
            )
        for lookback in time_windows:
            pool_scores, pool_timestamps = history.values_before(
                timestamp_int,
                lookback_blocks=lookback,
            )
            method_rows.append(
                _evaluate_threshold(
                    seed=seed,
                    deletion_rate=deletion_rate,
                    method=f"time_window_{lookback}",
                    timestamp=timestamp_int,
                    scores=scores,
                    labels=labels,
                    threshold=finite_sample_quantile(
                        pool_scores,
                        alpha=1.0 - target_coverage,
                    ),
                    query_count=query_count,
                    pool_score_count=len(pool_scores),
                    pool_span_blocks=int(timestamp_int - pool_timestamps.min()),
                )
            )

        history.add(timestamp_int, margin_nonconformity(scores, labels))
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return method_rows, pool_rows, selection_rows


def export_window_ablation(
    run_root: Path,
    paper_root: Path,
    *,
    data_root: Path | None = None,
    output_name: str = "final_confirmatory",
    count_windows: tuple[int, ...] = DEFAULT_COUNT_WINDOWS,
    time_windows: tuple[int, ...] = DEFAULT_TIME_WINDOWS,
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

    seeds = config.seeds or (config.seed,)
    method_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for deletion_rate in config.deletion_rates:
            current_methods, current_pool, current_selection = _evaluate_condition(
                run_root=run_root,
                table=table,
                split=split,
                seed=int(seed),
                deletion_rate=float(deletion_rate),
                embedding_dim=config.embedding_dim,
                batch_size=config.batch_size,
                target_coverage=config.target_coverage,
                half_lives=config.half_lives,
                count_windows=count_windows,
                time_windows=time_windows,
                device=device,
            )
            method_rows.extend(current_methods)
            pool_rows.extend(current_pool)
            selection_rows.extend(current_selection)

    data_dir = paper_root / "data" / output_name
    method_frame = pd.DataFrame(method_rows)
    method_by_seed = summarize_method_rows(method_frame, config.target_coverage)
    method_summary = aggregate_method_summary(method_by_seed)
    pool_frame = pd.DataFrame(pool_rows)
    pool_summary = summarize_pool_rows(pool_frame)
    selection_frame = pd.DataFrame(selection_rows)
    selection_summary = summarize_selection_rows(selection_frame)
    interaction = build_interaction_effects(method_by_seed)

    _write_csv(method_frame, data_dir / "window_ablation_by_timestamp.csv")
    _write_csv(method_by_seed, data_dir / "window_ablation_by_seed.csv")
    _write_csv(method_summary, data_dir / "window_ablation_summary.csv")
    _write_csv(pool_frame, data_dir / "pool_diagnostics_by_timestamp.csv")
    _write_csv(pool_summary, data_dir / "pool_diagnostics_summary.csv")
    _write_csv(selection_frame, data_dir / "half_life_selection_by_condition.csv")
    _write_csv(selection_summary, data_dir / "half_life_selection_summary.csv")
    _write_csv(interaction, data_dir / "deletion_interaction_by_seed.csv")

    outputs = {
        "window_ablation_by_timestamp.csv": sha256_file(
            data_dir / "window_ablation_by_timestamp.csv"
        ),
        "window_ablation_by_seed.csv": sha256_file(
            data_dir / "window_ablation_by_seed.csv"
        ),
        "window_ablation_summary.csv": sha256_file(
            data_dir / "window_ablation_summary.csv"
        ),
        "pool_diagnostics_by_timestamp.csv": sha256_file(
            data_dir / "pool_diagnostics_by_timestamp.csv"
        ),
        "pool_diagnostics_summary.csv": sha256_file(
            data_dir / "pool_diagnostics_summary.csv"
        ),
        "half_life_selection_by_condition.csv": sha256_file(
            data_dir / "half_life_selection_by_condition.csv"
        ),
        "half_life_selection_summary.csv": sha256_file(
            data_dir / "half_life_selection_summary.csv"
        ),
        "deletion_interaction_by_seed.csv": sha256_file(
            data_dir / "deletion_interaction_by_seed.csv"
        ),
    }
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "target_coverage": config.target_coverage,
        "count_windows": list(count_windows),
        "time_windows": list(time_windows),
        "half_lives": [_finite_or_inf(value) for value in config.half_lives],
        "device": str(device),
        "condition_count": len(seeds) * len(config.deletion_rates),
        "method_timestamp_rows": int(len(method_frame)),
        "pool_timestamp_rows": int(len(pool_frame)),
        "selection_rows": int(len(selection_frame)),
        "definition": {
            "expanding": (
                "Unweighted conformal threshold over all previous final-calibration "
                "and prequential test scores."
            ),
            "rolling_count_W": (
                "Unweighted conformal threshold over the most recent W scores."
            ),
            "weighted_count_W": (
                "Exponentially weighted threshold over the most recent W scores, "
                "using the half-life selected on calibrator-tuning and "
                "selector-validation batches for that W."
            ),
            "time_window_D": (
                "Unweighted conformal threshold over strictly earlier scores whose "
                "internal timestamp IDs fall in the previous D timestamp blocks."
            ),
            "effective_sample_size": (
                "Kish effective sample size computed from half-life weights on the "
                "actual score pool available before each test timestamp."
            ),
        },
        "outputs": outputs,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    _write_json(manifest, data_dir / "window_ablation_manifest.json")
    manifest["outputs"]["window_ablation_manifest.json"] = sha256_file(
        data_dir / "window_ablation_manifest.json"
    )
    _write_json(manifest, data_dir / "window_ablation_manifest.json")
    return manifest


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("at least one integer is required")
    return parsed


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-name", default="final_confirmatory")
    parser.add_argument(
        "--count-windows",
        default="250,500,1000,2000",
        help="Comma-separated score-count windows.",
    )
    parser.add_argument(
        "--time-windows",
        default="3,7,14,30",
        help="Comma-separated timestamp-block lookback windows.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    manifest = export_window_ablation(
        args.run_root,
        args.paper_root,
        data_root=args.data_root,
        output_name=args.output_name,
        count_windows=_parse_int_tuple(args.count_windows),
        time_windows=_parse_int_tuple(args.time_windows),
        device_name=args.device,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
