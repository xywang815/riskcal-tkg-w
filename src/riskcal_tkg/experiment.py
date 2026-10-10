from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch

from .artifacts import (
    RunDirectory,
    atomic_save_checkpoint,
    atomic_write_dataframe,
    atomic_write_json,
    capture_environment,
    load_verified_checkpoint,
    sha256_file,
)
from .calibration import (
    CalibrationBatch,
    CalibrationPool,
    DRIFT_FEATURE_NAMES,
    DriftHistory,
    AdaptiveHalfLifeSelector,
    choose_adaptive_half_life,
    drift_feature_vector,
    evaluate_adaptive_half_life_selector,
    finite_sample_quantile,
    fit_adaptive_half_life_selector,
    kgcp_nonconformity,
    kgcp_prediction_set_mask,
    margin_nonconformity,
    prediction_set_mask,
    rolling_threshold,
    select_half_life,
    select_half_life_with_validation,
    weighted_threshold,
)
from .config import ExperimentConfig, load_config
from .corruptions import delete_training_edges
from .data import (
    QuadrupleTable,
    add_inverse_relations,
    built_in_toy_table,
    configured_quadruple_paths,
    load_configured_table,
    split_calibration_roles,
    split_model_selection,
    table_fingerprint,
    temporal_split,
)
from .metrics import coverage_and_size, filtered_rank, ranking_metrics, selective_metrics
from .model import (
    TemporalModel,
    TrainingConfig,
    TrainingResult,
    build_temporal_model,
    train_model,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _config_hash_payload(resolved: dict[str, Any]) -> dict[str, Any]:
    payload = dict(resolved)
    if payload.get("time_encoding") == "polynomial_fourier":
        payload.pop("time_encoding")
    return payload


def _score_all_objects(
    model: TemporalModel,
    values: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                values[start : start + batch_size][:, [0, 1, 3]],
                dtype=torch.long,
            )
            batches.append(model.score_all_objects(batch).cpu().numpy())
    if not batches:
        raise ValueError("cannot score an empty fact array")
    return np.concatenate(batches, axis=0)


def _score_spread_summary(
    model: TemporalModel,
    values: np.ndarray,
    batch_size: int,
    max_queries: int = 256,
) -> dict[str, float | int]:
    sample = values[:max_queries]
    scores = _score_all_objects(model, sample, batch_size)
    row_std = np.std(scores, axis=1)
    return {
        "query_count": int(len(sample)),
        "min_row_std": float(row_std.min()),
        "median_row_std": float(np.median(row_std)),
        "max_row_std": float(row_std.max()),
    }


def _score_calibration_batches(
    model: TemporalModel,
    table: QuadrupleTable,
    relation_count: int,
    batch_size: int,
) -> tuple[list[CalibrationBatch], list[np.ndarray]]:
    batches: list[CalibrationBatch] = []
    margins: list[np.ndarray] = []
    for timestamp in np.unique(table.timestamps):
        raw_facts = table.values[table.timestamps == timestamp]
        facts = add_inverse_relations(raw_facts, relation_count)
        scores = _score_all_objects(model, facts, batch_size)
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


def _truth_index(values: np.ndarray) -> dict[tuple[int, int, int], set[int]]:
    index: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for subject, relation, object_, timestamp in values:
        index[(int(subject), int(relation), int(timestamp))].add(int(object_))
    return dict(index)


def _rate_label(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def _retention_audit(
    values: np.ndarray,
    keep_mask: np.ndarray,
    column: int,
) -> dict[str, dict[str, float | int]]:
    audit: dict[str, dict[str, float | int]] = {}
    for value in np.unique(values[:, column]):
        selected = values[:, column] == value
        original = int(selected.sum())
        retained = int(keep_mask[selected].sum())
        audit[str(int(value))] = {
            "original": original,
            "retained": retained,
            "ratio": retained / original,
        }
    return audit


def _dataset_manifest(
    table: QuadrupleTable,
    split: Any,
    config: ExperimentConfig,
) -> dict[str, Any]:
    normalized_digest = hashlib.sha256(table.values.tobytes()).hexdigest()
    timestamps_by_id = {value: key for key, value in table.timestamp_to_id.items()}

    def timestamp_range(partition: QuadrupleTable) -> list[str]:
        return [
            timestamps_by_id[int(partition.timestamps.min())],
            timestamps_by_id[int(partition.timestamps.max())],
        ]

    source_files: dict[str, dict[str, Any]] = {}
    if config.data_mode != "toy":
        for path in configured_quadruple_paths(config):
            if path.is_file():
                source_files[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        for name in ("SOURCE.json", "LICENSE"):
            path = config.data_path / name
            if path.is_file():
                source_files[name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
    source_hashes = {
        name: str(record["sha256"]) for name, record in source_files.items()
    }
    return {
        "sha256": table_fingerprint(table, source_hashes),
        "normalized_values_sha256": normalized_digest,
        "input_facts": table.input_row_count,
        "duplicate_facts_removed": (
            0
            if table.input_row_count is None
            else table.input_row_count - len(table.values)
        ),
        "source_files": source_files,
        "num_facts": len(table.values),
        "num_entities": len(table.entity_to_id),
        "num_relations": len(table.relation_to_id),
        "num_timestamps": len(table.timestamp_to_id),
        "train_facts": len(split.train.values),
        "calibration_facts": len(split.calibration.values),
        "test_facts": len(split.test.values),
        "train_max_timestamp": int(split.train.timestamps.max()),
        "calibration_max_timestamp": int(split.calibration.timestamps.max()),
        "test_max_timestamp": int(split.test.timestamps.max()),
        "train_timestamp_range": timestamp_range(split.train),
        "calibration_timestamp_range": timestamp_range(split.calibration),
        "test_timestamp_range": timestamp_range(split.test),
    }


def _evaluate_condition(
    config: ExperimentConfig,
    table: QuadrupleTable,
    split: Any,
    seed: int,
    deletion_rate: float,
    run: RunDirectory,
    config_sha256: str,
    dataset_sha256: str,
    reuse_checkpoint: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    process = psutil.Process()
    sampled_rss = [process.memory_info().rss]
    if torch.cuda.is_available() and config.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    deletion = delete_training_edges(split.train.values, deletion_rate, seed)
    relation_count = len(table.relation_to_id)
    training_values = add_inverse_relations(deletion.values, relation_count)
    calibration_protocol = "four_role"
    try:
        calibration_roles = split_calibration_roles(
            split.calibration,
            fractions=config.calibration_role_fractions,
        )
        model_selection = calibration_roles.scorer_validation
        selector_tuning = calibration_roles.calibrator_tuning
        selector_validation = calibration_roles.selector_validation
        conformal_initial = calibration_roles.final_calibration
    except ValueError:
        calibration_protocol = "legacy_two_role"
        model_selection, conformal_initial = split_model_selection(split.calibration)
        selector_tuning = None
        selector_validation = None
    validation_values = add_inverse_relations(model_selection.values, relation_count)
    label = f"seed{seed}_delete{_rate_label(deletion_rate)}"
    deletion_path = run.root / "deletion_masks" / f"{label}.json"
    deletion_record = {
        "seed": seed,
        "requested_rate": deletion.requested_rate,
        "actual_rate": deletion.actual_rate,
        "mask_sha256": deletion.mask_sha256,
        "original_facts": len(split.train.values),
        "retained_facts": len(deletion.values),
        "retention_by_timestamp": _retention_audit(
            split.train.values, deletion.keep_mask, 3
        ),
        "retention_by_relation": _retention_audit(
            split.train.values, deletion.keep_mask, 1
        ),
    }
    if reuse_checkpoint:
        existing_deletion = json.loads(deletion_path.read_text(encoding="utf-8"))
        if existing_deletion != deletion_record:
            raise ValueError(f"deletion audit for {label} does not match checkpoint")
    else:
        atomic_write_json(deletion_path, deletion_record)
    training_config = TrainingConfig(
        model_name=config.model_name,
        time_encoding=config.time_encoding,
        negative_sampling=config.negative_sampling,
        embedding_dim=config.embedding_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        negatives=config.negatives,
        margin=config.training_margin,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=seed,
        eval_every=config.eval_every,
        patience=config.patience,
    )
    checkpoint_path = run.root / "checkpoints" / f"{label}.ckpt"
    checkpoint_payload: dict[str, object] | None = None
    if reuse_checkpoint:
        checkpoint_payload = load_verified_checkpoint(
            checkpoint_path,
            config_sha256=config_sha256,
            dataset_sha256=dataset_sha256,
            deletion_mask_sha256=deletion.mask_sha256,
        )
        model = build_temporal_model(
            config.model_name,
            len(table.entity_to_id),
            2 * relation_count,
            len(table.timestamp_to_id),
            config.embedding_dim,
            time_scale=int(float(checkpoint_payload["time_scale"])),
            time_encoding=config.time_encoding,
        )
        model.load_state_dict(checkpoint_payload["state_dict"], strict=True)  # type: ignore[arg-type]
        trained = TrainingResult(
            model=model,
            loss_history=tuple(checkpoint_payload["loss_history"]),  # type: ignore[arg-type]
            epochs_trained=int(checkpoint_payload["epochs_trained"]),
            best_epoch=int(checkpoint_payload["best_epoch"]),
            best_validation_mrr=(
                None
                if checkpoint_payload["best_validation_mrr"] is None
                else float(checkpoint_payload["best_validation_mrr"])
            ),
            epoch_seconds=tuple(checkpoint_payload["epoch_seconds"]),  # type: ignore[arg-type]
        )
        training_seconds = float(checkpoint_payload["training_seconds"])
    else:
        training_started = time.perf_counter()
        trained = train_model(
            training_values,
            num_entities=len(table.entity_to_id),
            num_relations=2 * relation_count,
            num_timestamps=len(table.timestamp_to_id),
            config=training_config,
            device=config.device,
            validation_facts=validation_values,
        )
        training_seconds = time.perf_counter() - training_started
    sampled_rss.append(process.memory_info().rss)
    training_score_spread = _score_spread_summary(
        trained.model,
        validation_values,
        config.batch_size,
    )
    if float(training_score_spread["max_row_std"]) <= config.min_score_std:
        raise FloatingPointError(
            "trained model produced degenerate all-object scores; "
            "increase score spread before running conformal evaluation"
        )

    calibration_started = time.perf_counter()
    tuning_batches: list[CalibrationBatch] = []
    validation_batches: list[CalibrationBatch] = []
    if selector_tuning is not None and selector_validation is not None:
        tuning_batches, _ = _score_calibration_batches(
            trained.model,
            selector_tuning,
            relation_count,
            config.batch_size,
        )
        validation_batches, _ = _score_calibration_batches(
            trained.model,
            selector_validation,
            relation_count,
            config.batch_size,
        )
    calibration_batches, initial_margins = _score_calibration_batches(
        trained.model,
        conformal_initial,
        relation_count,
        config.batch_size,
    )
    pool = CalibrationPool(max_size=config.rolling_window)
    drift_history = DriftHistory(num_relations=2 * relation_count)
    for batch, margins in zip(calibration_batches, initial_margins, strict=True):
        pool.add(batch.timestamp, margins)
        drift_history.add_batch(batch)

    selection_batches = tuning_batches + validation_batches
    if tuning_batches and validation_batches:
        half_life_selection = select_half_life_with_validation(
            tuning_batches,
            validation_batches,
            candidates=config.half_lives,
            target_coverage=config.target_coverage,
            max_size=config.rolling_window,
        )
    else:
        if len(selection_batches) < 2:
            selection_batches = calibration_batches
        half_life_selection = select_half_life(
            selection_batches,
            candidates=config.half_lives,
            target_coverage=config.target_coverage,
            max_size=config.rolling_window,
        )
    selected_half_life = half_life_selection.selected_half_life
    half_life_evaluations = half_life_selection.evaluations
    adaptive_fit_batches = tuning_batches if len(tuning_batches) >= 2 else selection_batches
    adaptive_selector: AdaptiveHalfLifeSelector | None = None
    adaptive_validation_evaluations: tuple[dict[str, float], ...] = ()
    if len(adaptive_fit_batches) >= 2:
        adaptive_selector = fit_adaptive_half_life_selector(
            adaptive_fit_batches,
            candidates=config.half_lives,
            target_coverage=config.target_coverage,
            max_size=config.rolling_window,
            num_relations=2 * relation_count,
            ridge=config.adaptive_selector_ridge,
            coverage_tolerance=config.adaptive_coverage_tolerance,
        )
        if validation_batches:
            adaptive_validation_evaluations = evaluate_adaptive_half_life_selector(
                adaptive_selector,
                validation_batches,
                max_size=config.rolling_window,
                warmup_batches=tuning_batches,
            )
    else:
        adaptive_selector = None
    static = finite_sample_quantile(
        np.concatenate(initial_margins),
        alpha=1.0 - config.target_coverage,
    )
    initial_scores = np.concatenate(
        [batch.scores for batch in calibration_batches],
        axis=0,
    )
    initial_labels = np.concatenate(
        [batch.true_ids for batch in calibration_batches],
        axis=0,
    )
    kgcp_thresholds = {
        method: finite_sample_quantile(
            kgcp_nonconformity(
                initial_scores,
                initial_labels,
                method,
                config.kgcp_temperature,
            ),
            alpha=1.0 - config.target_coverage,
        )
        for method in ("negscore", "minmax", "softmax")
    }
    if checkpoint_payload is None:
        atomic_save_checkpoint(
            checkpoint_path,
            {
                "state_dict": trained.model.state_dict(),
                "training_config": asdict(training_config),
                "model_name": config.model_name,
                "time_encoding": config.time_encoding,
                "negative_sampling": config.negative_sampling,
                "time_scale": float(trained.model.time.scale.item()),
                "loss_history": trained.loss_history,
                "epochs_trained": trained.epochs_trained,
                "best_epoch": trained.best_epoch,
                "best_validation_mrr": trained.best_validation_mrr,
                "epoch_seconds": trained.epoch_seconds,
                "training_seconds": training_seconds,
                "training_score_spread": training_score_spread,
                "deletion_mask_sha256": deletion.mask_sha256,
                "config_sha256": config_sha256,
                "dataset_sha256": dataset_sha256,
                "calibration_protocol": calibration_protocol,
                "selected_half_life": selected_half_life,
                "half_life_evaluations": half_life_evaluations,
                "adaptive_selector_fit_samples": (
                    None if adaptive_selector is None else adaptive_selector.fit_samples
                ),
                "adaptive_validation_evaluations": adaptive_validation_evaluations,
            },
        )
    calibration_seconds = time.perf_counter() - calibration_started
    sampled_rss.append(process.memory_info().rss)

    truth = _truth_index(add_inverse_relations(table.values, relation_count))
    object_frequency = np.bincount(
        training_values[:, 2], minlength=len(table.entity_to_id)
    ).astype(float)
    window_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    static_name = "static_margin" if config.explicit_method_names else "static"
    rolling_name = "rolling_margin" if config.explicit_method_names else "rolling"
    method_definitions = {
        "top1": "single highest-scoring entity",
        static_name: "static split-conformal max-score margin",
        rolling_name: "rolling prequential max-score margin",
        "weighted": "exponentially weighted prequential max-score margin",
        "adaptive": "validation-selected weighted prequential max-score margin",
        "kgcp_negscore_static": "published KGCP NegScore with static split calibration",
        "kgcp_minmax_static": "published KGCP Minmax with static split calibration",
        "kgcp_softmax_static": "published KGCP Softmax with static split calibration",
    }
    inference_started = time.perf_counter()
    for timestamp in np.unique(split.test.timestamps):
        raw_facts = split.test.values[split.test.timestamps == timestamp]
        facts = add_inverse_relations(raw_facts, relation_count)
        prediction_sides = np.asarray(
            ["object"] * len(raw_facts) + ["subject"] * len(raw_facts)
        )
        scores = _score_all_objects(trained.model, facts, config.batch_size)
        labels = facts[:, 2]
        calibration = pool.values_before(int(timestamp))
        drift_features = drift_feature_vector(
            scores,
            facts[:, 0],
            facts[:, 1],
            drift_history.reference(),
        )
        adaptive_decision = (
            choose_adaptive_half_life(adaptive_selector, drift_features)
            if adaptive_selector is not None
            else None
        )
        adaptive_half_life = (
            selected_half_life
            if adaptive_decision is None
            else adaptive_decision.half_life
        )
        thresholds = {
            "top1": 0.0,
            static_name: static,
            rolling_name: rolling_threshold(
                pool, int(timestamp), alpha=1.0 - config.target_coverage
            ),
            "weighted": weighted_threshold(
                pool,
                int(timestamp),
                alpha=1.0 - config.target_coverage,
                half_life=selected_half_life,
            ),
            "adaptive": weighted_threshold(
                pool,
                int(timestamp),
                alpha=1.0 - config.target_coverage,
                half_life=adaptive_half_life,
            ),
        }
        method_half_lives = {
            "top1": math.nan,
            static_name: math.nan,
            rolling_name: math.nan,
            "weighted": selected_half_life,
            "adaptive": adaptive_half_life,
        }
        if config.include_kgcp_baselines:
            for kgcp_method, threshold in kgcp_thresholds.items():
                method_name = f"kgcp_{kgcp_method}_static"
                thresholds[method_name] = threshold
                method_half_lives[method_name] = math.nan
        ranks = np.asarray(
            [
                filtered_rank(
                    scores[index],
                    true_id=int(object_),
                    other_true_ids=truth[
                        (int(subject), int(relation), int(timestamp))
                    ]
                    - {int(object_)},
                )
                for index, (subject, relation, object_, _) in enumerate(facts)
            ],
            dtype=np.int64,
        )
        ranking = ranking_metrics(ranks)
        frequency_ranks = np.asarray(
            [
                filtered_rank(
                    object_frequency,
                    true_id=int(object_),
                    other_true_ids=truth[
                        (int(subject), int(relation), int(timestamp))
                    ]
                    - {int(object_)},
                )
                for subject, relation, object_, _ in facts
            ],
            dtype=np.int64,
        )
        frequency_ranking = ranking_metrics(frequency_ranks)
        top1 = scores.argmax(axis=1)
        top1_correct = top1 == labels
        row_indices = np.arange(len(scores))
        true_scores = scores[row_indices, labels]
        score_minima = scores.min(axis=1)
        score_maxima = scores.max(axis=1)
        score_ranges = score_maxima - score_minima
        score_standard_deviations = scores.std(axis=1)
        true_margins = score_maxima - true_scores
        answer_counts = np.asarray(
            [
                len(truth[(int(subject), int(relation), int(timestamp))])
                for subject, relation, _, _ in facts
            ],
            dtype=np.int64,
        )

        for method, threshold in thresholds.items():
            if method == "top1":
                mask = np.zeros_like(scores, dtype=bool)
                mask[np.arange(len(scores)), top1] = True
            elif method.startswith("kgcp_"):
                kgcp_method = method.removeprefix("kgcp_").removesuffix("_static")
                mask = kgcp_prediction_set_mask(
                    scores,
                    threshold,
                    kgcp_method,
                    config.kgcp_temperature,
                )
            else:
                mask = prediction_set_mask(scores, threshold)
            calibrated = coverage_and_size(mask, labels)
            sizes = mask.sum(axis=1)
            row: dict[str, Any] = {
                "seed": seed,
                "deletion_rate": deletion_rate,
                "actual_deletion_rate": deletion.actual_rate,
                "dataset_mode": config.data_mode,
                "model_name": config.model_name,
                "time_encoding": config.time_encoding,
                "negative_sampling": config.negative_sampling,
                "method": method,
                "method_definition": method_definitions[method],
                "timestamp": int(timestamp),
                "calibration_max_timestamp": int(calibration.timestamps.max()),
                "threshold": threshold,
                "selected_half_life": selected_half_life,
                "method_half_life": method_half_lives[method],
                "adaptive_half_life": adaptive_half_life,
                "adaptive_predicted_coverage": (
                    None
                    if adaptive_decision is None
                    else adaptive_decision.predicted_coverage
                ),
                "adaptive_predicted_mean_size": (
                    None
                    if adaptive_decision is None
                    else adaptive_decision.predicted_mean_size
                ),
                "adaptive_coverage_feasible": (
                    None
                    if adaptive_decision is None
                    else adaptive_decision.coverage_feasible
                ),
                "calibration_protocol": calibration_protocol,
                "query_count": len(facts),
                "score_global_min": float(scores.min()),
                "score_global_max": float(scores.max()),
                "score_global_range": float(scores.max() - scores.min()),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "mean_query_score_range": float(score_ranges.mean()),
                "median_query_score_range": float(np.median(score_ranges)),
                "mean_query_score_std": float(score_standard_deviations.mean()),
                "true_score_mean": float(true_scores.mean()),
                "true_score_std": float(true_scores.std()),
                "true_margin_mean": float(true_margins.mean()),
                "true_margin_std": float(true_margins.std()),
                "coverage": calibrated.coverage,
                "mean_size": calibrated.mean_size,
                "median_size": calibrated.median_size,
                "p90_size": calibrated.p90_size,
                "singleton_rate": float((sizes == 1).mean()),
                "mrr": ranking.mrr,
                "frequency_mrr": frequency_ranking.mrr,
                "hits_at_1": ranking.hits_at_1,
                "hits_at_3": ranking.hits_at_3,
                "hits_at_10": ranking.hits_at_10,
            }
            for maximum in config.max_set_sizes:
                selective = selective_metrics(sizes, top1_correct, maximum)
                row[f"answer_rate_at_{maximum}"] = selective.answer_rate
                row[f"abstention_rate_at_{maximum}"] = selective.abstention_rate
                row[f"risk_at_{maximum}"] = selective.risk
            for feature_name, value in zip(
                DRIFT_FEATURE_NAMES,
                drift_features,
                strict=True,
            ):
                row[f"drift_{feature_name}"] = float(value)
            window_rows.append(row)
            if config.query_output_methods and method not in config.query_output_methods:
                continue
            for index, fact in enumerate(facts):
                query_rows.append(
                    {
                        "seed": seed,
                        "deletion_rate": deletion_rate,
                        "dataset_mode": config.data_mode,
                        "model_name": config.model_name,
                        "time_encoding": config.time_encoding,
                        "negative_sampling": config.negative_sampling,
                        "method": method,
                        "method_definition": method_definitions[method],
                        "prediction_side": str(prediction_sides[index]),
                        "timestamp": int(timestamp),
                        "subject_id": int(fact[0]),
                        "relation_id": int(fact[1]),
                        "true_object_id": int(fact[2]),
                        "answer_count": int(answer_counts[index]),
                        "is_multi_answer": bool(answer_counts[index] > 1),
                        "rank": int(ranks[index]),
                        "frequency_rank": int(frequency_ranks[index]),
                        "set_size": int(sizes[index]),
                        "covered": bool(mask[index, labels[index]]),
                        "top1_correct": bool(top1_correct[index]),
                        "true_score": float(true_scores[index]),
                        "query_score_min": float(score_minima[index]),
                        "query_score_max": float(score_maxima[index]),
                        "query_score_range": float(score_ranges[index]),
                        "query_score_std": float(score_standard_deviations[index]),
                        "true_margin": float(true_margins[index]),
                    }
                )
        current_margins = margin_nonconformity(scores, labels)
        pool.add(int(timestamp), current_margins)
        drift_history.add(scores, facts[:, 0], facts[:, 1])
    inference_seconds = time.perf_counter() - inference_started
    final_memory = process.memory_info()
    sampled_rss.append(final_memory.rss)
    atomic_write_json(
        run.root / "resources" / f"{label}.json",
        {
            "seed": seed,
            "deletion_rate": deletion_rate,
            "dataset_mode": config.data_mode,
            "model_name": config.model_name,
            "time_encoding": config.time_encoding,
            "negative_sampling": config.negative_sampling,
            "kgcp_temperature": config.kgcp_temperature,
            "calibration_protocol": calibration_protocol,
            "selected_half_life": selected_half_life,
            "half_life_evaluations": half_life_evaluations,
            "adaptive_selector_fit_samples": (
                None if adaptive_selector is None else adaptive_selector.fit_samples
            ),
            "adaptive_validation_evaluations": adaptive_validation_evaluations,
            "training_seconds": training_seconds,
            "training_score_spread": training_score_spread,
            "calibration_seconds": calibration_seconds,
            "inference_seconds": inference_seconds,
            "epoch_seconds": trained.epoch_seconds,
            "peak_sampled_rss_bytes": max(sampled_rss),
            "peak_rss_bytes": int(
                getattr(final_memory, "peak_wset", max(sampled_rss))
            ),
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available() and config.device != "cpu"
                else None
            ),
        },
    )
    return window_rows, query_rows


def _condition_paths(run: RunDirectory, label: str) -> dict[str, Path]:
    return {
        "checkpoint": run.root / "checkpoints" / f"{label}.ckpt",
        "deletion_mask": run.root / "deletion_masks" / f"{label}.json",
        "resources": run.root / "resources" / f"{label}.json",
        "windows": run.root / "conditions" / f"{label}.windows.csv",
        "queries": run.root / "conditions" / f"{label}.queries.parquet",
        "marker": run.root / "conditions" / f"{label}.complete.json",
    }


def _archive_partial_condition(
    run: RunDirectory,
    label: str,
    preserve: set[str] | None = None,
) -> None:
    paths = _condition_paths(run, label)
    preserved = preserve or set()
    existing = [
        path for key, path in paths.items() if key not in preserved and path.exists()
    ]
    if not existing:
        return
    archive = run.root / "incomplete" / (
        f"{label}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        path.replace(archive / path.name)


def _archive_derived_outputs(run: RunDirectory) -> None:
    targets = [
        run.root / "metrics" / "per_window.csv",
        run.root / "metrics" / "per_query.parquet",
        run.root / "metrics" / "summary.json",
        run.root / "SUCCESS_GATE.json",
        run.root / "run_manifest.json",
        *(run.root / "figures").glob("*"),
    ]
    existing = [path for path in targets if path.is_file()]
    if not existing:
        return
    archive = run.root / "incomplete" / (
        f"finalization-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    for path in existing:
        destination = archive / path.relative_to(run.root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.replace(destination)


def _load_completed_condition(
    run: RunDirectory,
    label: str,
    config_sha256: str,
    dataset_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    paths = _condition_paths(run, label)
    if not paths["marker"].is_file():
        return None
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    if marker.get("config_sha256") != config_sha256:
        raise ValueError(f"completed condition {label} has a different config hash")
    if marker.get("dataset_sha256") != dataset_sha256:
        raise ValueError(f"completed condition {label} has a different dataset hash")
    for key in ("checkpoint", "deletion_mask", "resources", "windows", "queries"):
        path = paths[key]
        if not path.is_file() or sha256_file(path) != marker["artifacts"].get(key):
            raise ValueError(f"completed condition {label} has invalid {key}")
    load_verified_checkpoint(
        paths["checkpoint"],
        config_sha256=config_sha256,
        dataset_sha256=dataset_sha256,
        deletion_mask_sha256=marker["deletion_mask_sha256"],
    )
    return pd.read_csv(paths["windows"]), pd.read_parquet(paths["queries"])


def _write_completed_condition(
    run: RunDirectory,
    label: str,
    windows: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    config_sha256: str,
    dataset_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _condition_paths(run, label)
    window_frame = pd.DataFrame(windows)
    query_frame = pd.DataFrame(queries)
    atomic_write_dataframe(paths["windows"], window_frame)
    atomic_write_dataframe(paths["queries"], query_frame)
    deletion_record = json.loads(paths["deletion_mask"].read_text(encoding="utf-8"))
    artifacts = {
        key: sha256_file(paths[key])
        for key in ("checkpoint", "deletion_mask", "resources", "windows", "queries")
    }
    atomic_write_json(
        paths["marker"],
        {
            "status": "complete",
            "config_sha256": config_sha256,
            "dataset_sha256": dataset_sha256,
            "deletion_mask_sha256": deletion_record["mask_sha256"],
            "artifacts": artifacts,
        },
    )
    return window_frame, query_frame


def run_table_experiment(
    config: ExperimentConfig,
    table: QuadrupleTable,
    output_parent: Path,
    resume_root: Path | None = None,
) -> Path:
    started = time.perf_counter()
    resolved = _jsonable(asdict(config))
    config_bytes = json.dumps(
        _config_hash_payload(resolved),
        sort_keys=True,
    ).encode("utf-8")
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config_hash = config_sha256[:12]
    split = temporal_split(
        table,
        train_fraction=config.train_fraction,
        calibration_fraction=config.calibration_fraction,
    )
    dataset_manifest = _dataset_manifest(table, split, config)
    dataset_sha256 = str(dataset_manifest["sha256"])
    if resume_root is None:
        run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{config_hash}"
        output_parent.mkdir(parents=True, exist_ok=True)
        run = RunDirectory.create(output_parent / run_id)
        atomic_write_json(run.root / "config.resolved.yaml", resolved)
        atomic_write_json(
            run.root / "environment.json",
            capture_environment(Path(__file__).parents[3]),
        )
        atomic_write_json(run.root / "dataset_manifest.json", dataset_manifest)
    else:
        run = RunDirectory(resume_root.resolve())
        run_id = run.root.name
        if not run.root.is_dir():
            raise FileNotFoundError(run.root)
        manifest_path = run.root / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete":
                raise ValueError("cannot resume an already complete run")
        existing_config = json.loads(
            (run.root / "config.resolved.yaml").read_text(encoding="utf-8")
        )
        existing_config.setdefault("time_encoding", "polynomial_fourier")
        if existing_config != resolved:
            raise ValueError("resume config does not match the original run")
        existing_dataset = json.loads(
            (run.root / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        if existing_dataset.get("sha256") != dataset_sha256:
            raise ValueError("resume dataset does not match the original run")
        for child in ("conditions", "incomplete", "resources"):
            (run.root / child).mkdir(exist_ok=True)
        _archive_derived_outputs(run)

    window_frames: list[pd.DataFrame] = []
    query_frames: list[pd.DataFrame] = []
    seeds = config.seeds or (config.seed,)
    for seed in seeds:
        for deletion_rate in config.deletion_rates:
            label = f"seed{seed}_delete{_rate_label(deletion_rate)}"
            completed = _load_completed_condition(
                run, label, config_sha256, dataset_sha256
            )
            if completed is not None:
                condition_window_frame, condition_query_frame = completed
                window_frames.append(condition_window_frame)
                query_frames.append(condition_query_frame)
                continue
            paths = _condition_paths(run, label)
            reuse_checkpoint = (
                paths["checkpoint"].is_file() and paths["deletion_mask"].is_file()
            )
            _archive_partial_condition(
                run,
                label,
                preserve=(
                    {"checkpoint", "deletion_mask"} if reuse_checkpoint else None
                ),
            )
            condition_windows, condition_queries = _evaluate_condition(
                config,
                table,
                split,
                seed,
                deletion_rate,
                run,
                config_sha256,
                dataset_sha256,
                reuse_checkpoint=reuse_checkpoint,
            )
            condition_window_frame, condition_query_frame = _write_completed_condition(
                run,
                label,
                condition_windows,
                condition_queries,
                config_sha256,
                dataset_sha256,
            )
            window_frames.append(condition_window_frame)
            query_frames.append(condition_query_frame)

    window_frame = pd.concat(window_frames, ignore_index=True)
    query_frame = pd.concat(query_frames, ignore_index=True)
    atomic_write_dataframe(run.root / "metrics" / "per_window.csv", window_frame)
    atomic_write_dataframe(run.root / "metrics" / "per_query.parquet", query_frame)
    summary = (
        window_frame.groupby(["seed", "deletion_rate", "method"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            worst_window_coverage=("coverage", "min"),
            mean_size=("mean_size", "mean"),
            mrr=("mrr", "mean"),
        )
        .to_dict("records")
    )
    atomic_write_json(run.root / "metrics" / "summary.json", summary)
    atomic_write_json(
        run.root / "SUCCESS_GATE.json",
        {"status": "pending_summary", "supported": None},
    )
    summary_script = Path(__file__).parents[2] / "scripts" / "summarize_results.py"
    subprocess.run(
        [
            sys.executable,
            str(summary_script),
            "--run-root",
            str(run.root),
            "--target",
            str(config.target_coverage),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    atomic_write_json(
        run.root / "run_manifest.json",
        {
            "status": "complete",
            "run_id": run_id,
            "conditions": len(seeds) * len(config.deletion_rates),
            "window_rows": len(window_frame),
            "query_rows": len(query_frame),
            "duration_seconds": time.perf_counter() - started,
            "directory_bytes_before_manifest": sum(
                path.stat().st_size for path in run.root.rglob("*") if path.is_file()
            ),
        },
    )
    return run.root


def run_experiment(
    config_path: Path,
    output_parent: Path | None = None,
    resume_root: Path | None = None,
) -> Path:
    config = load_config(config_path)
    table = load_configured_table(config)
    parent = output_parent if output_parent is not None else config.output_root
    return run_table_experiment(config, table, parent, resume_root=resume_root)


def smoke(output_parent: Path) -> Path:
    config_path = Path(__file__).parents[2] / "configs" / "smoke.yaml"
    config = load_config(config_path)
    return run_table_experiment(config, built_in_toy_table(), output_parent)


run_experiment.smoke = smoke  # type: ignore[attr-defined]
