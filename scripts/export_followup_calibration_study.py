from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from riskcal_tkg.artifacts import sha256_file
from riskcal_tkg.calibration import CalibrationPool, finite_sample_quantile
from riskcal_tkg.config import load_config
from riskcal_tkg.data import (
    add_inverse_relations,
    load_configured_table,
    split_calibration_roles,
    temporal_split,
)
from riskcal_tkg.followup import (
    SCORE_NAMES,
    build_query_grouping,
    candidate_nonconformity,
    query_max_true_nonconformity,
    summarize_query_mask,
)
from riskcal_tkg.model import build_temporal_model


HISTORY_NAMES = ("static", "expanding", "rolling")


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _score_all_objects(
    model: torch.nn.Module,
    facts: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(facts), batch_size):
            queries = torch.as_tensor(
                facts[start : start + batch_size][:, [0, 1, 3]],
                dtype=torch.long,
                device=device,
            )
            outputs.append(model.score_all_objects(queries).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _condition_label(seed: int, deletion_rate: float) -> str:
    rate = f"{deletion_rate:.2f}".replace(".", "p")
    return f"seed{seed}_delete{rate}"


def _verified_checkpoint(run_root: Path, label: str) -> tuple[dict[str, Any], str]:
    marker_path = run_root / "conditions" / f"{label}.complete.json"
    checkpoint_path = run_root / "checkpoints" / f"{label}.ckpt"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError(f"condition is not complete: {marker_path}")
    digest = sha256_file(checkpoint_path)
    if digest != marker["artifacts"]["checkpoint"]:
        raise ValueError(f"checkpoint checksum mismatch: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("config_sha256", "dataset_sha256", "deletion_mask_sha256"):
        if payload.get(key) != marker.get(key):
            raise ValueError(f"checkpoint {key} does not match marker for {label}")
    return payload, digest


def _add_history_batch(
    pools: dict[str, dict[str, CalibrationPool]],
    timestamp: int,
    true_scores: dict[str, np.ndarray],
) -> None:
    for score_name, values in true_scores.items():
        pools[score_name]["expanding"].add(timestamp, values)
        pools[score_name]["rolling"].add(timestamp, values)


def _history_thresholds(
    static_threshold: float,
    pools: dict[str, CalibrationPool],
    timestamp: int,
    alpha: float,
) -> dict[str, float]:
    return {
        "static": static_threshold,
        "expanding": finite_sample_quantile(
            pools["expanding"].values_before(timestamp).values,
            alpha,
        ),
        "rolling": finite_sample_quantile(
            pools["rolling"].values_before(timestamp).values,
            alpha,
        ),
    }


def _evaluate_case_seed(
    *,
    case: dict[str, Any],
    seed: int,
    deletion_rate: float,
    target_coverage: float,
    temperature: float,
    rolling_window: int,
    matrix_root: Path,
    data_root: Path | None,
    repo_root: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config_path = repo_root / str(case["config"])
    config = load_config(config_path)
    source_config_sha256 = _sha256_json(_jsonable(asdict(config)))
    if data_root is not None:
        config = replace(config, data_path=data_root / config.data_mode)
    table = load_configured_table(config)
    split = temporal_split(
        table,
        train_fraction=config.train_fraction,
        calibration_fraction=config.calibration_fraction,
    )
    roles = split_calibration_roles(
        split.calibration,
        fractions=config.calibration_role_fractions,
    )
    run_root = matrix_root / str(case["run_relative"])
    label = _condition_label(seed, deletion_rate)
    checkpoint, checkpoint_sha256 = _verified_checkpoint(run_root, label)
    if source_config_sha256 != checkpoint["config_sha256"]:
        raise ValueError(
            f"local configuration does not match the source checkpoint: {config_path}"
        )
    model = build_temporal_model(
        config.model_name,
        len(table.entity_to_id),
        2 * len(table.relation_to_id),
        len(table.timestamp_to_id),
        config.embedding_dim,
        time_scale=int(float(checkpoint["time_scale"])),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device)

    alpha = 1.0 - target_coverage
    maximum_history = len(table.values) * 2 + 1
    pools = {
        score_name: {
            "expanding": CalibrationPool(maximum_history),
            "rolling": CalibrationPool(rolling_window),
        }
        for score_name in SCORE_NAMES
    }
    query_pools = {
        "expanding": CalibrationPool(maximum_history),
        "rolling": CalibrationPool(rolling_window),
    }
    static_values = {score_name: [] for score_name in SCORE_NAMES}
    static_query_values: list[np.ndarray] = []
    relation_count = len(table.relation_to_id)

    for timestamp in np.unique(roles.final_calibration.timestamps):
        raw = roles.final_calibration.values[
            roles.final_calibration.timestamps == timestamp
        ]
        facts = add_inverse_relations(raw, relation_count)
        scores = _score_all_objects(
            model,
            facts,
            batch_size=config.batch_size,
            device=device,
        )
        labels = facts[:, 2]
        grouping = build_query_grouping(facts)
        true_scores: dict[str, np.ndarray] = {}
        for score_name in SCORE_NAMES:
            candidate_values = candidate_nonconformity(
                scores,
                score_name,
                temperature=temperature,
            )
            values = candidate_values[np.arange(len(labels)), labels]
            true_scores[score_name] = values
            static_values[score_name].append(values)
            if score_name == "margin":
                query_values = query_max_true_nonconformity(
                    candidate_values,
                    facts,
                    grouping,
                )
                static_query_values.append(query_values)
                query_pools["expanding"].add(int(timestamp), query_values)
                query_pools["rolling"].add(int(timestamp), query_values)
        _add_history_batch(pools, int(timestamp), true_scores)

    static_thresholds = {
        score_name: finite_sample_quantile(np.concatenate(values), alpha)
        for score_name, values in static_values.items()
    }
    static_query_threshold = finite_sample_quantile(
        np.concatenate(static_query_values),
        alpha,
    )

    rows: list[dict[str, Any]] = []
    for timestamp in np.unique(split.test.timestamps):
        raw = split.test.values[split.test.timestamps == timestamp]
        facts = add_inverse_relations(raw, relation_count)
        scores = _score_all_objects(
            model,
            facts,
            batch_size=config.batch_size,
            device=device,
        )
        labels = facts[:, 2]
        grouping = build_query_grouping(facts)
        true_scores = {}
        margin_candidate_values: np.ndarray | None = None

        for score_name in SCORE_NAMES:
            candidate_values = candidate_nonconformity(
                scores,
                score_name,
                temperature=temperature,
            )
            true_scores[score_name] = candidate_values[
                np.arange(len(labels)), labels
            ]
            thresholds = _history_thresholds(
                static_thresholds[score_name],
                pools[score_name],
                int(timestamp),
                alpha,
            )
            unique_candidates = candidate_values[grouping.first_indices]
            for history in HISTORY_NAMES:
                summary = summarize_query_mask(
                    unique_candidates <= thresholds[history],
                    facts,
                    grouping,
                )
                rows.append(
                    {
                        "case": case["name"],
                        "dataset_mode": config.data_mode,
                        "model_name": config.model_name,
                        "negative_sampling": config.negative_sampling,
                        "seed": seed,
                        "deletion_rate": deletion_rate,
                        "timestamp": int(timestamp),
                        "objective": "label",
                        "score": score_name,
                        "history": history,
                        "method": f"label_{score_name}_{history}",
                        "threshold": thresholds[history],
                        **summary,
                    }
                )
            if score_name == "margin":
                margin_candidate_values = candidate_values

        if margin_candidate_values is None:
            raise RuntimeError("margin nonconformity was not evaluated")
        query_thresholds = _history_thresholds(
            static_query_threshold,
            query_pools,
            int(timestamp),
            alpha,
        )
        unique_margin = margin_candidate_values[grouping.first_indices]
        for history in HISTORY_NAMES:
            summary = summarize_query_mask(
                unique_margin <= query_thresholds[history],
                facts,
                grouping,
            )
            rows.append(
                {
                    "case": case["name"],
                    "dataset_mode": config.data_mode,
                    "model_name": config.model_name,
                    "negative_sampling": config.negative_sampling,
                    "seed": seed,
                    "deletion_rate": deletion_rate,
                    "timestamp": int(timestamp),
                    "objective": "query_max",
                    "score": "margin",
                    "history": history,
                    "method": f"query_max_margin_{history}",
                    "threshold": query_thresholds[history],
                    **summary,
                }
            )

        query_values = query_max_true_nonconformity(
            margin_candidate_values,
            facts,
            grouping,
        )
        _add_history_batch(pools, int(timestamp), true_scores)
        query_pools["expanding"].add(int(timestamp), query_values)
        query_pools["rolling"].add(int(timestamp), query_values)

    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    provenance = {
        "case": case["name"],
        "seed": seed,
        "deletion_rate": deletion_rate,
        "config": str(case["config"]),
        "config_file_sha256": sha256_file(config_path),
        "resolved_config_sha256": source_config_sha256,
        "run_root": str(run_root),
        "source_checkpoint": str(run_root / "checkpoints" / f"{label}.ckpt"),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_dataset_sha256": checkpoint["dataset_sha256"],
        "timestamp_count": int(len(np.unique(split.test.timestamps))),
    }
    return pd.DataFrame(rows), provenance


def aggregate_by_seed(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "case",
        "dataset_mode",
        "model_name",
        "negative_sampling",
        "seed",
        "deletion_rate",
        "objective",
        "score",
        "history",
        "method",
    ]
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(keys, sort=True, dropna=False):
        record = dict(zip(keys, key, strict=True))
        label_total = group["label_count"].sum()
        query_total = group["query_count"].sum()
        single_total = group["single_query_count"].sum()
        multi_total = group["multi_query_count"].sum()
        record.update(
            {
                "label_count": int(label_total),
                "query_count": int(query_total),
                "single_query_count": int(single_total),
                "multi_query_count": int(multi_total),
                "label_coverage": float(
                    np.average(group["label_coverage"], weights=group["label_count"])
                ),
                "full_set_coverage": float(
                    np.average(group["full_set_coverage"], weights=group["query_count"])
                ),
                "partial_answer_recall": float(
                    np.average(group["partial_answer_recall"], weights=group["query_count"])
                ),
                "mean_set_size": float(
                    np.average(group["mean_set_size"], weights=group["query_count"])
                ),
                "single_full_set_coverage": (
                    float(np.average(group["single_full_set_coverage"], weights=group["single_query_count"]))
                    if single_total
                    else float("nan")
                ),
                "multi_full_set_coverage": (
                    float(np.average(group["multi_full_set_coverage"], weights=group["multi_query_count"]))
                    if multi_total
                    else float("nan")
                ),
                "single_mean_set_size": (
                    float(np.average(group["single_mean_set_size"], weights=group["single_query_count"]))
                    if single_total
                    else float("nan")
                ),
                "multi_mean_set_size": (
                    float(np.average(group["multi_mean_set_size"], weights=group["multi_query_count"]))
                    if multi_total
                    else float("nan")
                ),
            }
        )
        record["multi_minus_single_full_set_coverage"] = (
            record["multi_full_set_coverage"] - record["single_full_set_coverage"]
        )
        records.append(record)
    return pd.DataFrame(records)


def build_query_objective_contrasts(by_seed: pd.DataFrame) -> pd.DataFrame:
    selected = by_seed[
        (by_seed["score"] == "margin")
        & (by_seed["history"].isin(HISTORY_NAMES))
        & (by_seed["objective"].isin(("label", "query_max")))
    ]
    index = ["case", "dataset_mode", "model_name", "seed", "history"]
    wide = selected.pivot(index=index, columns="objective")
    records = wide.index.to_frame(index=False)
    for metric in (
        "label_coverage",
        "full_set_coverage",
        "partial_answer_recall",
        "mean_set_size",
        "multi_minus_single_full_set_coverage",
    ):
        records[f"{metric}_query_minus_label"] = (
            wide[(metric, "query_max")].to_numpy()
            - wide[(metric, "label")].to_numpy()
        )
        records[f"{metric}_label"] = wide[(metric, "label")].to_numpy()
        records[f"{metric}_query_max"] = wide[(metric, "query_max")].to_numpy()
    return records


def export_study(
    spec_path: Path,
    matrix_root: Path,
    output_root: Path,
    *,
    data_root: Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    resolved_device = (
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if resolved_device == "auto":
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    all_rows: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    for case in spec["cases"]:
        for seed in spec["seeds"]:
            print(f"[followup] case={case['name']} seed={seed}", flush=True)
            frame, record = _evaluate_case_seed(
                case=case,
                seed=int(seed),
                deletion_rate=float(spec["deletion_rate"]),
                target_coverage=float(spec["target_coverage"]),
                temperature=float(spec["temperature"]),
                rolling_window=int(spec["rolling_window"]),
                matrix_root=matrix_root,
                data_root=data_root,
                repo_root=repo_root,
                device=device,
            )
            all_rows.append(frame)
            provenance.append(record)
            _atomic_csv(
                output_root / "conditions" / f"{case['name']}_seed{seed}.csv",
                frame,
            )

    timestamp_rows = pd.concat(all_rows, ignore_index=True)
    by_seed = aggregate_by_seed(timestamp_rows)
    contrasts = build_query_objective_contrasts(by_seed)
    _atomic_csv(output_root / "followup_by_timestamp.csv", timestamp_rows)
    _atomic_csv(output_root / "followup_by_seed.csv", by_seed)
    _atomic_csv(output_root / "query_objective_contrasts_by_seed.csv", contrasts)

    manifest: dict[str, Any] = {
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(repo_root),
        "spec": spec,
        "spec_sha256": sha256_file(spec_path),
        "script_sha256": sha256_file(Path(__file__)),
        "device": str(device),
        "condition_count": len(provenance),
        "timestamp_row_count": len(timestamp_rows),
        "provenance": provenance,
    }
    manifest["provenance_sha256"] = _sha256_json(provenance)
    _atomic_json(output_root / "followup_manifest.json", manifest)
    checksums = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.relative_to(output_root)}")
    (output_root / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    manifest = export_study(
        arguments.spec,
        arguments.matrix_root,
        arguments.output_root,
        data_root=arguments.data_root,
        device_name=arguments.device,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
