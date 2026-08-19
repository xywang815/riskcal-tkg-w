"""Export relation-conditional and worst-group calibration diagnostics.

The confirmatory run records one row per observed query-label pair.  This
exporter reuses that artifact and computes relation-side slices without
retraining the scorer.  The main purpose is to check whether average
observed-label coverage hides severely under-covered relation groups.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_query_level_diagnostics import (
    QUERY_KEY,
    build_query_level_rows,
)


METHOD_ORDER = ["static", "rolling", "weighted", "adaptive", "top1"]
PAPER_METHODS = ["static", "rolling", "weighted", "adaptive"]
GROUP_KEYS = [
    "seed",
    "deletion_rate",
    "method",
    "prediction_side",
    "base_relation_id",
    "relation_label",
]


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


def _timestamp_key(value: str) -> tuple[int, Decimal | str]:
    try:
        return (0, Decimal(value))
    except InvalidOperation:
        return (1, value)


def relation_labels_from_raw(data_root: Path) -> dict[int, str]:
    """Rebuild the loader's relation-ID to raw-label mapping."""
    labels: set[str] = set()
    for name in ("train.txt", "valid.txt", "test.txt"):
        path = data_root / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 4:
                    raise ValueError(f"{path}:{line_number}: expected four columns")
                labels.add(fields[1])
    # The project loader sorts relation labels lexicographically.
    ordered = sorted(labels)
    return {index: label for index, label in enumerate(ordered)}


def _validate_per_query(rows: pd.DataFrame) -> None:
    required = {
        *QUERY_KEY,
        "true_object_id",
        "set_size",
        "covered",
        "rank",
        "top1_correct",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"per-query metrics are missing columns: {missing}")
    if rows.empty:
        raise ValueError("per-query metrics are empty")
    if (rows["set_size"] < 0).any():
        raise ValueError("set_size must be nonnegative")


def add_relation_metadata(
    rows: pd.DataFrame,
    *,
    num_relations: int,
    relation_labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Add original-relation IDs and labels to object and subject queries."""
    if num_relations <= 0:
        raise ValueError("num_relations must be positive")
    if "relation_id" not in rows.columns:
        raise ValueError("rows must contain relation_id")
    result = rows.copy()
    relation_ids = result["relation_id"].astype(int)
    result["base_relation_id"] = np.where(
        relation_ids >= num_relations,
        relation_ids - num_relations,
        relation_ids,
    ).astype(int)
    if (result["base_relation_id"] < 0).any() or (
        result["base_relation_id"] >= num_relations
    ).any():
        bad = result.loc[
            (result["base_relation_id"] < 0)
            | (result["base_relation_id"] >= num_relations),
            "relation_id",
        ].head(5)
        raise ValueError(f"relation_id values exceed the relation vocabulary: {bad.tolist()}")
    labels = relation_labels or {}
    result["relation_label"] = result["base_relation_id"].map(
        lambda value: labels.get(int(value), str(int(value)))
    )
    result["relation_side"] = (
        result["prediction_side"].astype(str)
        + ":"
        + result["relation_label"].astype(str)
    )
    return result


def _method_sort(frame: pd.DataFrame, prefix: list[str]) -> pd.DataFrame:
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result = frame.copy()
    result["_method_rank"] = result["method"].map(method_rank).fillna(len(method_rank))
    sort_columns = [
        column
        for column in [
            *prefix,
            "_method_rank",
            "method",
            "prediction_side",
            "base_relation_id",
        ]
        if column in result.columns
    ]
    return result.sort_values(
        sort_columns,
        kind="stable",
    ).drop(columns="_method_rank").reset_index(drop=True)


def build_relation_slice_by_seed(
    rows: pd.DataFrame,
    *,
    target_coverage: float,
    num_entities: int | None = None,
    num_relations: int,
    relation_labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build seed-level observed-label and query-level relation-side summaries."""
    _validate_per_query(rows)
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between 0 and 1")
    if num_entities is not None and num_entities <= 0:
        raise ValueError("num_entities must be positive")

    labeled = add_relation_metadata(
        rows,
        num_relations=num_relations,
        relation_labels=relation_labels,
    )
    labeled["covered"] = labeled["covered"].astype(bool)
    labeled["top1_correct"] = labeled["top1_correct"].astype(bool)
    labeled["reciprocal_rank"] = 1.0 / labeled["rank"].astype(float)

    label_summary = (
        labeled.groupby(GROUP_KEYS, as_index=False, sort=False)
        .agg(
            label_count=("covered", "size"),
            query_count=("subject_id", "size"),
            unique_query_count=("subject_id", lambda values: np.nan),
            label_coverage=("covered", "mean"),
            mean_size=("set_size", "mean"),
            median_size=("set_size", "median"),
            p90_size=("set_size", lambda values: float(values.quantile(0.9))),
            mrr=("reciprocal_rank", "mean"),
            top1_accuracy=("top1_correct", "mean"),
        )
        .reset_index(drop=True)
    )
    unique_counts = (
        labeled.drop_duplicates(QUERY_KEY)
        .groupby(GROUP_KEYS, as_index=False, sort=False)
        .size()
        .rename(columns={"size": "unique_query_count"})
    )
    label_summary = label_summary.drop(columns="unique_query_count").merge(
        unique_counts,
        on=GROUP_KEYS,
        how="left",
        validate="one_to_one",
    )
    label_summary["positive_undercoverage"] = np.maximum(
        target_coverage - label_summary["label_coverage"], 0.0
    )
    if num_entities is not None:
        vocab = int(num_entities)
        full_vocab = (
            labeled.assign(full_vocabulary_set=labeled["set_size"] >= vocab)
            .groupby(GROUP_KEYS, as_index=False, sort=False)
            .agg(full_vocabulary_set_rate=("full_vocabulary_set", "mean"))
        )
        label_summary = label_summary.merge(
            full_vocab,
            on=GROUP_KEYS,
            how="left",
            validate="one_to_one",
        )
        label_summary["normalized_mean_size"] = label_summary["mean_size"] / vocab

    queries = add_relation_metadata(
        build_query_level_rows(rows),
        num_relations=num_relations,
        relation_labels=relation_labels,
    )
    query_summary = (
        queries.groupby(GROUP_KEYS, as_index=False, sort=False)
        .agg(
            query_full_set_coverage=("full_set_covered", "mean"),
            query_partial_answer_recall=("partial_answer_recall", "mean"),
            multi_answer_query_fraction=("multi_answer", "mean"),
            max_answer_count=("answer_count", "max"),
        )
        .reset_index(drop=True)
    )

    result = label_summary.merge(
        query_summary,
        on=GROUP_KEYS,
        how="left",
        validate="one_to_one",
    )
    return _method_sort(result, ["seed", "deletion_rate"])


def aggregate_relation_slices(by_seed: pd.DataFrame) -> pd.DataFrame:
    """Average relation-side summaries over seeds."""
    metrics = [
        "label_coverage",
        "positive_undercoverage",
        "mean_size",
        "median_size",
        "p90_size",
        "mrr",
        "top1_accuracy",
        "full_vocabulary_set_rate",
        "normalized_mean_size",
        "query_full_set_coverage",
        "query_partial_answer_recall",
        "multi_answer_query_fraction",
        "max_answer_count",
    ]
    present = [metric for metric in metrics if metric in by_seed.columns]
    records: list[dict[str, Any]] = []
    group_columns = [
        "deletion_rate",
        "method",
        "prediction_side",
        "base_relation_id",
        "relation_label",
    ]
    for keys, frame in by_seed.groupby(group_columns, sort=False):
        deletion_rate, method, prediction_side, base_relation_id, relation_label = keys
        record: dict[str, Any] = {
            "deletion_rate": float(deletion_rate),
            "method": str(method),
            "prediction_side": str(prediction_side),
            "base_relation_id": int(base_relation_id),
            "relation_label": str(relation_label),
            "seed_count": int(frame["seed"].nunique()),
            "label_count_total": int(frame["label_count"].sum()),
            "label_count_mean": float(frame["label_count"].mean()),
            "unique_query_count_total": int(frame["unique_query_count"].sum()),
        }
        for metric in present:
            values = frame[metric].dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        records.append(record)
    return _method_sort(pd.DataFrame(records), ["deletion_rate"])


def build_worst_group_summary(
    relation_summary: pd.DataFrame,
    *,
    target_coverage: float,
    min_total_labels: int,
    min_seed_count: int,
) -> pd.DataFrame:
    """Summarize the worst eligible relation-side slices per condition."""
    if min_total_labels <= 0:
        raise ValueError("min_total_labels must be positive")
    if min_seed_count <= 0:
        raise ValueError("min_seed_count must be positive")
    required = {
        "deletion_rate",
        "method",
        "prediction_side",
        "base_relation_id",
        "relation_label",
        "seed_count",
        "label_count_total",
        "label_coverage_mean",
        "positive_undercoverage_mean",
    }
    missing = sorted(required - set(relation_summary.columns))
    if missing:
        raise ValueError(f"relation summary is missing columns: {missing}")

    records: list[dict[str, Any]] = []
    for (deletion_rate, method), frame in relation_summary.groupby(
        ["deletion_rate", "method"], sort=False
    ):
        eligible = frame[
            (frame["label_count_total"] >= min_total_labels)
            & (frame["seed_count"] >= min_seed_count)
        ].copy()
        record: dict[str, Any] = {
            "deletion_rate": float(deletion_rate),
            "method": str(method),
            "min_total_labels": int(min_total_labels),
            "min_seed_count": int(min_seed_count),
            "eligible_relation_side_groups": int(len(eligible)),
        }
        if eligible.empty:
            record.update(
                {
                    "relation_side_coverage_min": np.nan,
                    "relation_side_coverage_p10": np.nan,
                    "relation_side_coverage_median": np.nan,
                    "fraction_relation_side_below_target": np.nan,
                    "mean_relation_side_undercoverage": np.nan,
                    "worst_relation_side_undercoverage": np.nan,
                    "worst_prediction_side": "",
                    "worst_base_relation_id": np.nan,
                    "worst_relation_label": "",
                    "worst_label_count_total": np.nan,
                    "worst_mean_size": np.nan,
                    "worst_query_full_set_coverage": np.nan,
                }
            )
        else:
            coverage = eligible["label_coverage_mean"]
            worst = eligible.sort_values(
                ["label_coverage_mean", "label_count_total"],
                ascending=[True, False],
                kind="stable",
            ).iloc[0]
            record.update(
                {
                    "relation_side_coverage_min": float(coverage.min()),
                    "relation_side_coverage_p10": float(coverage.quantile(0.10)),
                    "relation_side_coverage_median": float(coverage.median()),
                    "fraction_relation_side_below_target": float(
                        (coverage < target_coverage).mean()
                    ),
                    "mean_relation_side_undercoverage": float(
                        np.maximum(target_coverage - coverage, 0.0).mean()
                    ),
                    "worst_relation_side_undercoverage": float(
                        max(target_coverage - float(worst["label_coverage_mean"]), 0.0)
                    ),
                    "worst_prediction_side": str(worst["prediction_side"]),
                    "worst_base_relation_id": int(worst["base_relation_id"]),
                    "worst_relation_label": str(worst["relation_label"]),
                    "worst_label_count_total": int(worst["label_count_total"]),
                    "worst_mean_size": float(worst.get("mean_size_mean", np.nan)),
                    "worst_query_full_set_coverage": float(
                        worst.get("query_full_set_coverage_mean", np.nan)
                    ),
                }
            )
        records.append(record)
    return _method_sort(pd.DataFrame(records), ["deletion_rate"])


def build_paper_table(worst_summary: pd.DataFrame) -> pd.DataFrame:
    selected = worst_summary[
        worst_summary["method"].isin(PAPER_METHODS)
        & worst_summary["deletion_rate"].isin([0.0, 0.3])
    ].copy()
    columns = [
        "deletion_rate",
        "method",
        "eligible_relation_side_groups",
        "relation_side_coverage_min",
        "relation_side_coverage_p10",
        "relation_side_coverage_median",
        "fraction_relation_side_below_target",
        "worst_relation_side_undercoverage",
        "worst_prediction_side",
        "worst_base_relation_id",
        "worst_relation_label",
        "worst_label_count_total",
        "worst_query_full_set_coverage",
    ]
    return selected[[column for column in columns if column in selected.columns]]


def read_per_query(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as error:
        raise SystemExit(
            "Reading parquet requires pyarrow or fastparquet. Run this script on "
            "the server environment used for the experiment, or install pyarrow "
            "locally with: python -m pip install pyarrow"
        ) from error


def export_relation_slice_diagnostics(
    run_root: Path,
    paper_root: Path,
    *,
    output_name: str = "final_confirmatory",
    data_root: Path | None = None,
    target_coverage: float = 0.90,
    min_total_labels: int = 250,
    min_seed_count: int = 5,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    paper_root = paper_root.resolve()
    per_query_path = run_root / "metrics" / "per_query.parquet"
    if not per_query_path.is_file():
        raise FileNotFoundError(f"missing per-query artifact: {per_query_path}")

    manifest_path = run_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    num_entities = int(manifest["num_entities"])
    num_relations = int(manifest["num_relations"])

    relation_labels: dict[int, str] = {}
    if data_root is not None:
        relation_labels = relation_labels_from_raw(data_root)
        if relation_labels and len(relation_labels) != num_relations:
            raise ValueError(
                "raw relation-label count does not match dataset manifest: "
                f"{len(relation_labels)} != {num_relations}"
            )

    rows = read_per_query(per_query_path)
    by_seed = build_relation_slice_by_seed(
        rows,
        target_coverage=target_coverage,
        num_entities=num_entities,
        num_relations=num_relations,
        relation_labels=relation_labels,
    )
    relation_summary = aggregate_relation_slices(by_seed)
    worst_summary = build_worst_group_summary(
        relation_summary,
        target_coverage=target_coverage,
        min_total_labels=min_total_labels,
        min_seed_count=min_seed_count,
    )
    paper_table = build_paper_table(worst_summary)

    data_dir = paper_root / "data" / output_name
    _write_csv(by_seed, data_dir / "relation_slice_by_seed.csv")
    _write_csv(relation_summary, data_dir / "relation_slice_summary.csv")
    _write_csv(worst_summary, data_dir / "relation_worst_group_summary.csv")
    _write_csv(paper_table, data_dir / "relation_worst_group_paper_table.csv")

    outputs = {
        "relation_slice_by_seed.csv": sha256_file(
            data_dir / "relation_slice_by_seed.csv"
        ),
        "relation_slice_summary.csv": sha256_file(
            data_dir / "relation_slice_summary.csv"
        ),
        "relation_worst_group_summary.csv": sha256_file(
            data_dir / "relation_worst_group_summary.csv"
        ),
        "relation_worst_group_paper_table.csv": sha256_file(
            data_dir / "relation_worst_group_paper_table.csv"
        ),
    }
    manifest_out = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "source_per_query": str(per_query_path),
        "source_per_query_sha256": sha256_file(per_query_path),
        "target_coverage": float(target_coverage),
        "min_total_labels": int(min_total_labels),
        "min_seed_count": int(min_seed_count),
        "num_entities": int(num_entities),
        "num_relations": int(num_relations),
        "label_rows": int(len(rows)),
        "seed_relation_side_rows": int(len(by_seed)),
        "relation_side_rows": int(len(relation_summary)),
        "worst_group_rows": int(len(worst_summary)),
        "paper_table_rows": int(len(paper_table)),
        "group_key": GROUP_KEYS,
        "definition": {
            "relation_side": (
                "The original relation ID plus prediction side. Subject queries "
                "are mapped from inverse-relation IDs back to the base relation."
            ),
            "label_coverage": (
                "Observed-label coverage within a relation-side slice."
            ),
            "eligible_relation_side_groups": (
                "Relation-side groups with at least min_total_labels observed "
                "labels after aggregating over seeds and at least min_seed_count seeds."
            ),
            "relation_side_coverage_min": (
                "Minimum mean observed-label coverage across eligible relation-side groups."
            ),
        },
        "outputs": outputs,
    }
    _write_json(manifest_out, data_dir / "relation_slice_manifest.json")
    return manifest_out


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--output-name", default="final_confirmatory")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/icews14"))
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--min-total-labels", type=int, default=250)
    parser.add_argument("--min-seed-count", type=int, default=5)
    args = parser.parse_args()
    manifest = export_relation_slice_diagnostics(
        args.run_root,
        args.paper_root,
        output_name=args.output_name,
        data_root=args.data_root,
        target_coverage=args.target_coverage,
        min_total_labels=args.min_total_labels,
        min_seed_count=args.min_seed_count,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
