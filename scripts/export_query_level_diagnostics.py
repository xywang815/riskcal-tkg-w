"""Export query-level multi-answer diagnostics from per-query artifacts.

The main experiment records one row per observed query-label pair.  This script
groups those rows back into unique TKG queries, so that multi-answer behavior
can be reported separately from label-marginal observed-label coverage.
"""

from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUERY_KEY = [
    "seed",
    "deletion_rate",
    "method",
    "prediction_side",
    "timestamp",
    "subject_id",
    "relation_id",
]
LABEL_KEY = [*QUERY_KEY, "true_object_id"]
METHOD_ORDER = ["static", "rolling", "weighted", "adaptive", "top1"]
PAPER_METHODS = ["static", "rolling", "weighted", "adaptive"]


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


def _method_sort(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result = frame.copy()
    result["_method_rank"] = result["method"].map(method_rank).fillna(len(method_rank))
    return result.sort_values(
        [*columns, "_method_rank"], kind="stable"
    ).drop(columns="_method_rank").reset_index(drop=True)


def _validate_per_query(rows: pd.DataFrame) -> None:
    required = {
        *LABEL_KEY,
        "set_size",
        "covered",
        "rank",
        "frequency_rank",
        "top1_correct",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"per-query metrics are missing columns: {missing}")
    if rows.empty:
        raise ValueError("per-query metrics are empty")
    if (rows["set_size"] < 0).any():
        raise ValueError("set_size must be nonnegative")


def build_query_level_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse observed labels into one record per unique forecasting query."""
    _validate_per_query(rows)
    normalized = rows.copy()
    normalized["covered"] = normalized["covered"].astype(bool)
    normalized["top1_correct"] = normalized["top1_correct"].astype(bool)

    # Deduplicate repeated identical observed answers while preserving whether
    # that answer was covered. Full-set coverage is defined over unique labels.
    labels = (
        normalized.groupby(LABEL_KEY, as_index=False, sort=False)
        .agg(
            covered=("covered", "max"),
            set_size=("set_size", "first"),
            set_size_nunique=("set_size", "nunique"),
            rank=("rank", "min"),
            frequency_rank=("frequency_rank", "min"),
            top1_correct=("top1_correct", "max"),
        )
        .reset_index(drop=True)
    )
    if (labels["set_size_nunique"] > 1).any():
        bad = labels.loc[labels["set_size_nunique"] > 1, QUERY_KEY].head(3)
        raise ValueError(
            "same query-label group has inconsistent set_size values: "
            f"{bad.to_dict(orient='records')}"
        )
    labels = labels.drop(columns="set_size_nunique")

    grouped = labels.groupby(QUERY_KEY, as_index=False, sort=False)
    queries = grouped.agg(
        answer_count=("true_object_id", "nunique"),
        covered_answer_count=("covered", "sum"),
        full_set_covered=("covered", "all"),
        partial_answer_recall=("covered", "mean"),
        set_size=("set_size", "first"),
        set_size_nunique=("set_size", "nunique"),
        best_rank=("rank", "min"),
        mean_rank=("rank", "mean"),
        best_frequency_rank=("frequency_rank", "min"),
        any_top1_correct=("top1_correct", "max"),
    )
    if (queries["set_size_nunique"] > 1).any():
        bad = queries.loc[queries["set_size_nunique"] > 1, QUERY_KEY].head(3)
        raise ValueError(
            "same query has inconsistent set_size values across labels: "
            f"{bad.to_dict(orient='records')}"
        )
    queries = queries.drop(columns="set_size_nunique")
    queries["multi_answer"] = queries["answer_count"] > 1
    queries["full_set_covered"] = queries["full_set_covered"].astype(bool)
    queries["covered_answer_count"] = queries["covered_answer_count"].astype(int)
    return _method_sort(
        queries,
        ["seed", "deletion_rate", "timestamp", "prediction_side", "subject_id", "relation_id"],
    )


def summarize_query_level(
    queries: pd.DataFrame,
    *,
    num_entities: int | None = None,
) -> pd.DataFrame:
    """Summarize query-level diagnostics per seed, deletion rate, and method."""
    required = {
        *QUERY_KEY,
        "answer_count",
        "covered_answer_count",
        "full_set_covered",
        "partial_answer_recall",
        "set_size",
        "multi_answer",
    }
    missing = sorted(required - set(queries.columns))
    if missing:
        raise ValueError(f"query-level rows are missing columns: {missing}")
    if queries.empty:
        raise ValueError("query-level rows are empty")

    records: list[dict[str, Any]] = []
    for (seed, deletion_rate, method), frame in queries.groupby(
        ["seed", "deletion_rate", "method"], sort=False
    ):
        multi = frame[frame["multi_answer"]]
        label_count = int(frame["answer_count"].sum())
        covered_label_count = int(frame["covered_answer_count"].sum())
        record: dict[str, Any] = {
            "seed": int(seed),
            "deletion_rate": float(deletion_rate),
            "method": str(method),
            "query_count": int(len(frame)),
            "label_count": label_count,
            "multi_answer_query_count": int(len(multi)),
            "multi_answer_query_fraction": float(len(multi) / len(frame)),
            "mean_answer_count": float(frame["answer_count"].mean()),
            "max_answer_count": int(frame["answer_count"].max()),
            "full_set_coverage": float(frame["full_set_covered"].mean()),
            "partial_answer_recall": float(frame["partial_answer_recall"].mean()),
            "label_weighted_partial_recall": float(
                covered_label_count / label_count if label_count else np.nan
            ),
            "mean_size": float(frame["set_size"].mean()),
            "median_size": float(frame["set_size"].median()),
            "p90_size": float(frame["set_size"].quantile(0.9)),
            "singleton_rate": float((frame["set_size"] == 1).mean()),
        }
        if len(multi):
            record.update(
                {
                    "multi_answer_full_set_coverage": float(
                        multi["full_set_covered"].mean()
                    ),
                    "multi_answer_partial_recall": float(
                        multi["partial_answer_recall"].mean()
                    ),
                    "multi_answer_mean_answer_count": float(
                        multi["answer_count"].mean()
                    ),
                }
            )
        else:
            record.update(
                {
                    "multi_answer_full_set_coverage": np.nan,
                    "multi_answer_partial_recall": np.nan,
                    "multi_answer_mean_answer_count": np.nan,
                }
            )
        if num_entities is not None:
            if num_entities <= 0:
                raise ValueError("num_entities must be positive")
            record["full_vocabulary_set_rate"] = float(
                (frame["set_size"] >= num_entities).mean()
            )
            record["normalized_mean_size"] = float(frame["set_size"].mean() / num_entities)
            record["normalized_p90_size"] = float(frame["set_size"].quantile(0.9) / num_entities)
        records.append(record)

    result = pd.DataFrame(records)
    return _method_sort(result, ["seed", "deletion_rate"])


def aggregate_query_level(summary: pd.DataFrame) -> pd.DataFrame:
    """Average condition summaries over seeds for paper-facing tables."""
    records: list[dict[str, Any]] = []
    metrics = [
        "query_count",
        "label_count",
        "multi_answer_query_fraction",
        "mean_answer_count",
        "max_answer_count",
        "full_set_coverage",
        "partial_answer_recall",
        "label_weighted_partial_recall",
        "multi_answer_full_set_coverage",
        "multi_answer_partial_recall",
        "mean_size",
        "median_size",
        "p90_size",
        "singleton_rate",
        "full_vocabulary_set_rate",
        "normalized_mean_size",
        "normalized_p90_size",
    ]
    present = [metric for metric in metrics if metric in summary.columns]
    for (deletion_rate, method), frame in summary.groupby(
        ["deletion_rate", "method"], sort=False
    ):
        record: dict[str, Any] = {
            "deletion_rate": float(deletion_rate),
            "method": str(method),
            "seed_count": int(frame["seed"].nunique()),
        }
        for metric in present:
            values = frame[metric].dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        records.append(record)
    result = pd.DataFrame(records)
    return _method_sort(result, ["deletion_rate"])


def build_paper_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    selected = aggregate[
        aggregate["method"].isin(PAPER_METHODS)
        & aggregate["deletion_rate"].isin([0.0, 0.3])
    ].copy()
    columns = [
        "deletion_rate",
        "method",
        "full_set_coverage_mean",
        "partial_answer_recall_mean",
        "multi_answer_query_fraction_mean",
        "multi_answer_full_set_coverage_mean",
        "multi_answer_partial_recall_mean",
        "p90_size_mean",
        "full_vocabulary_set_rate_mean",
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


def export_query_level_diagnostics(
    run_root: Path,
    paper_root: Path,
    *,
    output_name: str = "final_confirmatory",
) -> dict[str, Any]:
    run_root = run_root.resolve()
    paper_root = paper_root.resolve()
    per_query_path = run_root / "metrics" / "per_query.parquet"
    if not per_query_path.is_file():
        raise FileNotFoundError(f"missing per-query artifact: {per_query_path}")

    manifest_path = run_root / "dataset_manifest.json"
    num_entities = None
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if "num_entities" in manifest:
            num_entities = int(manifest["num_entities"])

    rows = read_per_query(per_query_path)
    queries = build_query_level_rows(rows)
    summary = summarize_query_level(queries, num_entities=num_entities)
    aggregate = aggregate_query_level(summary)
    paper_table = build_paper_table(aggregate)

    data_dir = paper_root / "data" / output_name
    _write_csv(summary, data_dir / "query_level_by_seed.csv")
    _write_csv(aggregate, data_dir / "query_level_summary.csv")
    _write_csv(paper_table, data_dir / "query_level_paper_table.csv")

    outputs = {
        "query_level_by_seed.csv": sha256_file(data_dir / "query_level_by_seed.csv"),
        "query_level_summary.csv": sha256_file(data_dir / "query_level_summary.csv"),
        "query_level_paper_table.csv": sha256_file(
            data_dir / "query_level_paper_table.csv"
        ),
    }
    manifest_out = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "source_per_query": str(per_query_path),
        "source_per_query_sha256": sha256_file(per_query_path),
        "num_entities": num_entities,
        "label_rows": int(len(rows)),
        "unique_query_rows": int(len(queries)),
        "condition_rows": int(len(summary)),
        "aggregate_rows": int(len(aggregate)),
        "query_key": QUERY_KEY,
        "definition": {
            "full_set_coverage": (
                "Mean over unique queries of whether every recorded answer for "
                "that query is included in the prediction set."
            ),
            "partial_answer_recall": (
                "Mean over unique queries of the fraction of recorded answers "
                "included in the prediction set."
            ),
            "label_weighted_partial_recall": (
                "Covered observed labels divided by observed labels; this should "
                "match label-marginal coverage up to duplicate handling."
            ),
            "full_vocabulary_set_rate": (
                "Fraction of unique queries whose prediction-set size is at least "
                "the closed entity vocabulary size."
            ),
        },
        "outputs": outputs,
    }
    _write_json(manifest_out, data_dir / "query_level_manifest.json")
    manifest_out["outputs"]["query_level_manifest.json"] = sha256_file(
        data_dir / "query_level_manifest.json"
    )
    _write_json(manifest_out, data_dir / "query_level_manifest.json")
    return manifest_out


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--output-name", default="final_confirmatory")
    args = parser.parse_args()
    manifest = export_query_level_diagnostics(
        args.run_root,
        args.paper_root,
        output_name=args.output_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
