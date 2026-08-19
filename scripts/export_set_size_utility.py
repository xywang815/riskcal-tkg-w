"""Export set-size and utility operating-point diagnostics.

This exporter reuses the completed per-query artifact.  It does not retrain the
scorer and it does not change the conformal threshold.  Instead, it treats a
maximum prediction-set size as an explicit abstention rule: queries whose set is
larger than the cap are not answered.  The resulting tables quantify the
coverage retained, candidate-load reduction, and abstention rate for each cap.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.export_query_level_diagnostics import (  # noqa: E402
    QUERY_KEY,
    build_query_level_rows,
    read_per_query,
    sha256_file,
)
from scripts.export_timestamp_block_bootstrap import _bootstrap_statistic  # noqa: E402


DEFAULT_CAPS = (1, 10, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, math.inf)
DEFAULT_PAPER_CAPS = (500, 1000, 2000, 3000, 4000, 5000, math.inf)
DEFAULT_METHODS = ("static", "rolling", "weighted", "adaptive")
DEFAULT_BOOTSTRAP_SEED = 20260818
DEFAULT_BLOCK_LENGTH = 7
DEFAULT_ITERATIONS = 20_000
METHOD_ORDER = ["static", "rolling", "weighted", "adaptive", "top1"]


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


def _parse_cap(token: str) -> float:
    normalized = token.strip().lower()
    if normalized in {"inf", "infinity", "none", "uncapped", "all"}:
        return math.inf
    value = int(normalized)
    if value < 0:
        raise ValueError("set-size caps must be nonnegative")
    return float(value)


def parse_caps(value: str | Iterable[int | float]) -> tuple[float, ...]:
    if isinstance(value, str):
        raw = [_parse_cap(token) for token in value.split(",") if token.strip()]
    else:
        raw = [float(item) for item in value]
    if not raw:
        raise ValueError("at least one cap is required")
    if any(cap < 0 or math.isnan(cap) for cap in raw):
        raise ValueError("set-size caps must be finite nonnegative values or inf")
    unique = sorted(set(raw), key=lambda cap: (math.isinf(cap), cap))
    return tuple(unique)


def _cap_label(cap: float) -> str:
    return "inf" if math.isinf(float(cap)) else str(int(cap))


def _method_sort(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    result = frame.copy()
    result["_method_rank"] = result["method"].map(rank).fillna(len(rank))
    wanted = [*columns, "_method_rank", "method", "cap_rank"]
    present: list[str] = []
    for column in wanted:
        if column in result.columns and column not in present:
            present.append(column)
    return result.sort_values(
        present, kind="stable"
    ).drop(
        columns=[column for column in ("_method_rank",) if column in result],
    ).reset_index(drop=True)


def _required_query_columns() -> set[str]:
    return {
        *QUERY_KEY,
        "answer_count",
        "covered_answer_count",
        "full_set_covered",
        "partial_answer_recall",
        "set_size",
        "any_top1_correct",
    }


def _validate_queries(queries: pd.DataFrame) -> None:
    missing = sorted(_required_query_columns() - set(queries.columns))
    if missing:
        raise ValueError(f"query-level rows are missing columns: {missing}")
    if queries.empty:
        raise ValueError("query-level rows are empty")
    if (queries["answer_count"] <= 0).any():
        raise ValueError("answer_count must be positive")
    if (queries["set_size"] < 0).any():
        raise ValueError("set_size must be nonnegative")


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _operating_point_record(
    frame: pd.DataFrame,
    *,
    cap: float,
    num_entities: int | None,
) -> dict[str, Any]:
    answered = np.ones(len(frame), dtype=bool) if math.isinf(cap) else (
        frame["set_size"].to_numpy(dtype=float) <= cap
    )
    answered_rows = frame.loc[answered]
    query_count = int(len(frame))
    answered_count = int(answered.sum())
    label_count = int(frame["answer_count"].sum())
    answered_label_count = int(answered_rows["answer_count"].sum())
    covered_answered_label_count = int(answered_rows["covered_answer_count"].sum())
    full_set_covered_count = int(answered_rows["full_set_covered"].sum())
    partial_recall_sum = float(answered_rows["partial_answer_recall"].sum())
    top1_answered_count = int(answered_rows["any_top1_correct"].sum())
    candidate_load_sum = float(answered_rows["set_size"].sum())
    full_vocab_answered_count = (
        0
        if num_entities is None
        else int((answered_rows["set_size"] >= int(num_entities)).sum())
    )

    record: dict[str, Any] = {
        "query_count": query_count,
        "label_count": label_count,
        "cap_size": "inf" if math.isinf(cap) else int(cap),
        "cap_label": _cap_label(cap),
        "cap_rank": 10**12 if math.isinf(cap) else int(cap),
        "answered_count": answered_count,
        "answer_rate": _safe_fraction(answered_count, query_count),
        "abstention_rate": 1.0 - _safe_fraction(answered_count, query_count),
        "answered_label_count": answered_label_count,
        "label_answer_rate": _safe_fraction(answered_label_count, label_count),
        "conditional_full_set_coverage": _safe_fraction(
            full_set_covered_count, answered_count
        ),
        "unconditional_full_set_recall": _safe_fraction(
            full_set_covered_count, query_count
        ),
        "conditional_partial_answer_recall": _safe_fraction(
            partial_recall_sum, answered_count
        ),
        "unconditional_partial_answer_recall": _safe_fraction(
            partial_recall_sum, query_count
        ),
        "conditional_label_recall": _safe_fraction(
            covered_answered_label_count, answered_label_count
        ),
        "unconditional_label_recall": _safe_fraction(
            covered_answered_label_count, label_count
        ),
        "candidate_load": _safe_fraction(candidate_load_sum, query_count),
        "mean_answered_set_size": _safe_fraction(candidate_load_sum, answered_count),
        "top1_accuracy_on_answered": _safe_fraction(top1_answered_count, answered_count),
        "full_vocabulary_answered_rate": _safe_fraction(
            full_vocab_answered_count, answered_count
        ),
    }
    if answered_count:
        sizes = answered_rows["set_size"].to_numpy(dtype=float)
        record["median_answered_set_size"] = float(np.median(sizes))
        record["p90_answered_set_size"] = float(np.quantile(sizes, 0.9))
    else:
        record["median_answered_set_size"] = float("nan")
        record["p90_answered_set_size"] = float("nan")
    return record


def summarize_operating_points(
    queries: pd.DataFrame,
    *,
    caps: tuple[float, ...],
    num_entities: int | None = None,
    by_timestamp: bool = False,
) -> pd.DataFrame:
    _validate_queries(queries)
    group_columns = ["seed", "deletion_rate", "method"]
    if by_timestamp:
        group_columns.append("timestamp")
    records: list[dict[str, Any]] = []
    for key, frame in queries.groupby(group_columns, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_columns, key, strict=True))
        for cap in caps:
            records.append(
                {
                    **base,
                    **_operating_point_record(
                        frame,
                        cap=cap,
                        num_entities=num_entities,
                    ),
                }
            )
    return _method_sort(pd.DataFrame(records), group_columns)


def add_uncapped_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    group_columns = ["seed", "deletion_rate", "method"]
    if "timestamp" in rows.columns:
        group_columns.append("timestamp")
    result = rows.copy()
    for column in (
        "candidate_load_saved_vs_uncapped",
        "candidate_load_reduction_vs_uncapped",
        "unconditional_full_set_recall_loss_vs_uncapped",
        "conditional_full_set_coverage_delta_vs_uncapped",
    ):
        result[column] = np.nan
    for _, index in result.groupby(group_columns, sort=False).groups.items():
        group = result.loc[index]
        uncapped = group[group["cap_label"] == "inf"]
        if len(uncapped) != 1:
            raise ValueError("each seed/deletion/method group must contain one uncapped row")
        base = uncapped.iloc[0]
        candidate_base = float(base["candidate_load"])
        result.loc[index, "candidate_load_saved_vs_uncapped"] = (
            candidate_base - group["candidate_load"].to_numpy(dtype=float)
        )
        result.loc[index, "candidate_load_reduction_vs_uncapped"] = (
            0.0
            if candidate_base <= 0
            else (candidate_base - group["candidate_load"].to_numpy(dtype=float))
            / candidate_base
        )
        result.loc[index, "unconditional_full_set_recall_loss_vs_uncapped"] = (
            float(base["unconditional_full_set_recall"])
            - group["unconditional_full_set_recall"].to_numpy(dtype=float)
        )
        result.loc[index, "conditional_full_set_coverage_delta_vs_uncapped"] = (
            group["conditional_full_set_coverage"].to_numpy(dtype=float)
            - float(base["conditional_full_set_coverage"])
        )
    return _method_sort(result, ["deletion_rate", *group_columns])


def aggregate_over_seeds(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "query_count",
        "label_count",
        "answered_count",
        "answer_rate",
        "abstention_rate",
        "label_answer_rate",
        "conditional_full_set_coverage",
        "unconditional_full_set_recall",
        "conditional_partial_answer_recall",
        "unconditional_partial_answer_recall",
        "conditional_label_recall",
        "unconditional_label_recall",
        "candidate_load",
        "mean_answered_set_size",
        "median_answered_set_size",
        "p90_answered_set_size",
        "top1_accuracy_on_answered",
        "full_vocabulary_answered_rate",
        "candidate_load_saved_vs_uncapped",
        "candidate_load_reduction_vs_uncapped",
        "unconditional_full_set_recall_loss_vs_uncapped",
        "conditional_full_set_coverage_delta_vs_uncapped",
    ]
    records: list[dict[str, Any]] = []
    for (deletion_rate, method, cap_label), frame in rows.groupby(
        ["deletion_rate", "method", "cap_label"], sort=False
    ):
        record: dict[str, Any] = {
            "deletion_rate": float(deletion_rate),
            "method": str(method),
            "cap_label": str(cap_label),
            "cap_size": frame["cap_size"].iloc[0],
            "cap_rank": int(frame["cap_rank"].iloc[0]),
            "seed_count": int(frame["seed"].nunique()),
        }
        for metric in metrics:
            values = frame[metric].dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        records.append(record)
    return _method_sort(pd.DataFrame(records), ["deletion_rate"])


def build_paper_table(
    summary: pd.DataFrame,
    *,
    deletion_rate: float,
    methods: tuple[str, ...],
    caps: tuple[float, ...],
) -> pd.DataFrame:
    cap_labels = {_cap_label(cap) for cap in caps}
    selected = summary[
        (summary["deletion_rate"] == deletion_rate)
        & summary["method"].isin(methods)
        & summary["cap_label"].isin(cap_labels)
    ].copy()
    columns = [
        "deletion_rate",
        "method",
        "cap_label",
        "seed_count",
        "answer_rate_mean",
        "conditional_full_set_coverage_mean",
        "unconditional_full_set_recall_mean",
        "candidate_load_mean",
        "candidate_load_reduction_vs_uncapped_mean",
        "mean_answered_set_size_mean",
        "p90_answered_set_size_mean",
        "top1_accuracy_on_answered_mean",
        "full_vocabulary_answered_rate_mean",
    ]
    return _method_sort(
        selected[[column for column in columns if column in selected.columns]],
        ["deletion_rate"],
    )


def bootstrap_utility_effects(
    timestamp_rows: pd.DataFrame,
    *,
    deletion_rate: float,
    method: str,
    caps: tuple[float, ...],
    block_length: int,
    iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    selected = timestamp_rows[
        (timestamp_rows["deletion_rate"] == deletion_rate)
        & (timestamp_rows["method"] == method)
        & (timestamp_rows["cap_label"].isin({_cap_label(cap) for cap in caps}))
        & (timestamp_rows["cap_label"] != "inf")
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    statistics = [
        "candidate_load_saved_vs_uncapped",
        "candidate_load_reduction_vs_uncapped",
        "unconditional_full_set_recall_loss_vs_uncapped",
        "conditional_full_set_coverage_delta_vs_uncapped",
    ]
    seed = bootstrap_seed
    for cap_label, frame in selected.groupby("cap_label", sort=False):
        for statistic in statistics:
            result = _bootstrap_statistic(
                frame,
                statistic,
                weight_column="query_count",
                block_length=block_length,
                iterations=iterations,
                bootstrap_seed=seed,
            )
            seed += 1
            records.append(
                {
                    "deletion_rate": deletion_rate,
                    "method": method,
                    "cap_label": cap_label,
                    "statistic": statistic,
                    "observed": result["observed"],
                    "ci95_low": result["ci95"][0],
                    "ci95_high": result["ci95"][1],
                    "pvalue_positive": result["pvalue_positive"],
                    "seed_count": result["seed_count"],
                    "timestamp_count": result["timestamp_count"],
                    "block_length": block_length,
                    "iterations": iterations,
                    "bootstrap_seed": seed - 1,
                }
            )
    return _method_sort(pd.DataFrame(records), ["deletion_rate"])


def _detect_num_entities(run_root: Path) -> int | None:
    manifest_path = run_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    return None if "num_entities" not in manifest else int(manifest["num_entities"])


def export_set_size_utility(
    run_root: Path,
    paper_root: Path,
    *,
    output_name: str = "final_confirmatory",
    caps: tuple[float, ...] = DEFAULT_CAPS,
    paper_caps: tuple[float, ...] = DEFAULT_PAPER_CAPS,
    paper_methods: tuple[str, ...] = DEFAULT_METHODS,
    paper_deletion_rate: float | None = None,
    bootstrap_method: str = "rolling",
    bootstrap_block_length: int = DEFAULT_BLOCK_LENGTH,
    bootstrap_iterations: int = DEFAULT_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    paper_root = paper_root.resolve()
    if math.inf not in caps:
        caps = (*caps, math.inf)
    caps = parse_caps(caps)
    paper_caps = parse_caps(paper_caps)
    per_query_path = run_root / "metrics" / "per_query.parquet"
    if not per_query_path.is_file():
        raise FileNotFoundError(f"missing per-query artifact: {per_query_path}")
    num_entities = _detect_num_entities(run_root)

    rows = read_per_query(per_query_path)
    queries = build_query_level_rows(rows)
    by_timestamp = add_uncapped_deltas(
        summarize_operating_points(
            queries,
            caps=caps,
            num_entities=num_entities,
            by_timestamp=True,
        )
    )
    by_seed = add_uncapped_deltas(
        summarize_operating_points(
            queries,
            caps=caps,
            num_entities=num_entities,
            by_timestamp=False,
        )
    )
    summary = aggregate_over_seeds(by_seed)
    if paper_deletion_rate is None:
        paper_deletion_rate = float(summary["deletion_rate"].max())
    paper_table = build_paper_table(
        summary,
        deletion_rate=paper_deletion_rate,
        methods=paper_methods,
        caps=paper_caps,
    )
    effects = bootstrap_utility_effects(
        by_timestamp,
        deletion_rate=paper_deletion_rate,
        method=bootstrap_method,
        caps=paper_caps,
        block_length=bootstrap_block_length,
        iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )

    data_dir = paper_root / "data" / output_name
    output_paths = {
        "set_size_utility_by_timestamp.csv": data_dir
        / "set_size_utility_by_timestamp.csv",
        "set_size_utility_by_seed.csv": data_dir / "set_size_utility_by_seed.csv",
        "set_size_utility_summary.csv": data_dir / "set_size_utility_summary.csv",
        "set_size_utility_paper_table.csv": data_dir
        / "set_size_utility_paper_table.csv",
        "set_size_utility_effects.csv": data_dir / "set_size_utility_effects.csv",
    }
    _write_csv(by_timestamp, output_paths["set_size_utility_by_timestamp.csv"])
    _write_csv(by_seed, output_paths["set_size_utility_by_seed.csv"])
    _write_csv(summary, output_paths["set_size_utility_summary.csv"])
    _write_csv(paper_table, output_paths["set_size_utility_paper_table.csv"])
    _write_csv(effects, output_paths["set_size_utility_effects.csv"])

    outputs = {name: sha256_file(path) for name, path in output_paths.items()}
    manifest_out = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "source_per_query": str(per_query_path),
        "source_per_query_sha256": sha256_file(per_query_path),
        "num_entities": num_entities,
        "label_rows": int(len(rows)),
        "unique_query_rows": int(len(queries)),
        "timestamp_operating_point_rows": int(len(by_timestamp)),
        "seed_operating_point_rows": int(len(by_seed)),
        "summary_rows": int(len(summary)),
        "paper_table_rows": int(len(paper_table)),
        "effect_rows": int(len(effects)),
        "caps": [_cap_label(cap) for cap in caps],
        "paper_caps": [_cap_label(cap) for cap in paper_caps],
        "paper_methods": list(paper_methods),
        "paper_deletion_rate": paper_deletion_rate,
        "bootstrap": {
            "method": bootstrap_method,
            "block_length": bootstrap_block_length,
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
        },
        "definition": {
            "cap": (
                "Maximum accepted prediction-set size. Queries above the cap "
                "are abstained, not counted as answered."
            ),
            "conditional_full_set_coverage": (
                "Full-set coverage among answered unique queries only."
            ),
            "unconditional_full_set_recall": (
                "Fraction of all unique queries that are both answered and "
                "full-set covered; abstained queries contribute zero."
            ),
            "candidate_load": (
                "Mean number of returned candidate entities per original unique "
                "query, counting abstentions as zero returned candidates."
            ),
            "candidate_load_reduction_vs_uncapped": (
                "Relative reduction in candidate_load compared with the same "
                "seed/deletion/method without a cap."
            ),
        },
        "outputs": outputs,
    }
    manifest_path = data_dir / "set_size_utility_manifest.json"
    _write_json(manifest_out, manifest_path)
    manifest_out["outputs"]["set_size_utility_manifest.json"] = sha256_file(
        manifest_path
    )
    _write_json(manifest_out, manifest_path)
    return manifest_out


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument("--output-name", default="final_confirmatory")
    parser.add_argument(
        "--caps",
        default=",".join(_cap_label(cap) for cap in DEFAULT_CAPS),
        help="Comma-separated max-set-size caps. Use inf for no cap.",
    )
    parser.add_argument(
        "--paper-caps",
        default=",".join(_cap_label(cap) for cap in DEFAULT_PAPER_CAPS),
    )
    parser.add_argument(
        "--paper-methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods to include in the compact paper table.",
    )
    parser.add_argument("--paper-deletion-rate", type=float)
    parser.add_argument("--bootstrap-method", default="rolling")
    parser.add_argument("--bootstrap-block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    manifest = export_set_size_utility(
        args.run_root,
        args.paper_root,
        output_name=args.output_name,
        caps=parse_caps(args.caps),
        paper_caps=parse_caps(args.paper_caps),
        paper_methods=tuple(
            method.strip() for method in args.paper_methods.split(",") if method.strip()
        ),
        paper_deletion_rate=args.paper_deletion_rate,
        bootstrap_method=args.bootstrap_method,
        bootstrap_block_length=args.bootstrap_block_length,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
