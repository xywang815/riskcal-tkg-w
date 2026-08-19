from argparse import ArgumentParser
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "top1": "#59a14f",
    "static": "#9c755f",
    "rolling": "#4e79a7",
    "weighted": "#e15759",
    "adaptive": "#b07aa1",
}


def _weighted_means(
    rows: pd.DataFrame,
    groups: list[str],
    values: list[str],
) -> pd.DataFrame:
    if "query_count" not in rows or (rows["query_count"] <= 0).any():
        raise ValueError("query_count must be present and positive")
    weighted = rows[groups + ["query_count", *values]].copy()
    totals: list[str] = []
    for value in values:
        total = f"_{value}_weighted_total"
        weighted[total] = weighted[value] * weighted["query_count"]
        totals.append(total)
    aggregated = weighted.groupby(groups, as_index=False).agg(
        {"query_count": "sum", **{total: "sum" for total in totals}}
    )
    for value, total in zip(values, totals, strict=True):
        aggregated[value] = aggregated[total] / aggregated["query_count"]
    return aggregated[groups + values]


def _condition_table(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seed",
        "deletion_rate",
        "method",
        "coverage",
        "mean_size",
        "query_count",
        "mrr",
        "frequency_mrr",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"missing metric columns: {missing}")
    methods = {"static", "rolling", "weighted"}
    absent_methods = sorted(methods - set(rows["method"]))
    if absent_methods:
        raise ValueError(f"missing methods: {absent_methods}")
    if (rows["query_count"] <= 0).any() or not np.isfinite(rows["query_count"]).all():
        raise ValueError("query_count must be finite and positive")
    weighted = rows.copy()
    for value in ("coverage", "mean_size", "mrr", "frequency_mrr"):
        weighted[f"_{value}_total"] = weighted[value] * weighted["query_count"]
    conditions = weighted.groupby(
        ["seed", "deletion_rate", "method"], as_index=False
    ).agg(
        query_count=("query_count", "sum"),
        coverage_total=("_coverage_total", "sum"),
        size_total=("_mean_size_total", "sum"),
        mrr_total=("_mrr_total", "sum"),
        frequency_mrr_total=("_frequency_mrr_total", "sum"),
    )
    conditions["coverage"] = conditions["coverage_total"] / conditions["query_count"]
    conditions["mean_size"] = conditions["size_total"] / conditions["query_count"]
    conditions["mrr"] = conditions["mrr_total"] / conditions["query_count"]
    conditions["frequency_mrr"] = (
        conditions["frequency_mrr_total"] / conditions["query_count"]
    )
    return conditions[
        [
            "seed",
            "deletion_rate",
            "method",
            "query_count",
            "coverage",
            "mean_size",
            "mrr",
            "frequency_mrr",
        ]
    ]


def _hierarchical_frequency_bootstrap(
    query_rows: pd.DataFrame | None,
    strongest_rate: float,
    expected_seeds: set[int],
    iterations: int = 10_000,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": False,
        "seed_count": 0,
        "timestamp_blocks": 0,
        "mean_difference": None,
        "pvalue": 1.0,
        "ci95": [None, None],
    }
    if query_rows is None:
        return result
    required = {"seed", "deletion_rate", "timestamp", "method", "rank", "frequency_rank"}
    missing = sorted(required - set(query_rows.columns))
    if missing:
        raise ValueError(f"missing per-query columns: {missing}")
    paired = query_rows[
        (query_rows["deletion_rate"] == strongest_rate)
        & (query_rows["method"] == "static")
    ].copy()
    seeds = sorted(int(value) for value in paired["seed"].unique())
    result["seed_count"] = len(seeds)
    if set(seeds) != expected_seeds or len(seeds) < 3:
        return result
    if (paired[["rank", "frequency_rank"]] <= 0).any().any():
        raise ValueError("ranks must be positive")
    paired["rr_difference"] = 1.0 / paired["rank"] - 1.0 / paired["frequency_rank"]
    clusters = (
        paired.groupby(["seed", "timestamp"], as_index=False)
        .agg(total=("rr_difference", "sum"), count=("rr_difference", "size"))
    )
    by_seed = {
        seed: frame[["total", "count"]].to_numpy(dtype=float)
        for seed, frame in clusters.groupby("seed")
    }
    result["timestamp_blocks"] = int(len(clusters))
    observed_by_seed = [
        float(values[:, 0].sum() / values[:, 1].sum())
        for values in by_seed.values()
    ]
    observed = float(np.mean(observed_by_seed))
    rng = np.random.default_rng(20260808)
    samples = np.empty(iterations, dtype=float)
    seed_array = np.asarray(seeds, dtype=int)
    for index in range(iterations):
        sampled_seeds = rng.choice(seed_array, size=len(seed_array), replace=True)
        seed_means: list[float] = []
        for seed in sampled_seeds:
            values = by_seed[int(seed)]
            chosen = rng.integers(0, len(values), size=len(values))
            sampled = values[chosen]
            seed_means.append(float(sampled[:, 0].sum() / sampled[:, 1].sum()))
        samples[index] = float(np.mean(seed_means))
    pvalue = float((np.count_nonzero(samples <= 0.0) + 1) / (iterations + 1))
    result.update(
        {
            "valid": True,
            "mean_difference": observed,
            "pvalue": pvalue,
            "ci95": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        }
    )
    return result


def evaluate_success_gate(
    rows: pd.DataFrame,
    target: float = 0.90,
    query_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    conditions = _condition_table(rows)
    coverage = conditions.pivot(
        index=["seed", "deletion_rate"], columns="method", values="coverage"
    )
    sizes = conditions.pivot(
        index=["seed", "deletion_rate"], columns="method", values="mean_size"
    )
    if coverage.isna().any().any() or sizes.isna().any().any():
        raise ValueError("one or more seed/deletion conditions are incomplete")

    riskcal_method = "adaptive" if "adaptive" in set(conditions["method"]) else "weighted"
    pressured = conditions[conditions["deletion_rate"] > 0]
    riskcal_by_rate = (
        pressured[pressured["method"] == riskcal_method]
        .groupby("deletion_rate")["coverage"]
        .mean()
    )
    coverage_rates = [
        float(rate)
        for rate, value in riskcal_by_rate.items()
        if target - 0.02 <= value <= target + 0.02
    ]
    coverage_conditions_met = len(coverage_rates) >= 2
    pressured_rates = sorted(float(value) for value in pressured["deletion_rate"].unique())
    strongest_rate = max(pressured_rates) if pressured_rates else float(
        conditions["deletion_rate"].max()
    )
    strongest_coverage = coverage.xs(strongest_rate, level="deletion_rate")
    strongest_improvement = (
        strongest_coverage[riskcal_method] - strongest_coverage["static"]
    )
    mean_strongest_improvement = float(strongest_improvement.mean())
    gap_condition_met = bool(mean_strongest_improvement >= 0.03)
    direction_consistent = bool((strongest_improvement > 0).all())
    size_ratio = sizes[riskcal_method] / sizes["rolling"].replace(0, np.nan)
    set_size_ratio_ok = bool((size_ratio <= 1.5).all() and size_ratio.notna().all())
    frequency_test = _hierarchical_frequency_bootstrap(
        query_rows,
        strongest_rate,
        {int(value) for value in conditions["seed"].unique()},
    )
    mean_scorer_improvement = frequency_test["mean_difference"]
    scorer_better_than_frequency = bool(
        frequency_test["valid"]
        and mean_scorer_improvement is not None
        and float(mean_scorer_improvement) > 0.0
        and frequency_test["pvalue"] < 0.05
    )
    supported = all(
        (
            coverage_conditions_met,
            gap_condition_met,
            direction_consistent,
            set_size_ratio_ok,
            scorer_better_than_frequency,
        )
    )
    return {
        "status": "evaluated",
        "target_coverage": target,
        "riskcal_method": riskcal_method,
        "coverage_conditions_met": coverage_conditions_met,
        "coverage_qualified_deletion_rates": coverage_rates,
        "strongest_deletion_rate": strongest_rate,
        "gap_condition_met": gap_condition_met,
        "riskcal_minus_static_coverage": mean_strongest_improvement,
        "weighted_minus_static_coverage": mean_strongest_improvement,
        "direction_consistent": direction_consistent,
        "set_size_ratio_ok": set_size_ratio_ok,
        "max_riskcal_to_rolling_size_ratio": float(size_ratio.max()),
        "max_weighted_to_rolling_size_ratio": float(size_ratio.max()),
        "scorer_better_than_frequency": scorer_better_than_frequency,
        "mean_mrr_minus_frequency": mean_scorer_improvement,
        "frequency_hierarchical_bootstrap": frequency_test,
        "supported": supported,
        "condition_rows": int(len(conditions)),
    }


def _replace_json(path: Path, value: object) -> None:
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


def _save_figure(figure: plt.Figure, figures: Path, stem: str) -> None:
    figure.tight_layout()
    temporary_paths: list[Path] = []
    try:
        for suffix, options in (("png", {"dpi": 180}), ("pdf", {})):
            target = figures / f"{stem}.{suffix}"
            descriptor, name = tempfile.mkstemp(dir=figures, suffix=".tmp")
            os.close(descriptor)
            temporary = Path(name)
            temporary_paths.append(temporary)
            figure.savefig(temporary, format=suffix, **options)
            os.replace(temporary, target)
        plt.close(figure)
    except BaseException:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        plt.close(figure)
        raise


def _plot_coverage(rows: pd.DataFrame, figures: Path, target: float) -> None:
    grouped = _weighted_means(rows, ["method", "timestamp"], ["coverage"])
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for method, frame in grouped.groupby("method"):
        axis.plot(
            frame["timestamp"],
            frame["coverage"],
            marker="o",
            label=method,
            color=METHOD_COLORS.get(str(method)),
        )
    axis.axhline(target, color="#222222", linestyle="--", label=f"target={target:.2f}")
    axis.set(xlabel="Timestamp", ylabel="Empirical coverage", ylim=(0, 1.02))
    axis.legend()
    _save_figure(figure, figures, "coverage_by_time")


def _plot_set_size(rows: pd.DataFrame, figures: Path) -> None:
    grouped = _weighted_means(
        rows, ["method", "deletion_rate"], ["mean_size"]
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for method, frame in grouped.groupby("method"):
        axis.plot(
            frame["deletion_rate"],
            frame["mean_size"],
            marker="o",
            label=method,
            color=METHOD_COLORS.get(str(method)),
        )
    axis.set(xlabel="Training-edge deletion rate", ylabel="Mean prediction-set size")
    axis.legend()
    _save_figure(figure, figures, "set_size_by_corruption")


def _plot_risk_coverage(rows: pd.DataFrame, figures: Path) -> None:
    suffixes = sorted(
        column.removeprefix("risk_at_")
        for column in rows.columns
        if column.startswith("risk_at_")
        and f"answer_rate_at_{column.removeprefix('risk_at_')}" in rows.columns
    )
    if not suffixes:
        raise ValueError("no paired risk/answer-rate columns found")
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for method, frame in rows.groupby("method"):
        points: list[tuple[float, float]] = []
        for suffix in suffixes:
            answered = frame["query_count"] * frame[f"answer_rate_at_{suffix}"]
            answer_rate = float(answered.sum() / frame["query_count"].sum())
            if answered.sum() > 0:
                risk = float(
                    (frame[f"risk_at_{suffix}"].fillna(0.0) * answered).sum()
                    / answered.sum()
                )
            else:
                risk = float("nan")
            points.append((answer_rate, risk))
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=method,
            color=METHOD_COLORS.get(str(method)),
        )
    axis.set(xlabel="Answer rate", ylabel="Selective risk", xlim=(0, 1.02))
    axis.legend()
    _save_figure(figure, figures, "risk_coverage")


def summarize_run(run_root: Path, target: float = 0.90) -> dict[str, Any]:
    rows = pd.read_csv(run_root / "metrics" / "per_window.csv")
    query_rows = pd.read_parquet(run_root / "metrics" / "per_query.parquet")
    figures = run_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    decision = evaluate_success_gate(rows, target, query_rows=query_rows)
    summary = _condition_table(rows).to_dict("records")
    _replace_json(run_root / "metrics" / "summary.json", summary)
    _replace_json(run_root / "SUCCESS_GATE.json", decision)
    _plot_coverage(rows, figures, target)
    _plot_set_size(rows, figures)
    _plot_risk_coverage(rows, figures)
    return decision


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.90)
    args = parser.parse_args()
    decision = summarize_run(args.run_root, args.target)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
