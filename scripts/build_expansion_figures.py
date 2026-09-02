"""Build traceable paper figures from the verified expansion summaries."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = (
    "static_margin",
    "rolling_margin",
    "kgcp_negscore_static",
    "kgcp_minmax_static",
    "kgcp_softmax_static",
)
METHOD_LABELS = {
    "static_margin": "Static margin",
    "rolling_margin": "Rolling margin",
    "kgcp_negscore_static": "KGCP NegScore",
    "kgcp_minmax_static": "KGCP Minmax",
    "kgcp_softmax_static": "KGCP Softmax",
}
METHOD_COLORS = {
    "static_margin": "#D55E00",
    "rolling_margin": "#0072B2",
    "kgcp_negscore_static": "#009E73",
    "kgcp_minmax_static": "#E69F00",
    "kgcp_softmax_static": "#CC79A7",
}
METHOD_MARKERS = {
    "static_margin": "s",
    "rolling_margin": "o",
    "kgcp_negscore_static": "^",
    "kgcp_minmax_static": "D",
    "kgcp_softmax_static": "v",
}
RUN_ORDER = (
    "icews14_distmult_filtered",
    "icews14_tcomplex_filtered",
    "icews05_15_distmult_filtered",
    "icews05_15_tcomplex_filtered",
)
RUN_LABELS = {
    "icews14_distmult_filtered": "ICEWS14 | DistMult",
    "icews14_tcomplex_filtered": "ICEWS14 | continuous ComplEx",
    "icews05_15_distmult_filtered": "ICEWS05-15 | DistMult",
    "icews05_15_tcomplex_filtered": "ICEWS05-15 | continuous ComplEx",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_checksum_manifest(directory: Path) -> dict[str, str]:
    path = directory / "SHA256SUMS.txt"
    if not path.is_file():
        raise ValueError(f"missing checksum manifest: {path}")
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        records[name.strip()] = digest
    return records


def _verify_inputs(directory: Path, names: Iterable[str]) -> dict[str, str]:
    manifest = _read_checksum_manifest(directory)
    verified: dict[str, str] = {}
    for name in names:
        path = directory / name
        expected = manifest.get(name)
        if expected is None or not path.is_file():
            raise ValueError(f"untracked figure input: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"checksum mismatch for figure input: {path}")
        verified[name] = observed
    return verified


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.2,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontweight="bold", va="top")
    ax.set_title(title, loc="left", pad=7)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "RiskCal-TKG", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        png,
        bbox_inches="tight",
        metadata={"Software": "RiskCal-TKG"},
    )
    plt.close(fig)
    return {pdf.name: _sha256(pdf), png.name: _sha256(png)}


def build_generalization(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    _require_columns(
        frame,
        (
            "run",
            "deletion_rate",
            "method",
            "coverage_mean",
            "coverage_sd",
            "seed_count",
        ),
        "condition_aggregate.csv",
    )
    selected = frame[
        frame["run"].isin(RUN_ORDER) & frame["method"].isin(METHOD_ORDER)
    ].copy()
    if set(selected["run"]) != set(RUN_ORDER):
        raise ValueError("generalization figure requires all four filtered runs")
    if selected[["coverage_mean", "coverage_sd"]].isna().any().any():
        raise ValueError("generalization coverage values must be finite")
    if (selected["seed_count"] < 2).any():
        raise ValueError("generalization figure requires at least two seeds per point")

    lower = float((selected["coverage_mean"] - selected["coverage_sd"]).min())
    upper = float((selected["coverage_mean"] + selected["coverage_sd"]).max())
    lower = min(lower, 0.90)
    upper = max(upper, 0.90)
    pad = max(0.015, 0.08 * (upper - lower))
    limits = (max(0.0, lower - pad), min(1.0, upper + pad))

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.05), sharex=True, sharey=True)
    for index, (ax, run) in enumerate(zip(axes.flat, RUN_ORDER, strict=True)):
        rows = selected[selected["run"] == run]
        for method in METHOD_ORDER:
            values = rows[rows["method"] == method].sort_values("deletion_rate")
            if values.empty:
                raise ValueError(f"missing {method} rows for {run}")
            ax.errorbar(
                values["deletion_rate"] * 100,
                values["coverage_mean"],
                yerr=values["coverage_sd"],
                marker=METHOD_MARKERS[method],
                markersize=3.4,
                linewidth=1.25,
                capsize=2,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.axhline(0.90, color="#333333", linestyle="--", linewidth=0.9)
        ax.set_ylim(*limits)
        ax.set_xticks(sorted(rows["deletion_rate"].unique() * 100))
        _panel(ax, f"({chr(97 + index)})", RUN_LABELS[run])
    axes[1, 0].set_xlabel("Deleted training facts (%)")
    axes[1, 1].set_xlabel("Deleted training facts (%)")
    axes[0, 0].set_ylabel("Observed-label coverage")
    axes[1, 0].set_ylabel("Observed-label coverage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.86, hspace=0.34, wspace=0.18)
    return _save(fig, output_dir, "expansion_generalization")


def _effect_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    labels: list[str],
    colors: list[str],
    title: str,
    panel: str,
) -> None:
    y = np.arange(len(rows))
    observed = rows["observed"].to_numpy(dtype=float)
    low = rows["ci95_low"].to_numpy(dtype=float)
    high = rows["ci95_high"].to_numpy(dtype=float)
    for position, value, interval_low, interval_high, color in zip(
        y, observed, low, high, colors, strict=True
    ):
        ax.errorbar(
            value,
            position,
            xerr=[[value - interval_low], [interval_high - value]],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=2,
        )
    ax.scatter(observed, y, c=colors, s=18, zorder=3)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.set_title(title, loc="left", pad=7)
    ax.text(-0.08, 1.06, panel, transform=ax.transAxes, fontweight="bold", va="top")


def build_reviewer_diagnostics(
    frame: pd.DataFrame, output_dir: Path, block_length: int
) -> dict[str, str]:
    _require_columns(
        frame,
        (
            "family",
            "run",
            "comparison",
            "statistic",
            "deletion_rate",
            "block_length",
            "observed",
            "ci95_low",
            "ci95_high",
        ),
        "expansion_bootstrap_summary.csv",
    )
    rows = frame[frame["block_length"] == block_length].copy()
    if rows.empty:
        raise ValueError(f"no bootstrap rows for block length {block_length}")

    method = rows[
        (rows["family"] == "rolling_vs_static_baseline")
        & (rows["statistic"] == "coverage_gain")
    ].copy()
    deletion = rows[
        (rows["family"] == "deletion_effect")
        & (rows["statistic"] == "coverage_change")
    ].copy()
    sampling = rows[
        (rows["family"] == "negative_sampling_sensitivity")
        & (rows["statistic"] == "coverage_change")
    ].copy()
    if method.empty or deletion.empty or sampling.empty:
        raise ValueError("reviewer diagnostics require method, deletion, and sampling families")

    method["baseline"] = method["comparison"].str.replace(
        "rolling_margin-minus-", "", regex=False
    )
    method["run_rank"] = method["run"].map({v: i for i, v in enumerate(RUN_ORDER)})
    method["method_rank"] = method["baseline"].map(
        {v: i for i, v in enumerate(METHOD_ORDER)})
    method = method.sort_values(["run_rank", "method_rank"], kind="stable")
    diagnostic_run_labels = {
        "icews14_distmult_filtered": "ICEWS14 | DistMult",
        "icews14_tcomplex_filtered": "ICEWS14 | cComplEx",
        "icews05_15_distmult_filtered": "ICEWS05-15 | DistMult",
        "icews05_15_tcomplex_filtered": "ICEWS05-15 | cComplEx",
    }
    method_labels = [
        f"{diagnostic_run_labels.get(run, run)} | {METHOD_LABELS.get(base, base)}"
        for run, base in zip(method["run"], method["baseline"], strict=True)
    ]
    method_colors = [METHOD_COLORS.get(value, "#777777") for value in method["baseline"]]

    deletion["method"] = deletion["comparison"].str.split("__").str[-1]
    deletion["run_rank"] = deletion["run"].map({v: i for i, v in enumerate(RUN_ORDER)})
    deletion["method_rank"] = deletion["method"].map(
        {"static_margin": 0, "rolling_margin": 1}
    )
    deletion = deletion.sort_values(["run_rank", "method_rank"], kind="stable")
    deletion_labels = [
        f"{diagnostic_run_labels.get(run, run)} | {METHOD_LABELS.get(method_name, method_name)}"
        for run, method_name in zip(deletion["run"], deletion["method"], strict=True)
    ]
    deletion_colors = [METHOD_COLORS.get(value, "#777777") for value in deletion["method"]]

    sampling["method"] = sampling["comparison"].str.split("__").str[-1]
    sampling = sampling.sort_values(
        ["dataset_mode", "model_name", "deletion_rate", "method"], kind="stable"
    )
    diagnostic_dataset_labels = {
        "icews14": "ICEWS14",
        "icews05_15": "ICEWS05-15",
    }
    diagnostic_model_labels = {
        "temporal_distmult": "DistMult",
        "continuous_tcomplex": "cComplEx",
    }
    sampling_labels = [
        f"{diagnostic_dataset_labels.get(dataset, dataset)} | "
        f"{diagnostic_model_labels.get(model, model)} | "
        f"delete {rate:.0%} | {METHOD_LABELS.get(method_name, method_name)}"
        for dataset, model, rate, method_name in zip(
            sampling["dataset_mode"],
            sampling["model_name"],
            sampling["deletion_rate"],
            sampling["method"],
            strict=True,
        )
    ]
    sampling_colors = [METHOD_COLORS.get(value, "#777777") for value in sampling["method"]]

    fig = plt.figure(figsize=(7.25, 8.3), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(1.55, 0.78, 0.9))
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
    )
    _effect_panel(
        axes[0], method, method_labels, method_colors,
        "Rolling minus static baseline", "(a)",
    )
    axes[0].set_xlabel("Coverage gain at 30% deletion")
    _effect_panel(
        axes[1], deletion, deletion_labels, deletion_colors,
        "Deletion interaction", "(b)",
    )
    axes[1].set_xlabel("Coverage change: 30% minus 0% deletion")
    _effect_panel(
        axes[2], sampling, sampling_labels, sampling_colors,
        "Filtered negative sampling sensitivity", "(c)",
    )
    axes[2].set_xlabel("Coverage change: filtered minus uniform")
    return _save(fig, output_dir, "expansion_reviewer_diagnostics")


def build_multi_answer(
    frame: pd.DataFrame, output_dir: Path, block_length: int
) -> dict[str, str]:
    rows = frame[
        (frame["block_length"] == block_length)
        & (frame["family"] == "multi_answer_degradation")
        & (frame["statistic"] == "full_set_coverage_gap")
    ].copy()
    if rows.empty:
        raise ValueError("multi-answer bootstrap rows are missing")
    rows["method"] = rows["comparison"].str.replace(
        "multi-minus-single__", "", regex=False
    )
    rows["run_rank"] = rows["run"].map({v: i for i, v in enumerate(RUN_ORDER)})
    rows["method_rank"] = rows["method"].map(
        {
            "static_margin": 0,
            "rolling_margin": 1,
            "kgcp_softmax_static": 2,
        }
    )
    rows = rows.sort_values(["run_rank", "method_rank"], kind="stable")
    labels = [
        f"{RUN_LABELS.get(run, run)} | {METHOD_LABELS.get(method_name, method_name)}"
        for run, method_name in zip(rows["run"], rows["method"], strict=True)
    ]
    colors = [METHOD_COLORS.get(value, "#777777") for value in rows["method"]]
    fig, ax = plt.subplots(figsize=(7.25, 4.3), constrained_layout=True)
    _effect_panel(
        ax,
        rows,
        labels,
        colors,
        "Multi-answer full-set coverage relative to single-answer queries",
        "",
    )
    ax.set_xlabel("Coverage gap: multi-answer minus single-answer")
    return _save(fig, output_dir, "expansion_multi_answer")


def build_expansion_figures(
    analysis_dir: Path,
    bootstrap_dir: Path,
    output_dir: Path,
    *,
    block_length: int = 7,
) -> dict[str, object]:
    analysis_dir = analysis_dir.resolve()
    bootstrap_dir = bootstrap_dir.resolve()
    output_dir = output_dir.resolve()
    analysis_name = "condition_aggregate.csv"
    bootstrap_name = "expansion_bootstrap_summary.csv"
    inputs = {
        "analysis": _verify_inputs(analysis_dir, (analysis_name,)),
        "bootstrap": _verify_inputs(bootstrap_dir, (bootstrap_name,)),
    }
    conditions = pd.read_csv(analysis_dir / analysis_name)
    bootstrap = pd.read_csv(bootstrap_dir / bootstrap_name)
    _style()
    outputs: dict[str, str] = {}
    outputs.update(build_generalization(conditions, output_dir))
    outputs.update(build_reviewer_diagnostics(bootstrap, output_dir, block_length))
    outputs.update(build_multi_answer(bootstrap, output_dir, block_length))
    manifest = {
        "block_length": int(block_length),
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": inputs,
        "outputs": outputs,
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    manifest_path = output_dir / "expansion_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checksum_paths = [*(output_dir / name for name in outputs), manifest_path]
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n" for path in sorted(checksum_paths)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-length", type=int, default=7)
    args = parser.parse_args()
    if args.block_length <= 0:
        raise SystemExit("block length must be positive")
    manifest = build_expansion_figures(
        args.analysis_dir,
        args.bootstrap_dir,
        args.output_dir,
        block_length=args.block_length,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
