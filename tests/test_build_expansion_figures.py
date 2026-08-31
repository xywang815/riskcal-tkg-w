from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_expansion_figures import (
    METHOD_ORDER,
    RUN_ORDER,
    build_expansion_figures,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checked(directory: Path, name: str, frame: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_csv(path, index=False)
    (directory / "SHA256SUMS.txt").write_text(
        f"{_sha256(path)}  {name}\n", encoding="utf-8"
    )


def _condition_rows() -> pd.DataFrame:
    records = []
    for run_index, run in enumerate(RUN_ORDER):
        for rate in (0.0, 0.3):
            for method_index, method in enumerate(METHOD_ORDER):
                records.append(
                    {
                        "run": run,
                        "deletion_rate": rate,
                        "method": method,
                        "coverage_mean": 0.78 + 0.02 * method_index + 0.005 * run_index,
                        "coverage_sd": 0.01,
                        "seed_count": 5,
                    }
                )
    return pd.DataFrame(records)


def _bootstrap_rows() -> pd.DataFrame:
    records = []
    for run in RUN_ORDER:
        for baseline in (
            "static_margin",
            "kgcp_negscore_static",
            "kgcp_minmax_static",
            "kgcp_softmax_static",
        ):
            records.append(
                {
                    "family": "rolling_vs_static_baseline",
                    "run": run,
                    "comparison": f"rolling_margin-minus-{baseline}",
                    "statistic": "coverage_gain",
                    "dataset_mode": "icews14",
                    "model_name": "temporal_distmult",
                    "deletion_rate": 0.3,
                    "block_length": 7,
                    "observed": 0.05,
                    "ci95_low": 0.03,
                    "ci95_high": 0.07,
                }
            )
        for method in ("static_margin", "rolling_margin"):
            records.append(
                {
                    "family": "deletion_effect",
                    "run": run,
                    "comparison": f"delete0.3-minus-delete0__{method}",
                    "statistic": "coverage_change",
                    "dataset_mode": "icews14",
                    "model_name": "temporal_distmult",
                    "deletion_rate": 0.3,
                    "block_length": 7,
                    "observed": -0.01,
                    "ci95_low": -0.03,
                    "ci95_high": 0.01,
                }
            )
        for method in ("static_margin", "rolling_margin", "kgcp_softmax_static"):
            records.append(
                {
                    "family": "multi_answer_degradation",
                    "run": run,
                    "comparison": f"multi-minus-single__{method}",
                    "statistic": "full_set_coverage_gap",
                    "dataset_mode": "icews14",
                    "model_name": "temporal_distmult",
                    "deletion_rate": 0.3,
                    "block_length": 7,
                    "observed": -0.12,
                    "ci95_low": -0.15,
                    "ci95_high": -0.09,
                }
            )
    for dataset, model in (
        ("icews05_15", "temporal_distmult"),
        ("icews14", "continuous_temporal_complex"),
    ):
        for rate in (0.0, 0.3):
            for method in ("static_margin", "rolling_margin"):
                records.append(
                    {
                        "family": "negative_sampling_sensitivity",
                        "run": f"{dataset}_{model}_filtered|uniform",
                        "comparison": f"filtered-minus-uniform__{method}",
                        "statistic": "coverage_change",
                        "dataset_mode": dataset,
                        "model_name": model,
                        "deletion_rate": rate,
                        "block_length": 7,
                        "observed": 0.01,
                        "ci95_low": -0.01,
                        "ci95_high": 0.03,
                    }
                )
    return pd.DataFrame(records)


def test_build_expansion_figures_writes_traceable_outputs(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    bootstrap = tmp_path / "bootstrap"
    output = tmp_path / "figures"
    _write_checked(analysis, "condition_aggregate.csv", _condition_rows())
    _write_checked(bootstrap, "expansion_bootstrap_summary.csv", _bootstrap_rows())

    manifest = build_expansion_figures(analysis, bootstrap, output)

    assert len(manifest["outputs"]) == 6
    assert (output / "expansion_figure_manifest.json").is_file()
    assert (output / "SHA256SUMS.txt").is_file()
    for name, digest in manifest["outputs"].items():
        assert _sha256(output / name) == digest


def test_build_expansion_figures_rejects_tampered_input(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    bootstrap = tmp_path / "bootstrap"
    _write_checked(analysis, "condition_aggregate.csv", _condition_rows())
    _write_checked(bootstrap, "expansion_bootstrap_summary.csv", _bootstrap_rows())
    with (analysis / "condition_aggregate.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        build_expansion_figures(analysis, bootstrap, tmp_path / "figures")
