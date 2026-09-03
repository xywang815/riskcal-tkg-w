from pathlib import Path

import pandas as pd
import pytest

from scripts.export_followup_bootstrap import (
    _paired_contrast,
    export_followup_bootstrap,
)


def test_paired_contrast_aligns_methods_and_weights() -> None:
    rows = pd.DataFrame(
        [
            {
                "seed": 17,
                "timestamp": 1,
                "method": method,
                "label_coverage": coverage,
                "label_count": 10,
            }
            for method, coverage in (
                ("label_margin_static", 0.8),
                ("label_margin_rolling", 0.9),
            )
        ]
    )
    contrast = _paired_contrast(
        rows,
        left_method="label_margin_rolling",
        right_method="label_margin_static",
        metric="label_coverage",
        weight_column="label_count",
    )
    assert contrast["value"].iloc[0] == pytest.approx(0.1)
    assert contrast["weight"].iloc[0] == 10


def test_followup_bootstrap_exports_fixed_contrasts(tmp_path: Path) -> None:
    source = tmp_path / "followup"
    source.mkdir()
    methods = [
        "label_margin_static",
        "label_margin_rolling",
        "label_negscore_rolling",
        "label_minmax_rolling",
        "label_softmax_rolling",
        "query_max_margin_rolling",
    ]
    records = []
    for seed in (17, 29):
        for timestamp in (1, 2, 3):
            for index, method in enumerate(methods):
                records.append(
                    {
                        "case": "toy",
                        "dataset_mode": "toy",
                        "model_name": "toy_model",
                        "seed": seed,
                        "deletion_rate": 0.3,
                        "timestamp": timestamp,
                        "method": method,
                        "label_count": 20,
                        "query_count": 10,
                        "label_coverage": 0.8 + index * 0.01,
                        "full_set_coverage": 0.7 + index * 0.01,
                        "mean_set_size": 4.0 + index,
                        "single_full_set_coverage": 0.85 + index * 0.01,
                        "multi_full_set_coverage": 0.65 + index * 0.02,
                    }
                )
    rows = pd.DataFrame(records)
    rows.to_csv(source / "followup_by_timestamp.csv", index=False)
    (source / "followup_manifest.json").write_text(
        '{"git_commit":"abc","status":"complete","timestamp_row_count":36}\n',
        encoding="utf-8",
    )
    output = tmp_path / "bootstrap"
    manifest = export_followup_bootstrap(
        source,
        output,
        block_lengths=(1,),
        iterations=20,
        bootstrap_seed=3,
    )
    summary = pd.read_csv(output / "followup_bootstrap_summary.csv")
    assert manifest["status"] == "complete"
    assert manifest["statistics_count"] == 13
    assert len(summary) == 13
    assert set(summary["family"]) == {
        "history_policy",
        "query_objective",
        "score_function",
    }
    assert (output / "SHA256SUMS.txt").is_file()
