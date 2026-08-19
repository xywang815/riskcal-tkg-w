from pathlib import Path

import pandas as pd

from scripts.export_timestamp_block_sensitivity import export_timestamp_block_sensitivity


def test_timestamp_block_sensitivity_exports_all_block_lengths(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    metrics = run_root / "metrics"
    metrics.mkdir(parents=True)
    records = []
    for seed, adjustment in ((17, 0.0), (29, 0.01)):
        for timestamp in range(1, 5):
            for method, coverage in (
                ("static", 0.82 + adjustment),
                ("rolling", 0.90 + adjustment),
            ):
                records.append(
                    {
                        "seed": seed,
                        "timestamp": timestamp,
                        "deletion_rate": 0.3,
                        "method": method,
                        "coverage": coverage,
                        "query_count": 10,
                        "mrr": 0.31,
                        "frequency_mrr": 0.09,
                    }
                )
    pd.DataFrame(records).to_csv(metrics / "per_window.csv", index=False)

    manifest = export_timestamp_block_sensitivity(
        run_root,
        tmp_path / "paper",
        block_lengths=(1, 2),
        iterations=20,
        bootstrap_seed=3,
    )

    summary = pd.read_csv(
        tmp_path / "paper" / "data" / "final_confirmatory" / "timestamp_block_sensitivity_summary.csv"
    )
    assert manifest["block_lengths"] == [1, 2]
    assert set(summary["block_length"]) == {1, 2}
    assert len(summary) == 6

    primary = summary[
        summary["statistic"] == "rolling_undercoverage_reduction_vs_static"
    ].sort_values("block_length")
    assert list(primary["observed"].round(6)) == [0.075, 0.075]
    assert (primary["ci95_low"] > 0).all()
