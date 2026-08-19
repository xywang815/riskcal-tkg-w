from pathlib import Path
import hashlib
import json

import pandas as pd
import pytest

from scripts.export_paper_diagnostic import (
    build_dataset_manifest,
    build_timestamp_mapping,
    condition_summary,
    verify_source_files,
)


def test_dataset_manifest_names_internal_and_raw_timestamps_explicitly() -> None:
    source = {
        "train_max_timestamp": 1,
        "train_timestamp_range": ["1", "2"],
        "calibration_max_timestamp": 2,
        "calibration_timestamp_range": ["3", "3"],
        "test_max_timestamp": 3,
        "test_timestamp_range": ["4", "4"],
    }

    exported = build_dataset_manifest(source)

    assert "test_max_timestamp" not in exported
    assert exported["test_max_timestamp_id"] == 3
    assert exported["test_max_timestamp_raw"] == "4"


def test_timestamp_mapping_recovers_numeric_raw_labels(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.txt").write_text("a\tr\tb\t10\n", encoding="utf-8")
    (data / "valid.txt").write_text("a\tr\tb\t20\n", encoding="utf-8")
    (data / "test.txt").write_text("a\tr\tb\t30\n", encoding="utf-8")

    mapping = build_timestamp_mapping(data)

    assert mapping == {0: "10", 1: "20", 2: "30"}


def test_source_file_verification_rejects_hash_mismatch(tmp_path: Path) -> None:
    for name in ("train.txt", "valid.txt", "test.txt"):
        (tmp_path / name).write_text(f"a\tr\tb\t{name}\n", encoding="utf-8")
    manifest = {
        "source_files": {
            name: {
                "bytes": (tmp_path / name).stat().st_size,
                "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
            }
            for name in ("train.txt", "valid.txt", "test.txt")
        }
    }
    manifest["source_files"]["test.txt"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="test.txt.*SHA-256"):
        verify_source_files(tmp_path, manifest)


def test_condition_summary_uses_query_weighting_and_macro_time() -> None:
    rows = pd.DataFrame(
        [
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "actual_deletion_rate": 0.25,
                "method": "weighted",
                "timestamp": 2,
                "calibration_max_timestamp": 1,
                "selected_half_life": 7.0,
                "query_count": 3,
                "coverage": 1.0,
                "mean_size": 4.0,
                "mrr": 0.5,
                "frequency_mrr": 0.25,
            },
            {
                "seed": 17,
                "deletion_rate": 0.3,
                "actual_deletion_rate": 0.25,
                "method": "weighted",
                "timestamp": 3,
                "calibration_max_timestamp": 2,
                "selected_half_life": 7.0,
                "query_count": 1,
                "coverage": 0.0,
                "mean_size": 8.0,
                "mrr": 0.1,
                "frequency_mrr": 0.2,
            },
        ]
    )

    summary = condition_summary(rows, target=0.9, num_entities=10)
    record = summary.iloc[0]

    assert record["coverage"] == 0.75
    assert record["macro_time_coverage"] == 0.5
    assert record["positive_undercoverage"] == 0.45
    assert record["fraction_timestamps_below_target"] == 0.5
    assert record["mean_size"] == 5.0
    assert record["macro_time_mean_size"] == 6.0
    assert record["normalized_mean_size"] == 0.5
