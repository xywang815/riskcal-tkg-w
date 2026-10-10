from pathlib import Path
import json

import pandas as pd
import pytest
import torch

from riskcal_tkg.artifacts import (
    RunDirectory,
    atomic_save_checkpoint,
    atomic_write_dataframe,
    atomic_write_json,
    capture_environment,
    load_verified_checkpoint,
    sha256_file,
)


def test_run_directory_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "run"
    RunDirectory.create(target)
    with pytest.raises(FileExistsError):
        RunDirectory.create(target)


def test_atomic_json_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert len(sha256_file(path)) == 64
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_tabular_and_checkpoint_writes(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"x": 1, "y": 2.0}])
    csv_path = tmp_path / "table.csv"
    parquet_path = tmp_path / "table.parquet"
    checkpoint_path = tmp_path / "model.ckpt"
    atomic_write_dataframe(csv_path, frame)
    atomic_write_dataframe(parquet_path, frame)
    atomic_save_checkpoint(checkpoint_path, {"weight": torch.tensor([3.0])})
    assert pd.read_csv(csv_path).to_dict("records") == [{"x": 1, "y": 2.0}]
    assert pd.read_parquet(parquet_path).to_dict("records") == [{"x": 1, "y": 2.0}]
    assert torch.load(checkpoint_path)["weight"].item() == 3.0


def test_environment_capture_contains_reproducibility_fields() -> None:
    environment = capture_environment()
    for key in ("os", "python", "torch", "cuda_available", "cpu_count", "ram_bytes"):
        assert key in environment


def test_checkpoint_loading_rejects_provenance_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "model.ckpt"
    atomic_save_checkpoint(
        path,
        {
            "config_sha256": "config-a",
            "dataset_sha256": "dataset-a",
            "deletion_mask_sha256": "mask-a",
        },
    )
    with pytest.raises(ValueError, match="config_sha256"):
        load_verified_checkpoint(
            path,
            config_sha256="config-b",
            dataset_sha256="dataset-a",
            deletion_mask_sha256="mask-a",
        )
