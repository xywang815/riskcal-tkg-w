from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import pandas as pd
import psutil
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    return Path(name)


def atomic_write_json(path: Path, value: object) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary_path(path)
    try:
        if path.suffix.lower() == ".csv":
            frame.to_csv(temporary, index=False)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame.to_parquet(temporary, index=False)
        else:
            raise ValueError("tabular artifact must be CSV or Parquet")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_save_checkpoint(path: Path, value: object) -> None:
    temporary = _temporary_path(path)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_verified_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    dataset_sha256: str,
    deletion_mask_sha256: str,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    expected = {
        "config_sha256": config_sha256,
        "dataset_sha256": dataset_sha256,
        "deletion_mask_sha256": deletion_mask_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint {key} does not match current run")
    return payload


def capture_environment(repo_root: Path | None = None) -> dict[str, object]:
    environment: dict[str, object] = {
        "os": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cpu_count": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
    }
    if torch.cuda.is_available():
        environment["gpu"] = torch.cuda.get_device_name(0)
        environment["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    else:
        environment["gpu"] = None
        environment["gpu_memory_bytes"] = None
    if repo_root is not None:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        environment["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
    return environment


@dataclass(frozen=True)
class RunDirectory:
    root: Path

    @classmethod
    def create(cls, root: Path) -> "RunDirectory":
        root.mkdir(parents=True, exist_ok=False)
        for child in (
            "checkpoints",
            "deletion_masks",
            "metrics",
            "figures",
            "resources",
            "conditions",
            "incomplete",
        ):
            (root / child).mkdir()
        return cls(root)
