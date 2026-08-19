from pathlib import Path
import io
import tarfile

import pytest

from scripts.prepare_icews14 import prepare_archive


def _add_text(handle: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    handle.addfile(info, io.BytesIO(payload))


def test_prepare_archive_extracts_and_hashes_expected_files(tmp_path: Path) -> None:
    archive = tmp_path / "data-release.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name in ("train.txt", "valid.txt", "test.txt"):
            _add_text(
                handle,
                f"data/ICEWS14/{name}",
                "a\tr\tb\t2014-01-01\n",
            )
    manifest = prepare_archive(archive, tmp_path / "prepared")
    assert sorted(manifest["files"]) == ["test.txt", "train.txt", "valid.txt"]
    assert all(len(record["sha256"]) == 64 for record in manifest["files"].values())


def test_prepare_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        _add_text(handle, "../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe archive path"):
        prepare_archive(archive, tmp_path / "prepared")
