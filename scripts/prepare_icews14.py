from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from typing import Any
from urllib.request import urlopen

from riskcal_tkg.artifacts import atomic_write_json, sha256_file
from riskcal_tkg.config import load_config


DEFAULT_URL = (
    "https://raw.githubusercontent.com/Lee-zix/RE-GCN/master/"
    "data-release.tar.gz"
)
EXPECTED_FILES = ("train.txt", "valid.txt", "test.txt")


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return path


def prepare_archive(archive: Path, destination: Path) -> dict[str, Any]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if destination.exists():
        raise FileExistsError(destination)

    candidates: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            path = _safe_member_path(member.name)
            if not member.isfile() or path.name not in EXPECTED_FILES:
                continue
            if not any(part.upper().startswith("ICEWS14") for part in path.parts):
                continue
            candidates[str(path.parent)][path.name] = member
        complete = [
            (parent, files)
            for parent, files in candidates.items()
            if set(files) == set(EXPECTED_FILES)
        ]
        if not complete:
            raise ValueError("archive does not contain a complete ICEWS14 dataset")
        parent, members = sorted(complete, key=lambda item: item[0])[0]
        destination.mkdir(parents=True, exist_ok=False)
        for name in EXPECTED_FILES:
            source = handle.extractfile(members[name])
            if source is None:
                raise ValueError(f"could not read {parent}/{name}")
            (destination / name).write_bytes(source.read())

    manifest: dict[str, Any] = {
        "archive": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "archive_dataset_path": parent,
        "files": {},
    }
    for name in EXPECTED_FILES:
        path = destination / name
        manifest["files"][name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    atomic_write_json(destination / "dataset_manifest.json", manifest)
    return manifest


def _download_to_temporary(url: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = tempfile.NamedTemporaryFile(
        dir=directory,
        prefix="icews14-",
        suffix=".tar.gz",
        delete=False,
    )
    path = Path(descriptor.name)
    try:
        with descriptor, urlopen(url, timeout=120) as response:
            while block := response.read(1024 * 1024):
                descriptor.write(block)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    config = load_config(args.config)
    archive = args.archive
    downloaded = False
    if archive is None:
        archive = _download_to_temporary(args.url, config.data_path.parent)
        downloaded = True
    try:
        prepare_archive(archive, config.data_path)
    finally:
        if downloaded:
            archive.unlink(missing_ok=True)
    print(config.data_path.resolve())


if __name__ == "__main__":
    main()
