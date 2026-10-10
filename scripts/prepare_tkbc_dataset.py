from argparse import ArgumentParser
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen


ARCHIVE_URL = "https://dl.fbaipublicfiles.com/tkbc/data.tar.gz"
ARCHIVE_SHA256 = "2a993856622981535067a5ba54a5c649e7b50bf6ba0cb2197c17b2e9c069d25e"
DATASETS = {
    "icews14": "ICEWS14",
    "icews05_15": "ICEWS05-15",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(ARCHIVE_URL, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if sha256_file(temporary) != ARCHIVE_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("downloaded TKBC archive has an unexpected SHA-256")
    temporary.replace(path)


def extract_dataset(archive: Path, dataset: str, output: Path) -> dict[str, object]:
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("TKBC archive SHA-256 does not match the preregistered source")
    source_name = DATASETS[dataset]
    prefix = f"src_data/{source_name}/"
    output.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, object]] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for source_file, target_name in (
            ("train", "train.txt"),
            ("valid", "valid.txt"),
            ("test", "test.txt"),
            ("LICENSE", "LICENSE"),
        ):
            member = bundle.getmember(prefix + source_file)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member.name)
            target = output / target_name
            with target.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            outputs[target_name] = {
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
    manifest = {
        "dataset": source_name,
        "prepared_at": datetime.now(UTC).isoformat(),
        "source_archive_url": ARCHIVE_URL,
        "source_archive_sha256": ARCHIVE_SHA256,
        "source_repository": "https://github.com/facebookresearch/tkbc",
        "source_data_record": "https://doi.org/10.7910/DVN/28075",
        "files": outputs,
    }
    (output / "SOURCE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    arguments = parser.parse_args()
    archive = arguments.archive or arguments.output.parent / "tkbc_data.tar.gz"
    if not archive.is_file():
        download_archive(archive)
    manifest = extract_dataset(archive, arguments.dataset, arguments.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
