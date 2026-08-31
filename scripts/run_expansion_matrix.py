from argparse import ArgumentParser
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

from riskcal_tkg.experiment import run_experiment


DEFAULT_CONFIGS = (
    Path("configs/expansion/icews14_distmult_filtered.yaml"),
    Path("configs/expansion/icews14_tcomplex_filtered.yaml"),
    Path("configs/expansion/icews05_15_distmult_filtered.yaml"),
    Path("configs/expansion/icews05_15_tcomplex_filtered.yaml"),
    Path("configs/expansion/icews05_15_distmult_uniform_sensitivity.yaml"),
    Path("configs/expansion/icews14_tcomplex_uniform_sensitivity.yaml"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _incomplete_run(parent: Path) -> Path | None:
    if not parent.is_dir():
        return None
    candidates: list[Path] = []
    for path in sorted(parent.iterdir()):
        if not path.is_dir():
            continue
        manifest = path / "run_manifest.json"
        if not manifest.is_file():
            candidates.append(path)
            continue
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            candidates.append(path)
    if len(candidates) > 1:
        raise RuntimeError(f"multiple incomplete runs found under {parent}")
    return candidates[0] if candidates else None


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_CONFIGS),
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("results/expansion_formal"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    progress_path = args.output_parent / "matrix_progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(project_root),
            "runs": {},
        }
    runs = progress.setdefault("runs", {})
    if not isinstance(runs, dict):
        raise ValueError("matrix progress runs field must be a mapping")

    for config_path in args.configs:
        config_path = config_path.resolve()
        key = config_path.stem
        existing = runs.get(key)
        if isinstance(existing, dict) and existing.get("status") == "complete":
            run_root = Path(str(existing["run_root"]))
            manifest_path = run_root / "run_manifest.json"
            if manifest_path.is_file() and json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("status") == "complete":
                print(f"[matrix] verified completed {key}: {run_root}", flush=True)
                continue

        run_parent = args.output_parent / key
        resume_root = _incomplete_run(run_parent)
        runs[key] = {
            "status": "running",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "started_at": datetime.now(UTC).isoformat(),
            "resume_root": None if resume_root is None else str(resume_root),
        }
        _write_progress(progress_path, progress)
        print(
            f"[matrix] {'resuming' if resume_root else 'starting'} {key}",
            flush=True,
        )
        try:
            run_root = run_experiment(
                config_path,
                output_parent=run_parent,
                resume_root=resume_root,
            )
        except BaseException as error:
            runs[key]["status"] = "failed"
            runs[key]["failed_at"] = datetime.now(UTC).isoformat()
            runs[key]["error"] = f"{type(error).__name__}: {error}"
            _write_progress(progress_path, progress)
            raise
        manifest_path = run_root / "run_manifest.json"
        runs[key] = {
            "status": "complete",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "run_root": str(run_root.resolve()),
            "run_manifest_sha256": _sha256(manifest_path),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        _write_progress(progress_path, progress)
        print(f"[matrix] completed {key}: {run_root}", flush=True)


if __name__ == "__main__":
    main()
