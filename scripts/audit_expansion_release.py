"""Fail-closed audit for the public expansion evidence release."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from scripts.export_expansion_analysis import EXPECTED_RUNS


EXPECTED_BLOCK_LENGTHS = (3, 7, 14, 21)
EXPECTED_ITERATIONS = 20_000
REQUIRED_SOURCE_PATHS = (
    Path("scripts/run_expansion_matrix.py"),
    Path("scripts/export_expansion_analysis.py"),
    Path("scripts/export_expansion_bootstrap.py"),
    Path("scripts/build_expansion_figures.py"),
    Path("configs/expansion"),
    Path("src/riskcal_tkg"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _run_git(project_root: Path, arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _verify_checksum_directory(directory: Path) -> dict[str, str]:
    checksum_path = directory / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise ValueError(f"missing checksum file: {checksum_path}")
    records: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        path = directory / name
        if not path.is_file():
            raise ValueError(f"checksum target is missing: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"checksum mismatch: {path}")
        records[name] = observed
    if not records:
        raise ValueError(f"empty checksum file: {checksum_path}")
    return records


def _resolve_run_root(matrix_root: Path, key: str, record: dict[str, object]) -> Path:
    recorded = Path(str(record.get("run_root", "")))
    if recorded.is_dir():
        return recorded.resolve()
    candidates = [
        path.parent
        for path in (matrix_root / key).glob("*/run_manifest.json")
        if _load_json(path).get("status") == "complete"
    ]
    if len(candidates) != 1:
        raise ValueError(f"could not resolve one complete run for {key}")
    return candidates[0].resolve()


def _audit_matrix(matrix_root: Path) -> tuple[str, dict[str, object]]:
    progress_path = matrix_root / "matrix_progress.json"
    progress = _load_json(progress_path)
    commit = progress.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("matrix progress must contain a full Git commit")
    run_records = progress.get("runs")
    if not isinstance(run_records, dict):
        raise ValueError("matrix progress runs must be a mapping")
    if set(run_records) != set(EXPECTED_RUNS):
        raise ValueError("matrix progress does not contain exactly E1-E6")

    audited: dict[str, object] = {}
    for key in EXPECTED_RUNS:
        record = run_records[key]
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise ValueError(f"incomplete expansion run: {key}")
        run_root = _resolve_run_root(matrix_root, key, record)
        manifest_path = run_root / "run_manifest.json"
        manifest = _load_json(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete run manifest: {run_root}")
        marker_count = len(list((run_root / "conditions").glob("*.complete.json")))
        if marker_count != int(manifest.get("conditions", -1)):
            raise ValueError(f"condition-marker count mismatch: {key}")
        manifest_hash = _sha256(manifest_path)
        if manifest_hash != record.get("run_manifest_sha256"):
            raise ValueError(f"run-manifest hash mismatch: {key}")
        config_path = Path(str(record.get("config", "")))
        if not config_path.is_file():
            fallback = matrix_root.parents[1] / "configs" / "expansion" / f"{key}.yaml"
            config_path = fallback
        if not config_path.is_file() or _sha256(config_path) != record.get(
            "config_sha256"
        ):
            raise ValueError(f"config hash mismatch: {key}")
        for provenance_name in (
            "config.resolved.yaml",
            "environment.json",
            "dataset_manifest.json",
        ):
            if not (run_root / provenance_name).is_file():
                raise ValueError(f"missing run provenance: {run_root / provenance_name}")
        audited[key] = {
            "condition_markers": marker_count,
            "config_sha256": str(record["config_sha256"]),
            "dataset_manifest_sha256": _sha256(run_root / "dataset_manifest.json"),
            "environment_sha256": _sha256(run_root / "environment.json"),
            "run_manifest_sha256": manifest_hash,
        }
    return commit, audited


def _audit_git(project_root: Path, matrix_commit: str, paths: Iterable[Path]) -> str:
    project_root = project_root.resolve()
    head = _run_git(project_root, ["rev-parse", "HEAD"])
    if _run_git(project_root, ["status", "--porcelain", "--untracked-files=all"]):
        raise ValueError("release audit requires a clean Git worktree")
    _run_git(project_root, ["merge-base", "--is-ancestor", matrix_commit, head])

    all_paths = [*REQUIRED_SOURCE_PATHS, *paths]
    files: list[Path] = []
    for relative in all_paths:
        absolute = (project_root / relative).resolve()
        try:
            absolute.relative_to(project_root)
        except ValueError as error:
            raise ValueError(f"release path leaves project root: {relative}") from error
        if absolute.is_dir():
            files.extend(path for path in absolute.rglob("*") if path.is_file())
        elif absolute.is_file():
            files.append(absolute)
        else:
            raise ValueError(f"release path is missing: {relative}")
    if not files:
        raise ValueError("release audit found no published files")
    for path in sorted(set(files)):
        relative = path.relative_to(project_root)
        _run_git(project_root, ["ls-files", "--error-unmatch", str(relative)])
    return head


def audit_expansion_release(
    project_root: Path,
    matrix_root: Path,
    analysis_dir: Path,
    bootstrap_dir: Path,
    figure_dir: Path,
    published_paths: Iterable[Path],
) -> dict[str, object]:
    project_root = project_root.resolve()
    matrix_root = matrix_root.resolve()
    analysis_dir = analysis_dir.resolve()
    bootstrap_dir = bootstrap_dir.resolve()
    figure_dir = figure_dir.resolve()
    matrix_commit, runs = _audit_matrix(matrix_root)
    analysis_checksums = _verify_checksum_directory(analysis_dir)
    bootstrap_checksums = _verify_checksum_directory(bootstrap_dir)
    figure_checksums = _verify_checksum_directory(figure_dir)

    analysis_manifest = _load_json(analysis_dir / "analysis_manifest.json")
    bootstrap_manifest = _load_json(
        bootstrap_dir / "expansion_bootstrap_manifest.json"
    )
    figure_manifest = _load_json(figure_dir / "expansion_figure_manifest.json")
    if analysis_manifest.get("git_commit") != matrix_commit:
        raise ValueError("analysis and matrix commits differ")
    if bootstrap_manifest.get("git_commit") != matrix_commit:
        raise ValueError("bootstrap and matrix commits differ")
    if set(analysis_manifest.get("expected_runs", [])) != set(EXPECTED_RUNS):
        raise ValueError("analysis manifest does not cover E1-E6")
    if set(bootstrap_manifest.get("expected_runs", [])) != set(EXPECTED_RUNS):
        raise ValueError("bootstrap manifest does not cover E1-E6")
    if int(bootstrap_manifest.get("iterations", -1)) != EXPECTED_ITERATIONS:
        raise ValueError("bootstrap iteration count is not frozen at 20,000")
    if tuple(bootstrap_manifest.get("block_lengths", [])) != EXPECTED_BLOCK_LENGTHS:
        raise ValueError("bootstrap block lengths differ from the frozen protocol")
    figure_inputs = figure_manifest.get("inputs")
    if not isinstance(figure_inputs, dict):
        raise ValueError("figure manifest inputs are missing")
    if (
        figure_inputs.get("analysis", {}).get("condition_aggregate.csv")
        != analysis_checksums.get("condition_aggregate.csv")
    ):
        raise ValueError("figure manifest does not bind the audited analysis")
    if (
        figure_inputs.get("bootstrap", {}).get("expansion_bootstrap_summary.csv")
        != bootstrap_checksums.get("expansion_bootstrap_summary.csv")
    ):
        raise ValueError("figure manifest does not bind the audited bootstrap")

    head = _audit_git(project_root, matrix_commit, tuple(published_paths))
    return {
        "analysis_checksums": analysis_checksums,
        "audited_at": datetime.now(UTC).isoformat(),
        "bootstrap_checksums": bootstrap_checksums,
        "figure_checksums": figure_checksums,
        "matrix_commit": matrix_commit,
        "project_commit": head,
        "runs": runs,
        "status": "passed",
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--published-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_expansion_release(
        args.project_root,
        args.matrix_root,
        args.analysis_dir,
        args.bootstrap_dir,
        args.figure_dir,
        args.published_paths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
