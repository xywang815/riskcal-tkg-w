from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.audit_expansion_release import audit_expansion_release
from scripts.export_expansion_analysis import EXPECTED_RUNS


MATRIX_COMMIT = "1" * 40


def test_release_auditor_supports_direct_script_invocation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "audit_expansion_release.py"), "--help"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--matrix-root" in result.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_checked(directory: Path, values: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, value in values.items():
        path = directory / name
        if isinstance(value, dict):
            _write_json(path, value)
        else:
            path.write_text(str(value), encoding="utf-8")
        records.append(f"{_sha256(path)}  {name}\n")
    (directory / "SHA256SUMS.txt").write_text("".join(records), encoding="utf-8")


def _git(project: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(project), *arguments], check=True)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    project = tmp_path / "project"
    for path in (
        "scripts/run_expansion_matrix.py",
        "scripts/export_expansion_analysis.py",
        "scripts/export_expansion_bootstrap.py",
        "scripts/build_expansion_figures.py",
        "configs/expansion/e1.yaml",
        "src/riskcal_tkg/model.py",
        "paper/data/expansion/result.csv",
        "paper/figures/expansion/figure.pdf",
    ):
        destination = project / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(path, encoding="utf-8")
    (project / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    _git(project, "init")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "fixture")
    head = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()

    matrix = project / "matrix"
    runs = {}
    for key in EXPECTED_RUNS:
        run_root = matrix / key / "run"
        config = project / "configs" / "expansion" / f"{key}.yaml"
        config.write_text(key, encoding="utf-8")
        _write_json(run_root / "run_manifest.json", {"status": "complete", "conditions": 1})
        _write_json(run_root / "conditions" / "one.complete.json", {})
        (run_root / "config.resolved.yaml").write_text(key, encoding="utf-8")
        _write_json(run_root / "environment.json", {})
        _write_json(run_root / "dataset_manifest.json", {})
        runs[key] = {
            "status": "complete",
            "config": str(config),
            "config_sha256": _sha256(config),
            "run_root": str(run_root),
            "run_manifest_sha256": _sha256(run_root / "run_manifest.json"),
        }
    _write_json(matrix / "matrix_progress.json", {"git_commit": head, "runs": runs})

    analysis = project / "paper" / "data" / "expansion"
    analysis_manifest = {
        "git_commit": head,
        "expected_runs": list(EXPECTED_RUNS),
    }
    _write_checked(
        analysis,
        {
            "condition_aggregate.csv": "run,coverage\n",
            "analysis_manifest.json": analysis_manifest,
        },
    )
    bootstrap = project / "paper" / "data" / "bootstrap"
    bootstrap_manifest = {
        "git_commit": head,
        "expected_runs": list(EXPECTED_RUNS),
        "iterations": 20_000,
        "block_lengths": [3, 7, 14, 21],
    }
    _write_checked(
        bootstrap,
        {
            "expansion_bootstrap_summary.csv": "statistic,observed\n",
            "expansion_bootstrap_manifest.json": bootstrap_manifest,
        },
    )
    figures = project / "paper" / "figures" / "expansion"
    figure_manifest = {
        "inputs": {
            "analysis": {
                "condition_aggregate.csv": _sha256(
                    analysis / "condition_aggregate.csv"
                )
            },
            "bootstrap": {
                "expansion_bootstrap_summary.csv": _sha256(
                    bootstrap / "expansion_bootstrap_summary.csv"
                )
            },
        }
    }
    _write_checked(
        figures,
        {
            "expansion_generalization.pdf": "pdf",
            "expansion_figure_manifest.json": figure_manifest,
        },
    )
    _git(project, "add", ".")
    _git(project, "commit", "-m", "evidence")
    return project, matrix, analysis, bootstrap, figures


def test_audit_expansion_release_accepts_complete_tracked_evidence(
    tmp_path: Path,
) -> None:
    project, matrix, analysis, bootstrap, figures = _fixture(tmp_path)
    cache = project / "src" / "riskcal_tkg" / "__pycache__" / "model.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"ignored cache")

    audit = audit_expansion_release(
        project,
        matrix,
        analysis,
        bootstrap,
        figures,
        (
            Path("paper/data/expansion"),
            Path("paper/data/bootstrap"),
            Path("paper/figures/expansion"),
        ),
    )

    assert audit["status"] == "passed"
    assert set(audit["runs"]) == set(EXPECTED_RUNS)


def test_audit_expansion_release_rejects_incomplete_matrix(tmp_path: Path) -> None:
    project, matrix, analysis, bootstrap, figures = _fixture(tmp_path)
    progress_path = matrix / "matrix_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["runs"][EXPECTED_RUNS[0]]["status"] = "running"
    _write_json(progress_path, progress)

    with pytest.raises(ValueError, match="incomplete expansion run"):
        audit_expansion_release(
            project,
            matrix,
            analysis,
            bootstrap,
            figures,
            (Path("paper/data/expansion"),),
        )
