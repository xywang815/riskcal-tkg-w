from pathlib import Path
import json

import pandas as pd
import pytest

from riskcal_tkg.experiment import run_experiment


def test_smoke_produces_required_artifacts(tmp_path: Path) -> None:
    run_root = run_experiment.smoke(output_parent=tmp_path)
    required = {
        "config.resolved.yaml",
        "environment.json",
        "dataset_manifest.json",
        "run_manifest.json",
        "metrics/per_query.parquet",
        "metrics/per_window.csv",
        "metrics/summary.json",
        "SUCCESS_GATE.json",
    }
    files = {
        str(path.relative_to(run_root)).replace("\\", "/")
        for path in run_root.rglob("*")
        if path.is_file()
    }
    assert required <= files
    manifest = json.loads(
        (run_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    rows = pd.read_csv(run_root / "metrics" / "per_window.csv")
    assert {"static", "rolling", "weighted", "adaptive"} <= set(rows["method"])
    assert (rows["calibration_max_timestamp"] < rows["timestamp"]).all()


def test_smoke_writes_condition_checkpoints(tmp_path: Path) -> None:
    run_root = run_experiment.smoke(output_parent=tmp_path)
    checkpoints = sorted((run_root / "checkpoints").glob("*.ckpt"))
    masks = sorted((run_root / "deletion_masks").glob("*.json"))
    assert len(checkpoints) == 2
    assert len(masks) == 2


def test_smoke_writes_evaluated_gate_and_figures(tmp_path: Path) -> None:
    run_root = run_experiment.smoke(output_parent=tmp_path)
    gate = json.loads((run_root / "SUCCESS_GATE.json").read_text(encoding="utf-8"))
    assert gate["status"] == "evaluated"
    for stem in ("coverage_by_time", "set_size_by_corruption", "risk_coverage"):
        assert (run_root / "figures" / f"{stem}.png").stat().st_size > 0
        assert (run_root / "figures" / f"{stem}.pdf").stat().st_size > 0


def test_smoke_evaluates_both_subject_and_object_queries(tmp_path: Path) -> None:
    run_root = run_experiment.smoke(output_parent=tmp_path)
    queries = pd.read_parquet(run_root / "metrics" / "per_query.parquet")
    assert set(queries["prediction_side"]) == {"subject", "object"}
    assert set(queries["method"]) == {
        "top1",
        "static",
        "rolling",
        "weighted",
        "adaptive",
    }


def test_smoke_records_deletion_retention_audit(tmp_path: Path) -> None:
    run_root = run_experiment.smoke(output_parent=tmp_path)
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_root / "deletion_masks").glob("*.json")
    ]
    assert records
    assert all("retention_by_timestamp" in record for record in records)
    assert all("retention_by_relation" in record for record in records)


def test_smoke_records_environment_dataset_and_resource_provenance(tmp_path: Path) -> None:
    run_root = run_experiment.smoke(output_parent=tmp_path)
    environment = json.loads((run_root / "environment.json").read_text(encoding="utf-8"))
    dataset = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    resources = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_root / "resources").glob("*.json")
    ]
    assert environment["git_commit"] is None or len(environment["git_commit"]) == 40
    assert dataset["duplicate_facts_removed"] == 0
    assert dataset["source_files"] == {}
    assert dataset["train_timestamp_range"][1] < dataset["calibration_timestamp_range"][0]
    assert len(resources) == 2
    for record in resources:
        assert record["training_seconds"] >= 0
        assert record["calibration_seconds"] >= 0
        assert record["inference_seconds"] >= 0
        assert record["calibration_protocol"] in {"four_role", "legacy_two_role"}
        assert "half_life_evaluations" in record
        assert "adaptive_selector_fit_samples" in record
        assert "training_score_spread" in record
        assert record["training_score_spread"]["max_row_std"] >= 0.0
        assert record["peak_sampled_rss_bytes"] > 0
        assert record["peak_rss_bytes"] >= record["peak_sampled_rss_bytes"]
        assert len(record["epoch_seconds"]) >= 1


def test_resume_reuses_completed_conditions_with_strict_provenance(tmp_path: Path) -> None:
    run_root = run_experiment.smoke(output_parent=tmp_path)
    checkpoint_times = {
        path.name: path.stat().st_mtime_ns
        for path in (run_root / "checkpoints").glob("*.ckpt")
    }
    (run_root / "run_manifest.json").unlink()
    config = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    resumed = run_experiment(config, resume_root=run_root)
    assert resumed == run_root
    assert json.loads((run_root / "run_manifest.json").read_text())["status"] == "complete"
    assert checkpoint_times == {
        path.name: path.stat().st_mtime_ns
        for path in (run_root / "checkpoints").glob("*.ckpt")
    }


def test_window_metrics_include_singleton_and_abstention_rates(tmp_path: Path) -> None:
    run_root = run_experiment.smoke(output_parent=tmp_path)
    windows = pd.read_csv(run_root / "metrics" / "per_window.csv")
    assert "singleton_rate" in windows
    assert "abstention_rate_at_1" in windows


def test_resume_reuses_checkpoint_for_interrupted_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = run_experiment.smoke(output_parent=tmp_path)
    label = "seed17_delete0p20"
    checkpoint = run_root / "checkpoints" / f"{label}.ckpt"
    checkpoint_time = checkpoint.stat().st_mtime_ns
    for path in (
        run_root / "run_manifest.json",
        run_root / "conditions" / f"{label}.complete.json",
        run_root / "conditions" / f"{label}.windows.csv",
        run_root / "conditions" / f"{label}.queries.parquet",
        run_root / "resources" / f"{label}.json",
    ):
        path.unlink()

    def fail_if_training_restarts(*args: object, **kwargs: object) -> object:
        raise AssertionError("training restarted despite a verified checkpoint")

    monkeypatch.setattr("riskcal_tkg.experiment.train_model", fail_if_training_restarts)
    config = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    resumed = run_experiment(config, resume_root=run_root)
    assert resumed == run_root
    assert checkpoint.stat().st_mtime_ns == checkpoint_time


def test_readme_contains_reproducible_commands() -> None:
    project_root = Path(__file__).parents[1]
    text = (project_root / "README.md").read_text(encoding="utf-8")
    assert "prepare_icews14.py" in text
    assert "run_pilot.py" in text
    assert "summarize_results.py" in text
    assert "target_coverage: 0.9" in (
        project_root / "configs" / "icews14_pilot.yaml"
    ).read_text(encoding="utf-8")
