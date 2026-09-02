# RiskCal-TKG

This repository contains code and paper-facing artifacts for a diagnostic study of prequential answer-set calibration in temporal knowledge graph forecasting.

Repository URL: https://github.com/xywang815/riskcal-tkg-w

The study evaluates whether a static conformal-style threshold can become stale on temporally ordered event streams and whether rolling prequential calibration improves empirical observed-label coverage. The final confirmatory matrix covers ICEWS14 and ICEWS05-15, temporal DistMult and a matched-protocol continuous-time complex scorer, five seeds, controlled training-fact deletion, and filtered-negative sensitivity. It also reports prediction-set cost and multi-answer full-set coverage.

## Scope

This is an empirical reliability and utility study. It does not claim a new distribution-free coverage theorem under arbitrary temporal drift, and it does not claim state-of-the-art temporal KG ranking performance.

## Repository Contents

- `src/riskcal_tkg/`: core data, model, calibration, and experiment code.
- `scripts/`: experiment runners and diagnostic exporters.
- `configs/`: experiment configurations.
- `tests/`: regression and smoke tests.
- `data/README.md`: data preparation notes.
- `paper/`: manuscript source, references, figures, and paper-facing derived tables.
- `submission/information/`: MDPI Information submission checklist and cover-letter material.

## Data

The experiments use the public ICEWS14 and ICEWS05-15 temporal knowledge graph benchmarks after deterministic normalization and strict temporal splitting. Raw data files are not committed to this repository. The data preparation scripts record source metadata and hashes in the normalized local data directories.

## Main Experimental Claim

Under the stated protocols, rolling margin calibration restores near-target average observed-label coverage where static margin and KGCP Softmax under-cover. The comparison with KGCP Minmax and NegScore is dataset-dependent, and the reliability repair can require thousands of candidates. Multi-answer full-set coverage remains substantially below single-answer coverage. The repository therefore supports a bounded reliability--utility conclusion, not universal superiority.

## Reproducibility

The project records resolved configurations, seeds, deletion rates, derived paper-facing metric tables, checksum-bound figures, and clean-clone release audits. Large transient artifacts such as checkpoints, raw query dumps, logs, transfer archives, and build products are excluded from Git.

Recommended local setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest -q
```

The full confirmatory matrix requires a CUDA-capable GPU. The committed paper-facing tables and figures can be verified and rebuilt without retraining; the raw checkpoint and query-level exports are intentionally excluded because of size.

Prepare the historical ICEWS14 input used by the original experiment:

```bash
python scripts/prepare_icews14.py --help
```

Prepare ICEWS14 or ICEWS05-15 from the checksum-pinned official TKBC archive:

```bash
python scripts/prepare_tkbc_dataset.py --help
```

Run a configured experiment and summarize a completed run:

```bash
python scripts/run_pilot.py --config configs/icews14_pilot.yaml
python scripts/summarize_results.py --help
```

## License

The code in this repository is released under the MIT License. See `LICENSE`.

## Citation

The GitHub repository is available at https://github.com/xywang815/riskcal-tkg-w. If an archival release is created, cite the corresponding Zenodo DOI alongside the GitHub URL.
