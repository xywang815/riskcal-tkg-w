# RiskCal-TKG

This repository contains code and paper-facing artifacts for a diagnostic study of prequential answer-set calibration in temporal knowledge graph forecasting.

Repository URL: https://github.com/xywang815/riskcal-tkg-w

The study evaluates whether static conformal-style calibration becomes stale on an ICEWS14 temporal event stream and whether rolling prequential calibration improves empirical observed-label coverage. It also reports the utility cost of calibrated answer sets, including query-level coverage, set-size diagnostics, relation-side worst-group diagnostics, delayed-feedback sensitivity, rolling-window ablations, rank shortlists, and validation-selected RAPS rolling.

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

The experiments use the public ICEWS14 temporal knowledge graph benchmark after deterministic normalization and strict temporal splitting. Raw data files are not committed to this repository. The data preparation script records hashes and produces the normalized local data directory.

## Main Experimental Claim

Under the stated ICEWS14 protocol, static calibration under-covers later timestamps, while rolling prequential calibration restores near-target average observed-label coverage. The same reliability repair can require large answer sets, so the paper reports reliability and candidate-set utility jointly.

## Reproducibility

The project records resolved configurations, seeds, deletion rates, derived paper-facing metric tables, and figures. Large transient artifacts such as checkpoints, logs, transfer archives, and build products are excluded from Git.

Recommended local setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest -q
```

The full confirmatory run requires a CUDA-capable GPU. Paper-facing diagnostic tables can be reproduced from completed run artifacts using the scripts in `scripts/`.

## License

The code in this repository is released under the MIT License. See `LICENSE`.

## Citation

The GitHub repository is available at https://github.com/xywang815/riskcal-tkg-w. A Zenodo DOI should be added after a public archival release is created.
