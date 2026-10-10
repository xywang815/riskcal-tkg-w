# RiskCal-TKG

Code for temporal knowledge graph calibration experiments with controlled time encodings, score normalization, online feedback, answer-set coverage and candidate budgets.

The current matrix uses ICEWS14 and ICEWS05-15, matched DistMult and continuous complex scorers, four time encoders (`none`, `linear`, `bounded_fourier`, `polynomial_fourier`) and five seeds: 16 configurations and 80 model conditions, all at zero training-edge deletion. The seeds are `17, 29, 43, 59, 71`. This count describes the evaluation matrix, not the number of newly trained models in a historical experiment.

This repository contains source code, runtime configurations and tests. Datasets, checkpoints, generated results and manuscripts are not bundled. Each command below runs from the repository root and writes generated files under ignored `data/` or `results/` directories.

## Install and test

Use Python 3.11 or newer. The full training configurations require a CUDA-capable PyTorch installation; the toy smoke run and candidate replay use CPU. Install the PyTorch build appropriate to your machine, then install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
```

Dependencies are declared in `pyproject.toml`; they are not a version lock for a historical environment. Keep this editable checkout available because the experiment runner also invokes `scripts/summarize_results.py`.

Run a small CPU experiment without downloading data:

```bash
python -m scripts.run_pilot --config configs/smoke.yaml --output-parent results/smoke
```

The runner prints its run directory and automatically writes summaries, diagnostics, checkpoints and completion markers there. The smoke configuration is a toy test, not one of the benchmark conditions.

## Prepare benchmarks

The preparation script downloads the [TKBC benchmark archive](https://dl.fbaipublicfiles.com/tkbc/data.tar.gz), checks SHA-256 `2a993856622981535067a5ba54a5c649e7b50bf6ba0cb2197c17b2e9c069d25e`, and extracts the selected dataset and its license. The source is [facebookresearch/tkbc](https://github.com/facebookresearch/tkbc); the ICEWS data record is [doi:10.7910/DVN/28075](https://doi.org/10.7910/DVN/28075).

```bash
python -m scripts.prepare_tkbc_dataset --dataset icews14 --output data/raw/icews14
python -m scripts.prepare_tkbc_dataset --dataset icews05_15 --output data/raw/icews05_15
```

Use `--archive /path/to/data.tar.gz` to reuse a local copy. The experiment loader combines the archive's train/valid/test files, deduplicates quadruples and creates chronological train/calibration/test splits containing 60%/20%/20% of timestamps. It does not reuse the archive's original split. Benchmark data retain their own license terms.

## Train the current matrix

Run from a Git checkout so manifests can record the full source commit. The matrix controller defaults to all 16 files in `configs/time_encoder_audit/`; each file contains the five seeds and the unchanged experiment settings.

```bash
python -m scripts.check_server_environment --require-cuda
python -m scripts.run_expansion_matrix --output-parent results/matrix
```

The script name is retained for compatibility. To select configurations explicitly, add `--configs configs/time_encoder_audit/*.yaml`. A single configuration can also be run through `scripts.run_pilot`; the complete replay pipeline below requires the controller's `matrix_progress.json` and all 80 completed conditions.

Re-running the controller with the same source and output directory resumes incomplete runs and verifies completed ones. Keep configurations, source and recorded paths stable during a run. Use a fresh output directory after changing them. For a source deployment without `.git`, the controller accepts `--git-commit` with the actual full source commit identifier.

## Replay calibration and candidate utility

Run these stages in order after the full matrix completes. They consume locally generated checkpoints, score caches and manifests; no pretrained artifact download is provided. The source identifier is captured from your checkout, and the replay stages verify input identities before proceeding.

```bash
RISKCAL_COMMIT="$(git rev-parse HEAD)"

python -m scripts.export_time_encoder_audit \
  --matrix-root results/matrix --output-root results/analysis/time_encoder

python -m scripts.export_normalized_time_gate \
  --matrix-root results/matrix --output-root results/analysis/gate

python -m scripts.export_cached_calibration_followups \
  --phase online --gate-root results/analysis/gate \
  --output-root results/analysis/online --analysis-git-commit "$RISKCAL_COMMIT"

python -m scripts.export_cached_calibration_followups \
  --phase feedback --gate-root results/analysis/gate \
  --online-root results/analysis/online --output-root results/analysis/feedback \
  --analysis-git-commit "$RISKCAL_COMMIT"

python -m scripts.export_candidate_utility \
  --matrix-root results/matrix --gate-root results/analysis/gate \
  --p1-root results/analysis --output-root results/analysis/utility \
  --analysis-git-commit "$RISKCAL_COMMIT" --workers 4
```

The normalized gate and candidate-utility replay score candidates on CPU. Online and delayed-feedback stages reuse cached label scores. Bootstrap exports use 20,000 draws and block lengths 3, 7, 14 and 21 by default. These empirical resampling summaries describe the observed traces; they do not establish future-stream coverage guarantees. Candidate caps describe returned entities and do not reduce the full-vocabulary scoring computation.

The separate `scripts.export_followup_calibration_study` entry point accepts `--spec configs/followup/time_encoder_multianswer_utility.yaml`, `--matrix-root` and `--output-root` for configurable answer-set replay. It is not a replacement for the gate/online/feedback prerequisites of `scripts.export_candidate_utility`.

## Ranking, verification and diagnostics

After candidate utility completes, replay rankings, attach the corresponding size histograms and verify the generated compact rank arrays:

```bash
python -m scripts.export_ranking_topk_revision \
  --matrix-root results/matrix --gate-root results/analysis/gate \
  --utility-root results/analysis/utility --output results/ranking --workers 4

python -m scripts.freeze_ranking_topk_revision \
  --utility results/analysis/utility --output results/ranking \
  --export-script scripts/export_ranking_topk_revision.py

python -m scripts.verify_ranking_topk_revision \
  --data-root results/ranking --utility-root results/analysis/utility \
  --output results/ranking_verification.json

python -m scripts.analyze_review_diagnostics \
  --frozen-root results/analysis --output-root results/diagnostics \
  --gate-timestamps results/analysis/gate/gate_by_timestamp.csv
```

Ranking export verifies checkpoints and cached predictions. Compact-array verification recalculates metrics from generated ranks without rescoring candidates; add `--budget 750 --budget-output results/top750_by_seed.csv` to evaluate another returned-candidate cap. The diagnostic script reads generated gate, online and utility summaries and writes score-threshold, window-composition, temporal-bin and realized-budget diagnostics. Run any module with `--help` for its available arguments.

## Code layout

- `src/riskcal_tkg/`: data loading, models, calibration, metrics, experiment execution and candidate-utility calculations.
- `configs/`: the current matrix, toy smoke configuration and answer-set replay specification.
- `scripts/`: benchmark preparation, experiment orchestration, replay, aggregation and verification.
- `tests/`: synthetic and unit tests for the retained code.

## License

MIT License

Copyright (c) 2026 Xinyu Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
