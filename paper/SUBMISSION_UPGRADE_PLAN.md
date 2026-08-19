# Submission Upgrade Plan

This project is now aligned around Route A: a reliability-utility diagnostic
claim. Temporal ordering and history deletion make static answer-set
calibration under-cover, while rolling prequential recalibration recovers
near-target observed-label coverage with an explicit and substantial set-size
cost. The adaptive half-life selector is treated as an exploratory ablation, not
as the central contribution.

## Current Code Status

- Implemented: TemporalDistMult scorer, controlled training-edge deletion,
  static/rolling/global-weighted conformal prediction sets, per-window and
  per-query artifacts, resume-safe condition checkpoints.
- Added for the paper upgrade: four-role calibration splitting,
  leakage-controlled drift features, batch-adaptive half-life selector,
  `adaptive` method output, adaptive selector metadata, and server-ready config.
- Current empirical interpretation: `adaptive`, `weighted`, and `rolling` are
  very close. This supports the Route A claim about recent-history prequential
  recalibration, not a strong claim that the adaptive selector is superior.
- Not yet implemented: strong temporal KG baselines such as RE-GCN, TANGO, CyGNet,
  or a direct conformal baseline from another paper. These are needed for a
  stronger SCI submission.

## Minimum Route-A Submission Package

The current ICEWS14 run already supplies the minimum diagnostic backbone:

- Dataset: ICEWS14, immutable prepared files and SHA-256 manifest.
- Seeds: `[17, 29, 43, 59, 71]`.
- Deletion rates: `[0.0, 0.1, 0.2, 0.3]`.
- Methods: `top1`, `static`, `rolling`, `weighted`, `adaptive`.
- Main evidence: static undercoverage, rolling recovery, mean and P90
  prediction-set size, normalized size, undercoverage, MRR vs frequency
  baseline, query-level multi-answer diagnostics, runtime, memory, and artifact
  hashes.
- Current timestamp-block inference: at 30% deletion, Static-to-Rolling
  undercoverage reduction is 0.0723 with 95% CI [0.0624, 0.0828] and p < 1e-4.
- Block-length sensitivity: using block lengths 3, 7, 14, and 21, the primary
  undercoverage-reduction CI endpoints range from 0.0600 to 0.0853; all
  one-sided p-values remain < 1e-4.
- History-baseline audit: at 30% deletion, train-only Repeat reaches MRR 0.2774,
  while prequential Repeat reaches MRR 0.3631 and exceeds the frozen DistMult
  scorer's MRR 0.3105. This strengthens honesty but requires non-SOTA ranking
  wording.
- Required claim discipline: report reliability and utility separately. Do not
  claim practical usefulness if coverage is recovered only by very large sets.

Before submission, add the following low-risk improvements:

- Query-level multi-answer diagnostics: completed and integrated. At 30%
  deletion, Rolling reaches 0.9020 full-set coverage and 0.9174 partial answer
  recall, while 36.5% of its unique-query sets equal the full vocabulary.
- Moving-block or timestamp-block bootstrap: completed with paired seed
  resampling and circular seven-timestamp blocks. Block-length sensitivity is
  also completed for block lengths 3, 7, 14, and 21.
- Baseline sanity checks: completed for Frequency, relation-frequency, and
  Repeat in both train-only and prequential history modes.
- Rolling-window and half-life audit: completed and integrated. The default
  1000-score pool spans a median of two timestamp blocks, half-life selection
  chooses equal weighting in 20/20 default-window conditions, expanding
  prequential calibration improves Static but remains below Rolling, and
  count-window/time-window alternatives expose the reliability-utility trade-off.
- Set-size and utility operating-point audit: exporter added; server-side result
  export is pending. It applies explicit maximum-set-size abstention rules to
  the completed per-query artifact and reports the retained coverage, answer
  rate, candidate-load reduction, and timestamp-block uncertainty.
- Relation-conditional and worst-group coverage: exporter added; server-side
  result export is pending. This will show whether average coverage hides
  under-covered relation-side groups and will provide a direct reviewer-facing
  worst-group table.
- Figure/table polish: report the full 0%, 10%, 20%, and 30% deletion table in
  the appendix, and include P90/full-vocabulary set-size rates.
- Manuscript discipline: use `p < 1e-4`, avoid "success gate" as proof, and
  keep `adaptive` as an ablation.

## Stronger SCI-Q4+ Upgrade

After the completed window/half-life audit, add enough methodology and baseline breadth to
make the paper harder to reject for "single-dataset diagnostic only":

- Integrate relation-conditional and worst-group coverage after the server export.
- Compare nonconformity scores: raw margin, rank-based score, normalized margin,
  APS/RAPS-style scores, set-size-constrained calibration, and selective
  prediction.
- Add label-feedback delays of 1/3/7/14 timestamp blocks.
- Apply rolling calibration to the prequential Repeat scorer, because Repeat is
  stronger than the current frozen DistMult scorer on ICEWS14.
- Preferred stronger backbone: RE-GCN, TiRGN, or another stable recurrence-aware
  TKG scorer if implementation time allows.
- Acceptable fallback: a well-documented stronger neural baseline from a stable
  open-source implementation, with frozen hyperparameters.
- Add one additional dataset if server time allows, such as ICEWS18 or GDELT.
- Add ablations: no drift features, no novelty feature, fixed half-life only,
  final-calibration size sensitivity.
- Add an external conformal comparator where feasible: static KGCP-style split
  calibration, ACI-style online update, or a CFEP-inspired temporal calibration
  variant adapted to entity forecasting.

## What Codex Can Do

- Keep editing this repository.
- Add tests, configs, result exporters, and plotting scripts.
- Prepare the exact server commands.
- Help select and integrate a baseline implementation.
- Analyze produced result folders and rewrite the manuscript around the actual
  evidence.
- Produce paper tables, figures, response-to-reviewer notes, and submission
  checklists.

## What The User Must Provide

- A GPU server or cloud notebook with CUDA. Recommended minimum: 1 NVIDIA GPU
  with 16 GB VRAM, 32 GB RAM, 100 GB disk. Better: RTX 4090/A5000/A6000/A10/L4
  class GPU.
- Permission or credentials for the server if Codex is expected to run commands
  directly there.
- Final target journal choice and template.
- Author names, affiliations, funding, conflict-of-interest statement, and data
  availability wording.
- Confirmation that all generated experiments may be used in the paper.

## Server Commands

From the project root on the server:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/check_server_environment.py --require-cuda
python scripts/prepare_icews14.py --config configs/icews14_server_confirmatory.yaml
python scripts/run_pilot.py --config configs/icews14_server_confirmatory.yaml
```

If the run is interrupted:

```bash
python scripts/run_pilot.py \
  --config configs/icews14_server_confirmatory.yaml \
  --resume-run results/<run-id>
```

After completion:

```bash
python scripts/summarize_results.py --run-root results/<run-id> --target 0.9
python scripts/export_paper_diagnostic.py --run-root results/<run-id> --paper-root paper
python scripts/export_window_ablation.py \
  --run-root results/final_confirmatory/<run-id> \
  --paper-root paper \
  --count-windows 250,500,1000,2000 \
  --time-windows 3,7,14,30 \
  --device cuda
python scripts/export_relation_slice_diagnostics.py \
  --run-root results/final_confirmatory/<run-id> \
  --paper-root paper \
  --data-root data/raw/icews14 \
  --target-coverage 0.9 \
  --min-total-labels 250 \
  --min-seed-count 5
python scripts/export_set_size_utility.py \
  --run-root results/final_confirmatory/<run-id> \
  --paper-root paper \
  --caps 1,10,50,100,250,500,1000,2000,3000,4000,5000,inf \
  --paper-caps 500,1000,2000,3000,4000,5000,inf \
  --bootstrap-method rolling \
  --bootstrap-iterations 20000 \
  --bootstrap-block-length 7
```

## Route-A Manuscript Decision

- The current result belongs to Route A: reliable prequential calibration
  diagnostic, not a strong adaptive-selector contribution.
- The headline finding should be Static vs Rolling, especially at 30% deletion:
  Static coverage 0.8235 versus Rolling coverage 0.8998, with Rolling mean set
  size 3,802.5 of 7,128 entities.
- `adaptive` may be retained only as evidence that the present feature-driven
  selector did not materially improve over simple recent-history calibration.
- The paper is more likely to survive review if it is submitted as a careful
  empirical study with honest limitations than as a new state-of-the-art method.
