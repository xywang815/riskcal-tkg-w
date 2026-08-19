# Route A Revision Plan

路线A的目标不是把 RiskCal-TKG 包装成一个已经证明优于所有方法的新算法，而是写成一篇更稳妥的实证论文：

> Static calibration becomes stale in a temporal KG stream; rolling prequential recalibration restores near-target observed-label coverage, but this reliability comes with very large answer sets.

## Current Evidence We Can Safely Claim

- ICEWS14, five seeds, four deletion rates, 20 completed conditions.
- Static calibration under-covers the temporal test stream.
- At 30% history deletion, Static coverage is 0.8235 against a 0.90 target.
- Rolling prequential calibration recovers coverage to 0.8998.
- Timestamp-block bootstrap supports the primary estimand: at 30% deletion,
  the Static-to-Rolling undercoverage reduction is 0.0723 with 95% CI
  [0.0624, 0.0828] and p < 1e-4.
- Block-length sensitivity supports the same conclusion: for block lengths
  3, 7, 14, and 21, the primary 95% CI endpoints range from 0.0600 to 0.0853
  and all one-sided p-values are < 1e-4.
- The cost is large: Rolling mean set size is 3,802.5 out of 7,128 entities.
- Query-level diagnostics support the same conclusion: at 30% deletion,
  Rolling full-set coverage is 0.9020, partial answer recall is 0.9174, and
  36.5% of unique-query sets equal the full vocabulary.
- History-based baselines are now audited. At 30% deletion, train-only Repeat
  reaches MRR 0.2774, below frozen DistMult's 0.3105, while prequential Repeat
  reaches MRR 0.3631. The manuscript must therefore avoid ranking-SOTA wording.
- Fixed Weighted and Adaptive Diagnostic are very close to Rolling.
- Therefore the current evidence supports recent-history recalibration, not a strong adaptive-selector advantage.

## Manuscript Changes Needed

- Title and abstract should emphasize prequential rolling calibration and reliability-utility trade-off.
- Results should lead with Static vs Rolling, not Adaptive vs Static.
- Adaptive should be described as exploratory or diagnostic.
- Remove wording that treats `SUCCESS_GATE.json` as scientific proof.
- Report `p < 1e-4` instead of an over-precise p-value.
- Clearly state that current coverage is label-marginal observed-label coverage.
- Integrate query-level full-set coverage and partial answer recall as supporting diagnostics.

## Minimum Extra Experiments Before Submission

These are the most valuable additions if server time is available:

1. Query-level multi-answer diagnostics. Completed and integrated.
   - Full-set coverage: whether all recorded answers for a query are included.
   - Partial answer recall: fraction of recorded answers included.
   - This directly addresses the peer-review concern about multi-answer KG queries.
   - At 30% deletion, Rolling full-set coverage is 0.9020 and partial answer recall is 0.9174; 36.5% of Rolling query sets equal the full entity vocabulary.

2. Timestamp-block bootstrap.
   - Completed with a paired seed-resampled circular moving-block bootstrap.
   - Uses contiguous seven-timestamp blocks, 20,000 replicates, and the
     strongest deletion setting.
   - Use contiguous timestamp blocks instead of independent timestamp resampling.
   - This makes the statistical wording match temporal dependence.

3. Block-length sensitivity for timestamp-block bootstrap. Completed.
   - Repeats the same paired bootstrap with block lengths 3, 7, 14, and 21.
   - The primary Static-to-Rolling undercoverage reduction remains positive
     across all reported block lengths.

4. Repeat and relation-frequency baselines. Completed.
   - Frequency, relation-frequency, and Repeat are evaluated in train-only and
     prequential history modes.
   - The prequential Repeat result is stronger than the frozen scorer on
     ICEWS14, so the paper is framed as calibration reliability rather than
     ranking superiority.

5. Set-size and utility operating-point audit. Script added; server export pending.
   - Added `scripts/export_set_size_utility.py`.
   - Reuses `per_query.parquet`; it does not retrain the scorer.
   - Treats a maximum prediction-set size as an explicit abstention rule.
   - Reports answer rate, conditional full-set coverage among answered queries,
     unconditional full-set recall, candidate-load reduction, answered set size,
     full-vocabulary answered rate, and timestamp-block bootstrap effects.
   - This directly addresses the largest remaining weakness: Rolling restores
     coverage by returning very large sets.

6. Rolling-window and half-life ablation. Completed and integrated.
   - Added `scripts/export_window_ablation.py`.
   - Reuses completed checkpoints; it does not retrain the scorer.
   - Reports score-count windows 250/500/1000/2000, expanding prequential
     calibration, 3/7/14/30 timestamp-block windows, half-life selection
     frequency, actual pool span, actual Kish effective sample size, and
     deletion-rate interaction effects.
   - This directly addresses the second-round review concern that a 1000-score
     pool may span only a few timestamp blocks and therefore cannot adequately
     test 7/14/30-day half-lives.
   - Key result: the default 1000-score pool spans a median of two timestamp
     blocks, the selector chooses `inf` in 20/20 conditions for that window,
     expanding prequential calibration reaches only 0.8590 coverage at 30%
     deletion, and Rolling-1000 remains near target at 0.8998.

7. Relation-conditional and worst-group coverage. Script added; server export pending.
   - Added `scripts/export_relation_slice_diagnostics.py`.
   - Reuses the completed `per_query.parquet`; it does not retrain the scorer.
   - Reports relation-side observed-label coverage, query-level full-set
     coverage, set size, full-vocabulary set rate, and worst eligible
     relation-side groups.
   - Subject queries are mapped from inverse-relation IDs back to the original
     relation ID before grouping.
   - This addresses the next likely reviewer concern: average coverage can hide
     badly under-covered relations or prediction directions.

## What Codex Can Do Next

- Regenerate tables and figures.
- Rewrite the manuscript to match the final evidence.
- Prepare a response-to-reviewer document and submission checklist.

## What The User Needs To Provide

- Keep the GPU server available if additional experiments are needed.
- Provide the final target journal template once selected.
- Confirm final author list, corresponding author, affiliation, funding, conflict-of-interest statement, and data/code availability wording.
- Create a public repository or approve using a private archive first; a DOI can be minted later through Zenodo after the repository is ready.

## Recommended Route

Because server cost is no longer the bottleneck, the next target is a stronger
journal-facing revision rather than the smallest submit-ready version:

1. complete the window/half-life/expanding/time-window ablation;
2. integrate relation-conditional and worst-group coverage after the server export;
3. add rank-based, normalized-margin, APS/RAPS-style, and set-size-constrained
   nonconformity scores to obtain coverage-size Pareto curves;
4. add label-delay experiments for 1/3/7/14 timestamp blocks;
5. apply rolling calibration to the prequential Repeat scorer;
6. add at least one stronger temporal KG backbone or a second dataset before
   aiming above a conservative EI/SCI-Q4 diagnostic-study submission.

## Query-Level Diagnostics Command

Codex has added and used the exporter:

```bash
scripts/export_query_level_diagnostics.py
```

Because the local bundled Python may not include `pyarrow`, run this on the GPU
server environment that already read and wrote `per_query.parquet`:

```bash
cd /root/riskcal_tkg_mvp
python scripts/export_query_level_diagnostics.py \
  --run-root results/final_confirmatory/20260815T110014908249Z-eb736dbf6658 \
  --paper-root paper
```

The command writes the files now integrated into the manuscript:

```text
paper/data/final_confirmatory/query_level_by_seed.csv
paper/data/final_confirmatory/query_level_summary.csv
paper/data/final_confirmatory/query_level_paper_table.csv
paper/data/final_confirmatory/query_level_manifest.json
```

After running it, package the outputs for local manuscript integration:

```bash
tar -czf /root/riskcal_tkg_query_level_20260816.tar.gz \
  paper/data/final_confirmatory/query_level_by_seed.csv \
  paper/data/final_confirmatory/query_level_summary.csv \
  paper/data/final_confirmatory/query_level_paper_table.csv \
  paper/data/final_confirmatory/query_level_manifest.json
```

## Window/Half-Life Ablation Command

After the latest local project has been uploaded to the server and installed,
run this from `/root/riskcal_tkg_mvp`:

```bash
python scripts/export_window_ablation.py \
  --run-root results/final_confirmatory/20260815T110014908249Z-eb736dbf6658 \
  --paper-root paper \
  --count-windows 250,500,1000,2000 \
  --time-windows 3,7,14,30 \
  --device cuda
```

Package the outputs for local manuscript integration:

```bash
tar -czf /root/riskcal_tkg_window_ablation_20260817.tar.gz \
  paper/data/final_confirmatory/window_ablation_by_timestamp.csv \
  paper/data/final_confirmatory/window_ablation_by_seed.csv \
  paper/data/final_confirmatory/window_ablation_summary.csv \
  paper/data/final_confirmatory/pool_diagnostics_by_timestamp.csv \
  paper/data/final_confirmatory/pool_diagnostics_summary.csv \
  paper/data/final_confirmatory/half_life_selection_by_condition.csv \
  paper/data/final_confirmatory/half_life_selection_summary.csv \
  paper/data/final_confirmatory/deletion_interaction_by_seed.csv \
  paper/data/final_confirmatory/window_ablation_manifest.json
```

## Relation-Conditional/Worst-Group Coverage Command

After the latest local project has been uploaded to the server and installed,
run this from `/root/riskcal_tkg_mvp`:

```bash
python scripts/export_relation_slice_diagnostics.py \
  --run-root results/final_confirmatory/20260815T110014908249Z-eb736dbf6658 \
  --paper-root paper \
  --data-root data/raw/icews14 \
  --target-coverage 0.9 \
  --min-total-labels 250 \
  --min-seed-count 5
```

Package the outputs for local manuscript integration:

```bash
tar -czf /root/riskcal_tkg_relation_slices_20260817.tar.gz \
  paper/data/final_confirmatory/relation_slice_by_seed.csv \
  paper/data/final_confirmatory/relation_slice_summary.csv \
  paper/data/final_confirmatory/relation_worst_group_summary.csv \
  paper/data/final_confirmatory/relation_worst_group_paper_table.csv \
  paper/data/final_confirmatory/relation_slice_manifest.json
```

## Set-Size/Utility Operating-Point Command

After the latest local project has been uploaded to the server and installed,
run this from `/root/riskcal_tkg_mvp`:

```bash
python scripts/export_set_size_utility.py \
  --run-root results/final_confirmatory/20260815T110014908249Z-eb736dbf6658 \
  --paper-root paper \
  --caps 1,10,50,100,250,500,1000,2000,3000,4000,5000,inf \
  --paper-caps 500,1000,2000,3000,4000,5000,inf \
  --bootstrap-method rolling \
  --bootstrap-iterations 20000 \
  --bootstrap-block-length 7
```

Package the outputs for local manuscript integration:

```bash
tar -czf /root/riskcal_tkg_set_size_utility_20260818.tar.gz \
  paper/data/final_confirmatory/set_size_utility_by_timestamp.csv \
  paper/data/final_confirmatory/set_size_utility_by_seed.csv \
  paper/data/final_confirmatory/set_size_utility_summary.csv \
  paper/data/final_confirmatory/set_size_utility_paper_table.csv \
  paper/data/final_confirmatory/set_size_utility_effects.csv \
  paper/data/final_confirmatory/set_size_utility_manifest.json
```
