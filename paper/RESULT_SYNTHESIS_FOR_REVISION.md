# RiskCal-TKG Revision Result Synthesis

Last updated: 2026-08-19

Purpose: this file is the evidence-backed synthesis to guide the next unified manuscript rewrite. It is an internal writing control document, not text to paste directly into the paper. Local paths, archive names, and shell commands listed here must not appear in the main manuscript.

## Revised Central Thesis

The paper should be framed as a reliability-utility study for temporal knowledge graph prediction sets.

Static calibration becomes stale in a temporal KG stream. Rolling prequential calibration restores near-target observed-label and query-level reliability under temporal deletion stress, but it often produces very large candidate sets. Rank-based and adaptive mass shortlist variants reduce average set size with small observed-label coverage changes, but full-set answer coverage and tail-size behavior reveal real trade-offs. The correct claim is therefore not ranking superiority or universal dominance, but a documented reliability-utility trade-off with operational diagnostics.

## Evidence-Backed Claims

### 1. Static Calibration Is Unreliable Under Temporal Deletion

Evidence files:

- `paper/data/final_confirmatory/main_calibration_table.csv`
- `paper/data/final_confirmatory/timestamp_block_bootstrap_summary.csv`

At deletion rate 0.3:

| Method | Micro coverage | Macro-time coverage | Positive undercoverage | Mean size | Median size | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Static | 0.823484 | 0.819283 | 0.080765 | 2748.586346 | 748.297295 | 0.310518 |
| Rolling | 0.899782 | 0.899540 | 0.008502 | 3802.518862 | 4653.754037 | 0.310518 |

Timestamp-block bootstrap at deletion rate 0.3:

- Rolling undercoverage reduction vs static: 0.072263, 95% CI [0.053460, 0.093802], p = 0.00005.
- Rolling micro-coverage gain vs static: 0.076298, 95% CI [0.059520, 0.096221], p = 0.00005.

Manuscript claim allowed:

- Rolling calibration substantially reduces undercoverage relative to static calibration under temporal deletion.
- The claim is supported across seeds and timestamp-block bootstrap resampling.

Manuscript claim not allowed:

- Do not claim that rolling improves the underlying ranking model. The scorer is unchanged.

### 2. The Scorer Is Not a Ranking SOTA Claim

Evidence files:

- `paper/data/final_confirmatory/main_calibration_table.csv`
- `paper/data/final_confirmatory/history_baseline_summary.csv`

At deletion rate 0.3:

- Frozen DistMult scorer MRR: 0.310518.
- Train-only frequency MRR: 0.092539.
- Train-only repeat MRR: 0.277446.
- Prequential repeat MRR: 0.363136.
- Prequential frequency MRR: 0.093926.
- Prequential relation-frequency MRR: 0.153217.

Manuscript claim allowed:

- The learned scorer is a fixed base scorer used to study calibration.
- The scorer is stronger than simple train-only frequency baselines, but a prequential repeat baseline can exceed its MRR.

Manuscript claim not allowed:

- Do not present RiskCal-TKG as a temporal KG forecasting SOTA model.
- Do not imply that calibration improves MRR; the main calibration methods share the same scorer ranking.

### 3. Rolling Calibration Improves Query-Level Reliability But Increases Large Sets

Evidence file:

- `paper/data/final_confirmatory/query_level_paper_table.csv`

At deletion rate 0.3:

| Method | Full-set coverage | Partial answer recall | Multi-answer full-set coverage | Multi-answer partial recall | p90 size | Full-vocabulary rate |
|---|---:|---:|---:|---:|---:|---:|
| Static | 0.826669 | 0.846309 | 0.498924 | 0.692889 | 7128 | 0.244094 |
| Rolling | 0.901979 | 0.917410 | 0.651061 | 0.803468 | 7128 | 0.364710 |

Manuscript claim allowed:

- Rolling calibration improves full-set and partial answer coverage at query level, including multi-answer queries.
- This comes with large candidate sets; p90 set size reaches the full vocabulary size in the current experimental setting.

Manuscript claim not allowed:

- Do not hide the large-set behavior. It is central to the reliability-utility trade-off.

### 4. Simple Hard Caps Are a Negative Diagnostic, Not the Main Solution

Evidence files:

- `paper/data/final_confirmatory/set_size_utility_paper_table.csv`
- `paper/data/final_confirmatory/set_size_utility_effects.csv`

For rolling calibration at deletion rate 0.3:

| Cap | Answer rate | Conditional full-set coverage | Unconditional full-set recall | Mean candidate load | Candidate-load reduction |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.327771 | 0.744437 | 0.244035 | 35.226628 | 0.991336 |
| 1000 | 0.372400 | 0.756748 | 0.281863 | 67.365365 | 0.983436 |
| 2000 | 0.415917 | 0.771788 | 0.321046 | 131.569260 | 0.967644 |
| 3000 | 0.431086 | 0.777159 | 0.335072 | 192.108243 | 0.952767 |
| 4000 | 0.439595 | 0.780411 | 0.343128 | 248.013215 | 0.939028 |
| 5000 | 0.457567 | 0.787878 | 0.360541 | 277.576163 | 0.931753 |
| Uncapped | 1.000000 | 0.901979 | 0.901979 | 4069.435597 | 0.000000 |

Cap 5000 effect estimates:

- Candidate load saved: 3791.859434, 95% CI [3671.536202, 3911.027721], p = 0.00005.
- Candidate-load reduction: 0.930825, 95% CI [0.926645, 0.935114].
- Unconditional full-set recall loss: 0.541438, 95% CI [0.525271, 0.558133].
- Conditional full-set coverage delta: -0.114523, 95% CI [-0.122204, -0.107192].

Manuscript claim allowed:

- Hard caps sharply reduce candidate load but lose many answerable queries and substantially reduce unconditional full-set recall.
- Hard caps are useful as an operational diagnostic showing why naive truncation is not enough.

Manuscript claim not allowed:

- Do not present hard caps as the proposed practical solution.

### 5. Rank Shortlists Reduce Mean Size With Small Label-Coverage Change, But Full-Set Coverage Drops

Evidence files:

- `paper/data/final_confirmatory/shortlist_calibration_paper_table.csv`
- `paper/data/final_confirmatory/shortlist_calibration_effects.csv`

At deletion rate 0.3:

| Method | Observed-label coverage | Full-set coverage | Partial recall | Mean size | Median size | p90 size | Full-vocabulary rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Margin rolling | 0.899782 | 0.901979 | 0.917410 | 4069.435473 | 5746.219426 | 7128 | 0.364710 |
| Rank rolling | 0.899181 | 0.888052 | 0.892568 | 3668.160876 | 3668.160876 | 3668.160876 | 0.000000 |

Rank rolling vs margin rolling:

- Mean size reduction: 401.274597, 95% CI [148.984405, 651.723614], p = 0.000550.
- Relative mean reduction: 0.093508, 95% CI [0.032192, 0.154275], p = 0.001100.
- p90 size reduction: 3459.839124, 95% CI [3190.743903, 3707.934311].
- Observed-label coverage delta: -0.000601, 95% CI [-0.004717, 0.003378].
- Full-set coverage delta: -0.013927, 95% CI [-0.018389, -0.009465].

Manuscript claim allowed:

- Rank rolling reduces mean and tail set sizes while preserving observed-label coverage within a small empirical range.
- This reduction has a measurable full-set coverage cost.

Manuscript claim not allowed:

- Do not say rank rolling is strictly better. It trades some full-set coverage for smaller sets.

### 6. Adaptive Mass Shortlisting Improves Average Size Further, But Does Not Dominate Rank in the Tail

Evidence files:

- `paper/data/final_confirmatory/score_adaptive_shortlist_paper_table.csv`
- `paper/data/final_confirmatory/score_adaptive_shortlist_effects.csv`

At deletion rate 0.3:

| Method | Observed-label coverage | Full-set coverage | Partial recall | Mean size | Median size | p90 size | Full-vocabulary rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Margin rolling | 0.899782 | 0.901979 | 0.917410 | 4069.435473 | 5746.219426 | 7128.000000 | 0.364710 |
| Rank rolling | 0.899181 | 0.888052 | 0.892568 | 3668.160876 | 3668.160876 | 3668.160876 | 0.000000 |
| APS rolling | 0.897855 | 0.904540 | 0.916654 | 4911.812466 | 6906.063744 | 7126.940292 | 0.000000 |
| Adaptive mass rolling | 0.898710 | 0.887440 | 0.893369 | 3318.965316 | 2924.649393 | 4646.492393 | 0.000000 |

Adaptive mass rolling vs margin rolling:

- Mean size reduction: 750.470157, 95% CI [427.132861, 1072.834159], p = 0.00005.
- Relative mean reduction: 0.181713, 95% CI [0.102037, 0.259042].
- p90 size reduction: 2481.507607, 95% CI [2283.414519, 2667.428404].
- Observed-label coverage delta: -0.001073, 95% CI [-0.005248, 0.002931].
- Full-set coverage delta: -0.014539, 95% CI [-0.018825, -0.010329].

Adaptive mass rolling vs rank rolling:

- Mean size reduction: 349.195560, 95% CI [246.017675, 464.973824], p = 0.00005.
- Relative mean reduction: 0.100875, 95% CI [0.066488, 0.138727].
- p90 reduction is negative: -978.331517, 95% CI [-1086.153655, -875.357824].
- Label and full-set coverage deltas vs rank are small and should be reported as trade-offs, not dominance.

Manuscript claim allowed:

- Adaptive mass rolling reduces average and median set size relative to margin and rank rolling at target 0.90.
- It does not dominate rank rolling on tail size; rank rolling has a smaller p90 set size.

Manuscript claim not allowed:

- Do not claim universal adaptive superiority.
- Do not treat APS alone as a successful utility fix; in this run APS increases mean size relative to margin rolling.

### 7. Sensitivity Analysis Supports Targets 0.88 and 0.90, But Not 0.92

Evidence file:

- `paper/data/final_confirmatory/score_adaptive_sensitivity_success_table.csv`

Grid:

- Target coverages: 0.88, 0.90, 0.92.
- Selection tolerances: 0.00, 0.01, 0.02, 0.03.

Summary:

- Target 0.88: 4/4 tolerance settings satisfy the success criteria for mean-size reduction vs margin and rank.
- Target 0.90: 4/4 tolerance settings satisfy the success criteria for mean-size reduction vs margin and rank.
- Target 0.92: 0/4 tolerance settings satisfy the margin-comparison criterion; high target coverage removes the average-size advantage vs margin.
- Tail size is not uniformly better than rank for any grid cell.

Manuscript claim allowed:

- The adaptive mass shortlist is empirically useful at moderate targets, especially 0.88 and 0.90.
- At the higher target 0.92, the average-size advantage over margin is not supported by the current evidence.

Manuscript claim not allowed:

- Do not claim the adaptive method is robust for all target coverages.

### 8. Delayed Feedback Weakens But Does Not Remove the Rolling Advantage

Evidence file:

- `paper/data/final_confirmatory/delay_feedback_effects.csv`

At deletion rate 0.3:

| Extra delay blocks | Undercoverage reduction vs static | 95% CI | Micro-coverage gain vs static | 95% CI |
|---:|---:|---|---:|---|
| 0 | 0.072263 | [0.053445, 0.093573] | 0.076298 | [0.059328, 0.096328] |
| 1 | 0.071067 | [0.052289, 0.092387] | 0.075098 | [0.058216, 0.095134] |
| 3 | 0.067819 | [0.049179, 0.088198] | 0.071872 | [0.055053, 0.091349] |
| 7 | 0.063675 | [0.045067, 0.082730] | 0.067738 | [0.050960, 0.086766] |

All p-values are 0.00005 in the exported table.

Manuscript claim allowed:

- Delayed feedback reduces the estimated benefit but does not eliminate it within the tested delay range.

Manuscript claim not allowed:

- Do not claim arbitrary feedback delays are covered.

### 9. Window and Half-Life Diagnostics Explain Why the Default Rolling Window Works

Evidence files:

- `paper/data/final_confirmatory/window_ablation_summary.csv`
- `paper/data/final_confirmatory/pool_diagnostics_summary.csv`
- `paper/data/final_confirmatory/half_life_selection_summary.csv`

At deletion rate 0.3:

| Method | Coverage | Mean size |
|---|---:|---:|
| Static | 0.823484 | 2748.586506 |
| Expanding | 0.858985 | 3180.987175 |
| Rolling count 250 | 0.912867 | 4045.008239 |
| Rolling count 500 | 0.900057 | 3821.961110 |
| Rolling count 1000 | 0.899782 | 3802.518743 |
| Rolling count 2000 | 0.896399 | 3752.428806 |
| Time window 7 | 0.892890 | 3699.712058 |
| Time window 14 | 0.887444 | 3615.344341 |
| Time window 30 | 0.876552 | 3438.336268 |

Pool diagnostics for count 1000:

- The 1000-score pool spans mean 2.424658 timestamp blocks.
- Median span is 2 blocks.
- Effective sample sizes are approximately 996 to 1000.

Half-life selection:

- Count 1000 selects half-life `inf` in 20/20 conditions.
- Count 250 selects half-life 7 in 20/20 conditions.
- Count 500 selects half-life 7 in 19/20 conditions.
- Count 2000 is mixed: `inf` in 12/20, 7 in 7/20, and 14 in 1/20.

Manuscript claim allowed:

- The rolling window itself is the main recentness mechanism in the default setting.
- Exponential half-life selection is weakly identified when the rolling pool already spans only a few timestamp blocks.

Manuscript claim not allowed:

- Do not overstate half-life tuning as an independent source of improvement in the default setting.

### 10. Relation-Side Groups Still Expose Residual Risk

Evidence file:

- `paper/data/final_confirmatory/relation_worst_group_paper_table.csv`

At deletion rate 0.3:

| Method | Eligible groups | Min coverage | p10 coverage | Median coverage | Fraction below target | Worst undercoverage | Worst full-set coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static | 76 | 0.627389 | 0.770761 | 0.878699 | 0.631579 | 0.272611 | 0.578592 |
| Rolling | 76 | 0.762500 | 0.871312 | 0.948375 | 0.223684 | 0.137500 | 0.794052 |

Manuscript claim allowed:

- Rolling calibration improves relation-side worst-group behavior but does not eliminate residual undercoverage in all groups.

Manuscript claim not allowed:

- Do not claim group-conditional coverage is guaranteed.

## Claims To Avoid In The Paper

1. Do not claim temporal KG forecasting SOTA.
2. Do not claim calibration improves MRR.
3. Do not claim arbitrary distribution-free temporal drift guarantees.
4. Do not claim adaptive mass rolling universally dominates rank rolling.
5. Do not claim hard caps are an adequate practical solution.
6. Do not claim robust utility gains at target coverage 0.92.
7. Do not put local paths, terminal commands, tar archive names, or screenshots in the main manuscript.
8. Do not describe the study as if it used multiple benchmark datasets unless those experiments are actually added later.

## Recommended Results-Section Structure

1. Experimental setup and evaluation protocol.
   - Dataset: ICEWS14 only, unless more datasets are later added.
   - Temporal split, deletion-rate stress protocol, seeds, target coverage, and metrics.
   - Explain observed-label coverage, query-level full-set coverage, partial answer recall, set size, and candidate-load metrics.

2. Reliability failure of static calibration.
   - Main table: static vs rolling at deletion rates 0.0, 0.1, 0.2, 0.3.
   - Emphasize deletion rate 0.3 in text because it shows the strongest stress case.

3. Query-level utility and answer-set behavior.
   - Show that rolling improves query-level reliability but increases large sets and full-vocabulary outputs.

4. Negative hard-cap diagnostic.
   - Show why naive size caps save load but lose many answerable queries.
   - Keep this as a diagnostic subsection, not as the proposed method.

5. Shortlist calibration.
   - Rank rolling: reduces mean and tail size with small observed-label coverage change, but full-set coverage declines.
   - Adaptive mass rolling: improves mean and median size further at target 0.90, but tail-size dominance is not supported.

6. Robustness and boundary conditions.
   - Timestamp-block bootstrap.
   - Delay-feedback analysis.
   - Target/tolerance sensitivity.
   - Window and half-life diagnostics.

7. Group-level residual risk.
   - Relation-side worst-group table.
   - State that average reliability does not eliminate subgroup risk.

8. Scope limitations.
   - One dataset.
   - Frozen scorer.
   - Not ranking SOTA.
   - Adaptive utility is target dependent.

## Recommended Table And Figure Plan

Main paper:

- Table 1: Dataset, split, training, calibration, and evaluation protocol.
- Table 2: Main calibration results by deletion rate and method.
- Table 3: Query-level coverage and set-size diagnostics at deletion rate 0.3.
- Table 4: Shortlist comparison at deletion rate 0.3.
- Table 5: Sensitivity summary for adaptive mass rolling at targets 0.88, 0.90, and 0.92.
- Figure 1: Reliability-utility trade-off curve, preferably coverage vs mean set size.
- Figure 2: Coverage over timestamp blocks for static vs rolling.

Supplement or appendix:

- Hard cap table.
- Timestamp-block bootstrap details.
- Delay feedback details.
- Window and half-life diagnostics.
- Relation-side worst-group table.
- File hashes and reproducibility manifests.

## Abstract Wording Skeleton

Use this only as a factual draft, not final polished text:

> We study conformal prediction-set calibration for temporal knowledge graph link prediction under temporally evolving score distributions. Using ICEWS14 with controlled temporal deletion stress, we compare static calibration with rolling prequential calibration and shortlist variants. Static calibration undercovers under temporal deletion, whereas rolling calibration restores near-target observed-label and query-level coverage. However, this reliability gain increases candidate-set size. Rank and adaptive mass shortlist variants reduce average set size with small observed-label coverage changes, while full-set coverage and tail-size diagnostics reveal remaining utility trade-offs. Timestamp-block bootstrap, delay-feedback, relation-group, and target-sensitivity analyses show that the method is useful under moderate coverage targets but should be interpreted as a reliability-utility trade-off rather than a ranking improvement.

Important: remove or revise this if later experiments add datasets or change the base scorer.

## Conclusion Wording Skeleton

Use this only as a factual draft:

> The experiments indicate that temporal calibration matters for TKG prediction sets. In the tested ICEWS14 setting, static thresholds become unreliable under deletion stress, while rolling prequential thresholds substantially reduce undercoverage. The same mechanism increases set size, making utility diagnostics necessary. Shortlist calibration reduces average candidate load, but the trade-off depends on the target coverage and on whether observed-label coverage, full-set coverage, or tail size is prioritized. These results support RiskCal-TKG as an operational calibration framework and motivate future work on stronger base scorers, additional datasets, and group-conditional calibration.

## Next Manuscript Actions

1. Do not edit the manuscript piecemeal after every new experiment.
2. After the current experiment batch is complete, rewrite the manuscript in one consolidated pass.
3. Remove local paths, shell commands, archive names, and screenshot-like content from the main text.
4. Fix the large blank areas around Section 7 by reorganizing floats and table placement during the final LaTeX pass.
5. Replace broad claims with the evidence-backed claims above.
6. Move reproducibility implementation details to appendix or a separate replication package section.
7. Regenerate the PDF only after the consolidated rewrite, then inspect page layout.

