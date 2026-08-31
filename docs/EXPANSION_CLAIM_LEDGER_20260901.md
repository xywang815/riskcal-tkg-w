# Expansion Evidence and Claim Ledger

This file is the sole bridge between the locked expansion experiments and the
later one-pass manuscript revision. It records what may be claimed, which final
artifact supports it, and which wording remains prohibited. Pilot values and
terminal screenshots are not paper evidence.

## Manuscript lock

- The manuscript remains locked until E1-E6, the expansion analysis export, the
  timestamp-block bootstrap, figure generation, checksum verification, and a
  clean-clone reproduction check have all passed.
- Results are entered here before prose is revised.
- Null, adverse, or non-monotone results remain reportable and are never replaced
  by a more convenient pilot value.

## Method identity

| Output name | Permitted description | Prohibited description |
| --- | --- | --- |
| `static_margin` | Static split-conformal max-score margin baseline | KGCP; original KGCP |
| `rolling_margin` | Rolling prequential max-score margin calibration | Distribution-free guarantee under arbitrary temporal drift |
| `kgcp_negscore_static` | Published KGCP NegScore nonconformity with matched static split calibration | Exact reproduction of the authors' complete training pipeline |
| `kgcp_minmax_static` | Published KGCP query-wise Minmax nonconformity with matched static split calibration | Exact reproduction of the authors' complete training pipeline |
| `kgcp_softmax_static` | Published KGCP Softmax nonconformity at temperature 1 with matched static split calibration | Temperature-tuned KGCP or calibrated softmax unless separately evaluated |
| `temporal_distmult` | The repository's temporal DistMult scorer under the frozen prequential protocol | State-of-the-art TKG model |
| `continuous_tcomplex` | Matched-protocol continuous-time complex-valued scorer | Exact official TComplEx optimizer or checkpoint |

## Frozen evidence map

| Reviewer issue | Formal evidence | Acceptance condition | Required reporting if condition fails | Status |
| --- | --- | --- | --- | --- |
| Baseline identity | E1-E4 condition summaries for all three KGCP score transforms plus `static_margin` | Names, score definitions, temperature, split, and calibration role are explicit | State that these are matched-protocol score-transform baselines, not an exact end-to-end reproduction | Running |
| Dataset generalization | E2 and E4 on official ICEWS05-15 | Complete five-seed manifests, no leakage, finite nondegenerate scores | Describe any dataset-specific failure and avoid universal claims | Running |
| Scorer generalization | E3 and E4 with `continuous_tcomplex` | Complete five-seed manifests, no leakage, finite nondegenerate scores | Call the evidence model-specific and retain DistMult as the only positive setting if necessary | Running |
| Filtered negative sampling | E5 paired with E2; E6 paired with E3 | Same seeds, timestamps, deletion rates, model family, and evaluation labels | Report the direction and uncertainty even if filtered sampling has negligible or adverse effect | Running |
| Multi-answer degradation | Formal `per_query.parquet` from E1-E4; answer-set reconstruction by query | `answer_count` agrees with distinct labels; single/multi full-set coverage gap is estimable | Disclose magnitude prominently and discuss why label-wise and full-set coverage differ | Running |
| Deletion motivation | E1-E4 deletion 0 versus 0.30 paired contrasts | Timestamp-block interval excludes zero in a consistent adverse direction | Narrow the motivation to temporal calibration mismatch; explicitly report null/non-monotone deletion effects | Running |
| Public traceability | Final commits, result checksums, clean-clone commands, release/tag | Final scripts/configs/CSV/checksums reproduce figures and are public | Manuscript cannot claim artifact availability until the public release resolves | Pending formal outputs |

## Formal output authority

The following files will be paper-authoritative only after the matrix is marked
complete and their manifests/checksums pass:

1. `condition_by_seed.csv` and `condition_aggregate.csv`: primary coverage,
   prediction-set size, and ranking point estimates.
2. `deletion_effects_by_seed.csv`: deletion 0.10/0.20/0.30 changes from deletion
   zero without a monotonicity assumption.
3. `method_contrasts_by_seed.csv`: rolling-margin comparisons against the margin
   and published KGCP score-transform baselines.
4. `sampling_contrasts_by_seed.csv`: filtered-minus-uniform paired sensitivity.
5. `multi_answer_*.csv`: full-set coverage, partial answer recall, and size by
   single- versus multi-answer query.
6. `expansion_bootstrap_summary.csv`: primary 20,000-iteration timestamp-block
   intervals at block length 7 and sensitivity lengths 3, 14, and 21.
7. `analysis_manifest.json`, `expansion_bootstrap_manifest.json`, and both
   `SHA256SUMS.txt` files: input provenance and output integrity.

## Predeclared interpretation rules

- The primary target is 0.90 and the primary deletion contrast is 0.30 minus
  zero; intermediate deletion rates remain visible.
- Coverage comparisons use query-count weighting; undercoverage and
  multi-answer gaps use timestamp-macro summaries unless the output says
  otherwise.
- Five seeds are the between-training-run unit. Timestamp dependence is handled
  with a circular moving-block bootstrap, not by treating individual query rows
  as independent replicates.
- A confidence interval crossing zero is reported as inconclusive, not as proof
  of no effect.
- A small p-value never substitutes for effect size, interval, set-size cost, or
  the applicable exchangeability caveat.
- ICEWS14 and ICEWS05-15 are two related event-KG benchmarks; they do not justify
  claims about all temporal knowledge graphs.

## Result-entry table

Fill this table only from verified final CSV/JSON files.

| Evidence item | Point estimate | 95% interval / seed SD | Source file and SHA-256 | Allowed manuscript conclusion |
| --- | ---: | ---: | --- | --- |
| E1 rolling vs static/KGCP at deletion 0.30 | Pending | Pending | Pending | Pending |
| E2 added-dataset result | Pending | Pending | Pending | Pending |
| E3 added-scorer result | Pending | Pending | Pending | Pending |
| E4 joint boundary result | Pending | Pending | Pending | Pending |
| E5/E6 filtered-sampling sensitivity | Pending | Pending | Pending | Pending |
| Multi-answer full-set coverage gap | Pending | Pending | Pending | Pending |
| Static deletion effect | Pending | Pending | Pending | Pending |

## Final revision gate

The manuscript may be unlocked only when every row above has a verified source
or an explicit negative result, all figures are generated from those sources,
and the public release contains the exact scripts and configuration identities.
