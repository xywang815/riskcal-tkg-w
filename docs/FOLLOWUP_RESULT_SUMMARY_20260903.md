# Follow-up Calibration Result Summary

Date frozen: 2026-09-03 (Asia/Shanghai)

## Integrity

- Deterministic analysis commit: `7f19fe7538738311fd783d76dc8b72fde7822665`
- Bootstrap commit: `d6c9602ec37081599ab5767e77c24a2bb1ef1b4d`
- Conditions: 20 (two datasets, two scorers, five seeds)
- Timestamp-method rows: 131,550
- Bootstrap replicates: 20,000 per contrast and block length
- Primary block length: 7; sensitivity lengths: 3, 14, and 21
- File checksum verification: passed
- Duplicate primary keys: zero
- Missing aggregate metrics: zero

## Primary findings

All intervals below are 95% circular moving-block bootstrap intervals at block
length 7. Differences are paired within seed and timestamp.

### History policy

Rolling minus static label-margin calibration increased observed-label
coverage in every fixed case:

| Case | Coverage gain | 95% interval | Absolute target-error reduction | Mean set-size change |
| --- | ---: | ---: | ---: | ---: |
| ICEWS14, temporal DistMult | 0.0798 | [0.0621, 0.1006] | 0.0646 | +1,175 |
| ICEWS14, continuous complex | 0.0802 | [0.0612, 0.1014] | 0.0669 | +1,158 |
| ICEWS05-15, temporal DistMult | 0.1011 | [0.0921, 0.1102] | 0.0811 | +2,026 |
| ICEWS05-15, continuous complex | 0.1025 | [0.0938, 0.1111] | 0.0824 | +2,038 |

The target-error and set-size intervals are strictly positive for all four
cases and all tested block lengths. The defensible interpretation is that
recent-history calibration corrects undercoverage here, but pays a substantial
efficiency cost.

### Score function within rolling history

Minmax retained label coverage indistinguishable from rolling margin in the two
ICEWS14 cases while reducing mean set size by 1,650--1,773 entities. On
ICEWS05-15 it reduced mean size by 1,091--1,187 entities, but the small coverage
loss was sensitive to block length and must be described as mixed rather than
uniformly beneficial.

NegScore reduced mean set size by 854--1,016 entities on ICEWS14 without a
resolved coverage change. On ICEWS05-15 it produced slightly lower coverage and
larger sets on the primary analysis; some interval conclusions changed across
block lengths. Softmax with temperature 1 raised coverage by 0.069--0.085 but
increased mean size by 2,081--4,991 entities, revealing marked overcoverage and
poor efficiency rather than a free reliability gain.

### Query-level objective

Replacing label-level rolling margin by a query-max rolling score did not fix
multi-answer coverage. Relative to label-level rolling margin, it changed
full-set coverage by:

| Case | Full-set coverage change | 95% interval | Mean set-size change | Multi-minus-single gap change |
| --- | ---: | ---: | ---: | ---: |
| ICEWS14, temporal DistMult | -0.0081 | [-0.0107, -0.0058] | -141 | -0.0137 |
| ICEWS14, continuous complex | -0.0070 | [-0.0093, -0.0049] | -123 | -0.0112 |
| ICEWS05-15, temporal DistMult | -0.0011 | [-0.0022, 0.0001] | -39 | -0.0018 |
| ICEWS05-15, continuous complex | -0.0014 | [-0.0024, -0.0003] | -48 | -0.0022 |

The negative ICEWS14 changes persist across block lengths. The query-max rule
uses fewer calibration units and shifts the empirical quantile; its apparently
more conservative per-query score does not guarantee a larger finite-sample
threshold. This is a negative result and must remain visible.

## Claims allowed in the manuscript

1. Under this fixed protocol, recent-history calibration consistently repairs
   static label-margin undercoverage across two datasets and two scorer
   families, at a large set-size cost.
2. History policy and score geometry interact: Minmax can be much more efficient
   than margin on ICEWS14, whereas Softmax substantially overcovers.
3. Marginal label coverage does not imply multi-answer full-set reliability.
4. A simple query-max calibration objective is not an adequate remedy in these
   experiments.

## Claims prohibited

1. Universal superiority of rolling calibration or any score function.
2. Distribution-free validity under arbitrary temporal drift.
3. Successful reproduction of published KGCP, TComplEx, or TNTComplEx.
4. A claim that fact deletion is the main cause of static miscalibration.
5. A claim that query-level max calibration solves multi-answer degradation.
