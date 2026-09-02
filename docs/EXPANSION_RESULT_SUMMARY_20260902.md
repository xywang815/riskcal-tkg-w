# RiskCal-TKG Expansion Result Summary

Date: 2026-09-02 (Asia/Shanghai)

This document records the manuscript-facing interpretation of the completed
E1--E6 expansion matrix. It is derived only from the checksum-verified analysis
and timestamp-block-bootstrap outputs. Values from pilot runs, screenshots, or
terminal summaries are excluded.

## Evidence identity

- Source matrix commit: `52beca5c0ded61dc071c410ccf9cbd05691ac92e`.
- Complete matrix: 90/90 seed/deletion conditions.
- Aggregated evidence: 286,480 timestamp-window rows and 28,626,600 query rows,
  including 4,858,050 multi-answer label rows.
- Analysis manifest SHA-256:
  `2ef91e419cd7888c9a88e28c1392b9fbfacba339ab265612a647050e182bdad0`.
- Bootstrap summary SHA-256:
  `4c1e3b97f208e126375a914c33b254bb576dfc2c8c8b696b2ad8dbaba4bd5e01`.
- Figure manifest SHA-256:
  `5cc7c6815eecd3db02c0a41e1310cf8f37083e0a9821914f3e031c0567693116`.
- Pre-manuscript release-audit commit:
  `25df0b167fc6b26f80b23094a18208c0fa1576c6`.
- Pre-manuscript release-audit SHA-256:
  `93348d41108aa9587565ff0ca081fe356c29f704b2862cf10625937d1037c5de`;
  status `passed` after 124 tests in a fresh server clone.
- Primary uncertainty analysis: 20,000-iteration equal-seed circular
  moving-block bootstrap, block length 7, with lengths 3, 14, and 21 retained
  as sensitivity checks.

## Scope and method identity

The added evidence covers ICEWS14 and ICEWS05-15, temporal DistMult and a
matched-protocol continuous-time complex-valued scorer, and filtered/uniform
training corruption sensitivity. These are two related event knowledge-graph
benchmarks and two repository implementations; they do not establish universal
generalization to all temporal knowledge graphs or all forecasting models.

`static_margin` is a static split-conformal max-score-margin baseline, not
published KGCP. The three methods prefixed with `kgcp_` reproduce the published
NegScore, query-wise Minmax, and Softmax score transformations under a matched
static split-calibration protocol. They are not an exact reproduction of the
published authors' complete training pipeline. The Softmax temperature is 1.

## Primary reliability and utility results

At 30% training-fact deletion with filtered negative sampling, rolling-margin
coverage and mean prediction-set size were:

| Dataset | Scorer | Rolling coverage | Rolling mean size |
| --- | --- | ---: | ---: |
| ICEWS14 | temporal DistMult | 0.8996 | 3870.6 |
| ICEWS14 | continuous complex | 0.8988 | 3818.0 |
| ICEWS05-15 | temporal DistMult | 0.9002 | 4307.2 |
| ICEWS05-15 | continuous complex | 0.9002 | 4260.1 |

The corresponding rolling-minus-static-margin coverage gains were 0.0798
[0.0621, 0.0999], 0.0802 [0.0612, 0.1012], 0.1011
[0.0921, 0.1102], and 0.1025 [0.0937, 0.1111], respectively.

The KGCP comparisons are heterogeneous rather than uniformly favorable. On
ICEWS05-15, rolling calibration exceeded KGCP Minmax by 0.0436
[0.0387, 0.0486] with DistMult and 0.0471 [0.0418, 0.0524] with the complex
scorer. On ICEWS14, rolling-minus-KGCP-Minmax was -0.0017
[-0.0074, 0.0042] and -0.0010 [-0.0072, 0.0058], respectively. The analogous
ICEWS14 contrasts against KGCP NegScore were also inconclusive. KGCP Minmax and
NegScore therefore already reached approximately nominal coverage on ICEWS14.

Reliability came with larger prediction sets. At 30% deletion, the ratio of
rolling mean size to KGCP Minmax mean size was 1.67 and 1.63 on ICEWS14, and
2.17 and 2.22 on ICEWS05-15, for DistMult and the complex scorer respectively.
The ratio against static margin ranged from 1.41 to 1.85. The final manuscript
must present this reliability--utility trade-off and must not describe rolling
calibration as uniformly superior.

## Multi-answer diagnostic

At 30% deletion, multi-answer minus single-answer full-set-coverage gaps for
rolling margin were:

| Dataset | Scorer | Gap | 95% interval |
| --- | --- | ---: | ---: |
| ICEWS14 | temporal DistMult | -0.2792 | [-0.2992, -0.2600] |
| ICEWS14 | continuous complex | -0.3054 | [-0.3306, -0.2806] |
| ICEWS05-15 | temporal DistMult | -0.2174 | [-0.2323, -0.2023] |
| ICEWS05-15 | continuous complex | -0.2294 | [-0.2445, -0.2135] |

Rolling calibration reduced the gap relative to static margin and KGCP Softmax,
but did not remove it. For example, ICEWS14 DistMult rolling full-set coverage
was 0.6532 for multi-answer queries and 0.9305 for single-answer queries. This
failure mode is a primary result and must be disclosed in the abstract/results
when space allows, the discussion, and the conclusion.

## Deletion and negative-sampling diagnostics

The tested deletion mechanism did not materially worsen static calibration.
For deletion 0.30 minus 0.00, all static-margin coverage intervals crossed zero:

| Dataset | Scorer | Static coverage change | 95% interval |
| --- | --- | ---: | ---: |
| ICEWS14 | temporal DistMult | -0.0024 | [-0.0086, 0.0038] |
| ICEWS14 | continuous complex | -0.0006 | [-0.0049, 0.0046] |
| ICEWS05-15 | temporal DistMult | -0.0023 | [-0.0053, 0.0004] |
| ICEWS05-15 | continuous complex | -0.0004 | [-0.0018, 0.0011] |

The deletion experiment therefore does not support a claim that training-fact
deletion is the principal cause of static miscalibration. Its role is a
controlled stress axis; the motivation must focus on temporal mismatch and
prequential reliability diagnostics.

Filtered-minus-uniform coverage effects were small. Rolling-margin intervals
crossed zero in all four paired settings. Static margin showed a small positive
effect for ICEWS05-15 DistMult without deletion (0.0020 [0.0005, 0.0038]); the
remaining static intervals crossed zero. This sensitivity analysis does not
support a broad claim that filtered training negatives materially change the
calibration conclusions.

## Allowed conclusions

1. Under the tested prequential protocol, rolling-margin calibration restored
   near-nominal aggregate coverage where static margin and KGCP Softmax
   undercovered, across both datasets and both scorers.
2. The benefit is dataset- and baseline-dependent: on ICEWS14, KGCP Minmax and
   NegScore already achieved near-nominal coverage with substantially smaller
   prediction sets.
3. Near-nominal aggregate coverage can require large prediction sets, especially
   on ICEWS05-15.
4. Multi-answer queries remain a pronounced failure mode even after rolling
   calibration.
5. The tested deletion and negative-sampling interventions produced mostly
   small or inconclusive coverage changes.

## Prohibited conclusions

- Universal superiority of the rolling method over KGCP.
- A distribution-free guarantee under arbitrary temporal drift.
- Exact reproduction of the published KGCP or TComplEx training pipelines.
- Evidence that training-fact deletion causes static calibration failure.
- Evidence that filtered negative sampling materially improves calibration in
  general.
- Generalization beyond the two ICEWS event benchmarks and two tested scorers.

## Figure QA

The three checksum-bound figure pairs were visually inspected at full-page
resolution. All panels were nonblank and complete; axes, legends, intervals,
and annotations were readable; no panel was cropped or overlapped. The figures
are suitable for manuscript integration. Public-availability language remains
blocked until the final post-manuscript audit and GitHub release are complete.
