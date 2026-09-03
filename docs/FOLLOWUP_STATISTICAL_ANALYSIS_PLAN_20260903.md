# Follow-up Statistical Analysis Plan

Date frozen: 2026-09-03 (Asia/Shanghai)

This plan was written after the deterministic follow-up exports passed their
checksum and completeness audit, but before any bootstrap intervals were
computed. It therefore controls the reported uncertainty analysis but is not a
prospective preregistration of the point estimates.

For each of the four fixed dataset--scorer cases, the analysis reports the
following paired timestamp contrasts at 30% deletion:

1. rolling minus static label-margin calibration for observed-label coverage,
   unique-query full-set coverage, mean set size, and absolute target-error
   reduction;
2. rolling NegScore, Minmax, and Softmax minus rolling label margin for
   observed-label coverage and mean set size;
3. rolling query-max margin minus rolling label margin for unique-query
   full-set coverage, mean set size, and the multi-answer minus single-answer
   full-set-coverage gap.

Coverage and size contrasts use their corresponding label or unique-query
counts as weights. The multi-answer gap is averaged over timestamps for which
both groups are observed. A positive absolute target-error reduction means that
rolling calibration is closer to the nominal 0.90 target.

Uncertainty is estimated with an equal-seed circular moving-block bootstrap.
Each replicate resamples five seeds with replacement and applies a shared block
draw to all sampled seeds. The primary block length is 7 timestamps; lengths 3,
14, and 21 are sensitivity analyses. Every interval uses 20,000 replicates and
seed 20260903 plus a deterministic contrast index. Intervals crossing zero are
described as inconclusive.
