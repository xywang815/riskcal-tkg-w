# Revision Experiment and Reporting Plan

Frozen: 2026-09-04 (Asia/Shanghai), before the new time-encoder experiments.

## 1. Objective and paper identity

The revision is an audit of prequential answer-set calibration for temporal
knowledge graph forecasting. It will test when recent-history calibration
restores aggregate observed-label coverage, which mechanism produces that
effect, and what prediction-set utility is required. It will not claim a new
distribution-free guarantee or a universal temporal calibration method.

The final manuscript will use `xywang68@iflytek.com` as the correspondence
email. The manuscript will be revised once, after every experiment in this
plan has been frozen and checked.

## 2. Reporting discipline

The final narrative will be organized around the strongest evidence-backed
result, not around the chronological work log. Every main-text experiment must
answer a named research question. Results that do not change the argument will
be removed from the main text or retained only in reproducibility artifacts.

The writing may emphasize the paper's strongest contribution, but it must not
hide outcomes that change the interpretation, switch metrics after seeing the
results, or imply that an exploratory analysis was confirmatory. Trade-offs
will be stated in terms of the intended reliability and utility objectives.

## 3. Primary condition and estimands

The primary condition is zero training-fact deletion. Thirty-percent deletion
is a prespecified stress test, not the headline condition.

The primary reliability estimand is aggregate observed-label coverage at a
nominal target of 0.90. Full-set query coverage is co-primary for multi-answer
queries. Utility is measured by unique-query prediction-set size, including
mean, median, 90th percentile, vocabulary fraction, and the fraction of queries
that return the full vocabulary.

Uncertainty uses equal-seed paired circular moving-block bootstrap intervals
with 20,000 replicates and a primary block length of seven timestamps. Block
lengths 3, 14, and 21 are sensitivity analyses. Seeds are optimization repeats,
not independent data populations.

## 4. P0 experiment: time-encoder confounding

The current temporal DistMult and continuous complex scorers share an
unbounded polynomial-plus-periodic time map. The revision will compare four
prespecified time encodings under the same split, optimizer, negative sampler,
embedding dimension, early stopping rule, and seed set:

1. `none`: constant multiplicative modulation equal to one;
2. `linear`: constant and linear features only;
3. `bounded_fourier`: constant, sine, and cosine features only;
4. `polynomial_fourier`: the current constant, linear, quadratic, sine, and
   cosine features.

The matrix covers ICEWS14 and ICEWS05-15, both scorer families, and seeds
17, 29, 43, 59, and 71 at zero deletion. The existing
`polynomial_fourier` checkpoints may be reused only after checksum and config
identity are verified. A 30% deletion stress subset is run only after the
zero-deletion matrix passes completeness checks.

For each case, timestamp-level diagnostics will report score maximum, score
range, score standard deviation, true-label score, margin nonconformity,
static threshold, rolling threshold, and static/rolling coverage. Spearman
associations with timestamp and early-versus-late test summaries will be
computed without treating timestamps as independent populations.

The principal contrast is rolling-minus-static observed-label coverage within
each dataset, scorer, and time encoding. Query-normalized NegScore, Minmax, and
Softmax variants are prespecified secondary diagnostics. If the repair appears
only under `polynomial_fourier`, the paper will attribute the main phenomenon
to score-scale extrapolation under that encoder. If it persists under bounded
or absent time modulation, the paper may describe the result as robust to the
tested time-map choices, but not as universally general.

## 5. P1 experiments: calibration and feedback

Using frozen checkpoints and strictly past feedback, compare static,
expanding, rolling windows of 250, 500, 1000, and 2000 scores,
exponentially weighted quantiles, and timestamp-batched Adaptive Conformal
Inference. No test result may select a deployment rule.

For every retained dataset-scorer case, evaluate feedback delays of 0, 1, 3,
7, and 14 timestamp batches. Also evaluate prespecified random feedback
retention rates of 0.9 and 0.7 with fixed simulation seeds. Current-timestamp
labels must never enter predictions for that timestamp.

## 6. P1 experiments: multi-answer reliability and utility

Report absolute full-set coverage for single-answer and multi-answer queries.
Stratify diagnostics by recorded answer count (1, 2, 3--5, and more than 5),
relation, and prediction direction. A current query's observed answer count
must not be used to choose its calibrator. Relation/direction-conditional
calibration may be evaluated with a prespecified minimum-history fallback.

Report coverage under candidate budgets 50, 100, 250, 500, and 1000 as a
reliability-utility frontier. Budget truncation is a diagnostic and must not be
described as retaining conformal coverage unless that property is established.

## 7. Evidence gate before manuscript revision

The result batch is frozen only when all planned conditions have complete
manifests, config and script hashes, checkpoint provenance, row-count checks,
and SHA-256 inventories. The analysis will identify exploratory and newly
frozen evidence separately. Only then may the English manuscript, figures,
Chinese close translation, repository release, and submission materials be
updated.
