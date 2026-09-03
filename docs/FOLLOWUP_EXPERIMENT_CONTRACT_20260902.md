# Follow-up Calibration Experiment Contract

Date frozen: 2026-09-02 (Asia/Shanghai)

## Purpose

This follow-up study separates two choices that were coupled in the released
experiment matrix: the nonconformity score and the temporal history used to
estimate its threshold. It also evaluates a query-level calibration target for
multi-answer temporal queries. The manuscript remains locked until these
analyses complete and their outputs pass checksum and consistency checks.

## Reused evidence

The study reuses the trained checkpoints from the completed expansion matrix at
commit `52beca5c0ded61dc071c410ccf9cbd05691ac92e`. No checkpoint is retrained for
the component ablation. Each checkpoint is verified against its condition
marker before inference.

The fixed primary conditions are the Cartesian product of:

- datasets: ICEWS14 and ICEWS05-15;
- scorers: temporal DistMult and the matched-protocol continuous complex scorer;
- seeds: 17, 29, 43, 59, and 71;
- training-fact deletion: 0.30;
- training-negative sampling: filtered.

## A. Component ablation

The ablation crosses four nonconformity scores with three history policies.

| Axis | Fixed levels |
| --- | --- |
| Score | margin, KGCP NegScore, KGCP Minmax, KGCP Softmax with temperature 1 |
| History | static final-calibration interval, expanding strictly past history, rolling most recent 1000 revealed label scores |

The scorer, split, test queries, target coverage (0.90), and finite-sample order
statistic are held fixed. The principal outcomes are observed-label coverage,
unique-query full-set coverage, partial answer recall, and unique-query mean set
size. This is a diagnostic factorial comparison; it does not reproduce the
published KGCP training pipeline.

## B. Query-level calibration

For each unique temporal query, define its calibration score as the maximum
margin nonconformity among all recorded answers at that timestamp. The static,
expanding, and rolling-1000 thresholds are estimated from these unique-query
scores. During testing, all sets at timestamp `t` are emitted before the
recorded answer sets at `t` are used to update history.

The primary comparison is query-level rolling max-margin versus label-level
rolling margin. The primary outcome is full-set coverage. Secondary outcomes
are observed-label coverage, partial answer recall, mean set size, and the
multi-answer minus single-answer full-set-coverage gap. A larger set is reported
as a cost, not hidden by the coverage result.

## C. Additional backbone decision

The official TKBC repository and the published TComplEx optimizer are audited
separately. A reference TComplEx result may enter the paper only if the model,
optimizer, temporal split, inverse-relation training, hyperparameters, and
checkpoint provenance are all reproducible under the present prequential
protocol. A custom implementation or changed optimizer will be named as such
and will not be described as an exact reproduction.

## Fixed interpretation rules

1. No method is selected from test coverage or test set size.
2. A 95% interval crossing zero is inconclusive.
3. Label-marginal and full-set query coverage are distinct estimands.
4. Current-timestamp labels never enter a threshold used at that timestamp.
5. Negative, mixed, and utility-adverse results remain in the evidence record.
6. The paper is revised once, after the full follow-up batch is frozen.

