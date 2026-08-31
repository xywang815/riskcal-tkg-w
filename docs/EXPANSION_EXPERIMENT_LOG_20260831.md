# RiskCal-TKG Expansion Experiment Contract and Ledger

## Purpose

This document is the authoritative working record for the post-review expansion
started on 2026-08-31. It prevents experiment details from being reconstructed
from memory and separates evidence generation from manuscript writing.

## Hard gates

1. Do not edit the manuscript, its figures, or submission prose until every
   experiment below is either completed and verified or explicitly recorded as
   infeasible with evidence.
2. Preserve negative, null, and mixed findings. Do not tune acceptance criteria
   after inspecting test outcomes.
3. Every paper-facing number must resolve to a run directory, configuration,
   script version, output checksum, and verification entry in this ledger.
4. Never store server passwords, personal access tokens, or other credentials in
   this repository, experiment logs, shell scripts, or command-history exports.
5. The final paper must distinguish implementation-matched baselines from
   published methods. A modified baseline may not be named as the original
   published method without a precise qualification.

## Frozen starting point

- Local repository: `/Users/wangxinyu/Documents/Codex/riskcal_tkg_mvp`
- Public repository: `https://github.com/xywang815/riskcal-tkg-w`
- Starting Git commit: `9992a10f9d1a6fbb951dddc899db847a70a84e37`
- Existing confirmatory run:
  `/root/riskcal_tkg_mvp/results/final_confirmatory/20260815T110014908249Z-eb736dbf6658`
- Existing primary setting: ICEWS14, temporal DistMult-style scorer, five seeds
  `{17, 29, 43, 59, 71}`, deletion rates `{0.0, 0.1, 0.2, 0.3}`, target
  coverage `0.90`.
- Manuscript editing status: **LOCKED pending experiment completion**.

The working tree contains a newer 2026-08-31 manuscript/submission snapshot that
has not yet been committed. It is source material only; it is not the baseline
for new empirical claims until it is frozen in Git.

## Reviewer concerns translated into acceptance conditions

### A. Baseline identity

- Locate and cite the primary source and official implementation, if available,
  for the published KGCP method.
- Produce a method-by-method audit of calibration split, score definition,
  candidate set, filtering, temporal ordering, and online update behavior.
- Implement either the published KGCP protocol faithfully or label the closest
  reproducible implementation as a qualified approximation.
- Include at least one additional conformal-KG comparator when technically and
  scientifically compatible with temporal link prediction.

### B. Cross-dataset evidence

- Add at least one established temporal knowledge-graph dataset with a documented
  source, license/access route, temporal split, entity/relation counts, and data
  checksum.
- Run the full primary comparison on the added dataset using at least five fixed
  seeds unless runtime or dataset constraints are documented before inspection.

### C. Cross-model evidence

- Add at least one scoring architecture that is materially different from the
  existing DistMult-style model.
- Keep evaluation, calibration, corruption, and statistical procedures matched
  across models. Record any unavoidable model-specific tuning separately.

### D. Filtered negative-sampling sensitivity

- Compare the current uniform corruption scheme with a filtered variant that
  rejects known positive objects for the sampled `(subject, relation, time)`
  query.
- Predeclare the filtering scope before the run: training-only positives for
  optimization, with evaluation filtering unchanged.
- Report predictive utility, coverage, set size, convergence/runtime, and any
  interaction with deletion rate.

### E. Multi-answer behavior

- Report unique-query and answer-label weighted coverage separately.
- Stratify by answer multiplicity, including single-answer and multi-answer
  queries, and report set size/recall degradation with uncertainty.
- Include worst-group and relation-side diagnostics. Mixed or adverse results
  must be carried into the final discussion.

### F. Corruption motivation

- Test the deletion-rate interaction rather than assuming monotonic degradation.
- Report effect estimates and uncertainty for static and rolling calibration at
  every deletion rate.
- If deletion does not materially worsen static calibration, narrow the paper's
  motivation to temporal drift/reliability diagnostics and state that the tested
  deletion mechanism is not the main driver.

### G. Public traceability

- Track all final experiment scripts, configurations, paper-facing CSV/JSON
  outputs, checksums, and figure-generation scripts.
- Verify the public default branch from a clean clone.
- Create a self-describing release/tag only after the final manuscript and
  reproduction bundle pass verification.

## Planned experiment matrix

Choices marked `TBD after source and implementation audit` are deliberately not
filled from memory.

| Axis | Starting level | Required added level | Status |
| --- | --- | --- | --- |
| Dataset | ICEWS14 | TBD after primary-source audit | Pending |
| Scorer | Temporal DistMult-style | TBD different architecture | Pending |
| Negative sampling | Uniform object corruption | Training-positive filtered | Pending |
| Conformal baseline | Current static/rolling score methods | Published KGCP-compatible + comparator | Pending |
| Deletion | 0.0, 0.1, 0.2, 0.3 | Same, with interaction analysis | Pending |
| Seeds | 17, 29, 43, 59, 71 | Same unless predeclared otherwise | Pending |
| Multi-answer | Existing diagnostics | Cross-dataset/model stratification | Pending |

## Evidence and reporting rules

- Primary target coverage: `0.90`.
- Primary deletion effect setting: `0.30`, retained for comparability; all rates
  remain visible.
- Primary uncertainty unit for temporal dependence: timestamp-block bootstrap.
- Bootstrap iterations: `20,000`; primary block length: `7`; sensitivity block
  lengths: `3, 14, 21` when applicable.
- Report point estimates and 95% intervals, not only pass/fail gates or p-values.
- Mark exploratory model/dataset tuning separately from confirmatory evaluation.
- Do not use test outcomes to select conformal hyperparameters. Selection must use
  training/calibration/validation information available before each test point.

## Operation ledger

Append one row immediately after each operation. `Output checksums` must refer to
a checksum manifest when more than two files are produced.

| Time (Asia/Shanghai) | Operation | Code/config identity | Server run or path | Output checksums | Outcome and anomalies |
| --- | --- | --- | --- | --- | --- |
| 2026-08-31 | Expansion contract created | start commit `9992a10` | N/A | Pending first baseline commit | Manuscript locked; no expansion run started |

## Decision log

| Time | Decision | Evidence/rationale | Consequence |
| --- | --- | --- | --- |
| 2026-08-31 | Complete and verify the experiment batch before manuscript revision | User instruction; avoids repeated formatting and claim drift | Manuscript remains locked until final evidence audit |
| 2026-08-31 | Treat non-monotonic or null deletion effects as reportable findings | Reviewer concern and anti-HARKing rule | Motivation will be narrowed if the data require it |
| 2026-08-31 | Do not persist SSH credentials | Reproducibility does not require secrets | Authentication is interactive only |

## Completion checklist

- [ ] Current 2026-08-31 source snapshot frozen in Git without build/QA clutter.
- [ ] Published KGCP method and code audited against the current baseline.
- [ ] Added dataset selected from a primary source and prepared reproducibly.
- [ ] Added model implemented, tested, and documented.
- [ ] Filtered negative sampling implemented and unit-tested.
- [ ] Conformal KG baseline(s) implemented and unit-tested.
- [ ] Pilot matrix passes leakage, metric, and runtime checks.
- [ ] Full expansion matrix completed with logs and checkpoints.
- [ ] Multi-answer and deletion-interaction diagnostics verified.
- [ ] All output manifests and checksums verified locally.
- [ ] Manuscript unlocked and revised once from the verified evidence table.
- [ ] Submission PDF, figures, and statements pass final QA.
- [ ] Clean-clone reproduction check passes.
- [ ] Public branch updated and self-describing release/tag created.

