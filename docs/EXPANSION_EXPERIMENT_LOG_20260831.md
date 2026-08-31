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

The cleaned 2026-08-31 manuscript/submission snapshot was frozen locally in Git
as commit `01a71d4`. The push is pending because the first network attempt timed
out; this does not change the experiment identity.

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

The added levels below were fixed after primary-source and implementation audit,
before inspecting any expansion outcomes.

| Axis | Starting level | Required added level | Status |
| --- | --- | --- | --- |
| Dataset | ICEWS14 | ICEWS05-15 from the official TKBC data bundle | Prepared and pilot-validated |
| Scorer | Temporal DistMult-style | Continuous-time ComplEx scorer under the matched training protocol | Implemented and pilot-validated |
| Negative sampling | Uniform object corruption | Training-positive filtered | Implemented and unit-tested |
| Conformal baseline | Current static/rolling margin methods | Static KGCP NegScore, Minmax, and Softmax | Implemented and unit-tested |
| Deletion | 0.0, 0.1, 0.2, 0.3 | Same, with interaction analysis | Frozen in E1-E6; formal analysis pending |
| Seeds | 17, 29, 43, 59, 71 | Same unless predeclared otherwise | Frozen in E1-E6; formal runs pending |
| Multi-answer | Existing diagnostics | Cross-dataset/model stratification | Query-level multiplicity fields implemented and unit-tested; formal analysis pending |

### Frozen expansion runs

All paper-facing runs use seeds `{17, 29, 43, 59, 71}`, target coverage
`0.90`, the same chronological 60/20/20 split, and the four-role calibration
partition. DistMult uses dimension 256; the complex scorer uses dimension 128
so that its two real-valued embedding halves have a comparable entity/relation
parameter budget. Pilot dimensions/epochs are diagnostic only and will not enter
paper tables.

| Run | Dataset | Scorer | Sampling | Deletion rates | Purpose |
| --- | --- | --- | --- | --- | --- |
| E1 | ICEWS14 | temporal DistMult | filtered | 0, .1, .2, .3 | Primary filtered comparison and KGCP baselines |
| E2 | ICEWS05-15 | temporal DistMult | filtered | 0, .1, .2, .3 | Dataset generalization |
| E3 | ICEWS14 | continuous-time ComplEx | filtered | 0, .1, .2, .3 | Scorer generalization |
| E4 | ICEWS05-15 | continuous-time ComplEx | filtered | 0, .3 | Joint dataset/scorer boundary check |
| E5 | ICEWS05-15 | temporal DistMult | uniform | 0, .3 | Negative-sampling sensitivity on added dataset |
| E6 | ICEWS14 | continuous-time ComplEx | uniform | 0, .3 | Negative-sampling sensitivity on added scorer |

The frozen 2026-08-15 ICEWS14 DistMult uniform run remains an additional
historical comparison. It will not be silently merged with the new runs if code
or output schemas prevent a valid paired analysis.

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
| 2026-08-31 | Froze cleaned 2026-08-31 manuscript and experiment baseline | commit `01a71d4` | N/A | Git object identity `01a71d4`; generated export/QA directories ignored | Local commit succeeded; push to GitHub timed out on port 443 and remains pending |
| 2026-08-31 | Verified server runtime and preserved previous run | local commit `01a71d4`; server code is a non-Git tar copy | `/root/riskcal_tkg_mvp`; prior run `results/final_confirmatory/20260815T110014908249Z-eb736dbf6658` | Runtime inspection only | RTX 4090 24 GB, PyTorch 2.5.1+cu124, CUDA available; no competing compute job found; old run left unchanged |
| 2026-08-31 | Downloaded and verified official TKBC data archive | official `facebookresearch/tkbc` download route | `/root/autodl-tmp/riskcal_sources/data.tar.gz` | SHA-256 `2a993856622981535067a5ba54a5c649e7b50bf6ba0cb2197c17b2e9c069d25e` | Archive extracted without error; contains ICEWS14, ICEWS05-15, YAGO15K, and Wikidata source files |
| 2026-08-31 | Audited added-dataset statistics | unmodified official archive above | `/root/autodl-tmp/riskcal_sources/tkbc_data/src_data/ICEWS05-15` | Covered by archive checksum | ICEWS05-15 has 461,329 rows, 10,488 entities, 251 relations, 4,017 daily timestamps from 2005-01-01 to 2015-12-31; source license file points to Harvard Dataverse DOI `10.7910/DVN/28075` |
| 2026-08-31 | Audited published KGCP definitions against current implementation | KGCP primary paper arXiv `2408.08248`; local `calibration.py` and `experiment.py` | N/A | Source and code audit only | Current `static` is max-score minus true-score margin calibration, not published KGCP. Published static NegScore, Minmax, and Softmax constructions will be added; old method will be reported as `static-margin` |
| 2026-08-31 | Implemented the predeclared expansion code and frozen configurations for server validation | uncommitted child of `01a71d4`; bundle SHA-256 `39f2b66ddafad2209698f5eb87fc8afb783179b97d60d3fea233477b7ef1a009` | `/root/autodl-tmp/riskcal_expansion_staging` | Bundle checksum verified byte-for-byte on server | Added ICEWS05-15 preparation, continuous-time ComplEx, filtered sampling, three static KGCP score transforms, explicit baseline names, and E1-E6 configs; no formal run started |
| 2026-08-31 | Ran the first focused server test gate | same uncommitted bundle above | `/root/autodl-tmp/riskcal_expansion_staging` | Pytest terminal record | 42 tests passed and 13 failed. Twelve failures traced to `TemporalDistMult.score_all_objects` being placed in the wrong class scope; one packaging-only failure traced to README omission. Formal experiments remained blocked pending a corrected bundle. |
| 2026-08-31 | Corrected DistMult method scope locally | uncommitted child of `01a71d4` | Local source tree | `py_compile` and `git diff --check` passed | No protocol or metric definition changed; corrected bundle and full server retest pending |
| 2026-09-01 | Re-ran focused and full server test gates on corrected source | corrected bundle SHA-256 `b96d42ad11bcd40858fd5c2119e1ea9ca7cc7fde6c0c26885d492424c4ba0d18` | `/root/autodl-tmp/riskcal_expansion_staging_v3` | Bundle checksum verified on server; pytest records in terminal | Focused gate: 55 passed. Full suite: 111 passed with one existing trusted-checkpoint `torch.load` future warning. No test failed; formal runs remain unstarted pending data and pilot validation. |
| 2026-09-01 | Committed the server-tested expansion implementation and frozen E1-E6 configurations | commit `cec7816d619ec7876bccad777740fe41fd0ead0c` | Local source tree; server staging content is identical to the commit | Git object identity `cec7816` | Commit becomes the source identity for expansion runs; manuscript remains locked |
| 2026-09-01 | Prepared and reloaded official ICEWS14 and ICEWS05-15 data | commit `cec7816`; source archive SHA-256 `2a993856622981535067a5ba54a5c649e7b50bf6ba0cb2197c17b2e9c069d25e` | `/root/autodl-tmp/riskcal_expansion_staging_v3/data/raw/{icews14,icews05_15}` | Per-file checksums recorded in each `SOURCE.json` | ICEWS14: 90,730 facts, 7,128 entities, 230 relations, 365 timestamps. ICEWS05-15: 461,329 facts, 10,488 entities, 251 relations, 4,017 timestamps. Strict split-boundary checks passed with no overlap. |
| 2026-09-01 | Completed ICEWS14 continuous-time ComplEx filtered-sampling pilot | commit `cec7816`; pilot config `pilot_icews14_tcomplex_filtered.yaml` | `/root/autodl-tmp/riskcal_expansion_staging_v3/results/expansion_pilots/icews14_tcomplex_filtered/20260831T162503080495Z-3331ac95bcd1` | Condition markers verify checkpoint, deletion mask, resources, windows, and query artifacts | Completed 2 conditions in 250.93 s; 0 prequential leakage violations; 8 methods present; nondegenerate score spread; 10 epochs each. Single-seed diagnostic only: static-margin and KGCP Softmax under-covered while KGCP NegScore/Minmax were closer to 0.90. No manuscript claim made. |
| 2026-09-01 | Completed ICEWS05-15 temporal DistMult filtered-sampling pilot | commit `cec7816`; pilot config `pilot_icews05_15_distmult_filtered.yaml` | `/root/autodl-tmp/riskcal_expansion_staging_v3/results/expansion_pilots/icews05_15_distmult_filtered/20260831T163825064706Z-ca160db592a2` | Condition markers verify checkpoint, deletion mask, resources, windows, and query artifacts | Completed 2 conditions in 1,426.42 s; 0 prequential leakage violations; 8 methods present; nondegenerate score spread; 5 epochs each. Single-seed diagnostic: rolling/adaptive mean coverage was about 0.90, whereas all static variants were below target. No manuscript claim made. |
| 2026-09-01 | Tested GPU all-entity scoring as a possible runtime optimization | temporary uncommitted child of `cec7816` | `/root/autodl-tmp/riskcal_expansion_staging_v4/results/device_validation/icews14_tcomplex_filtered/20260831T171845638680Z-3331ac95bcd1` | Full suite 111 passed before pilot; matched per-window outputs compared with the `cec7816` CPU-scoring pilot | Coverage was identical and maximum MRR difference was `2.24e-9`, but duration increased from 250.93 s to 255.14 s and set-size boundary values differed slightly. The optimization was rejected and local code restored exactly to `cec7816`; formal runs retain CPU evaluation for reproducibility. |
| 2026-09-01 | Added per-query answer multiplicity and a resumable E1-E6 matrix controller | uncommitted child of `cec7816`; transfer archive SHA-256 `085b147bf3caec0e582224a0dfb9f77b577101bd6347efb7587223eb9b54d453` | `/root/autodl-tmp/riskcal_expansion_staging_v5` | Transfer checksum matched on server; full suite `112 passed` with one pre-existing trusted-checkpoint warning | Each query row now records `answer_count` and `is_multi_answer`. The controller writes progress atomically and records config/run-manifest hashes; no formal E1-E6 run was started from the uncommitted archive. |
| 2026-09-01 | Validated matrix completion detection and duplicate-run avoidance | same uncommitted staging source above; `configs/smoke.yaml` | `/root/autodl-tmp/riskcal_expansion_staging_v5/results/matrix_smoke/smoke/20260831T180018549279Z-4c824f5e96dd` | Config SHA-256 `b1c237280d85cce5fe6561a5873e4ad3212108128d689b4a9df6832c2653a0fd`; run-manifest SHA-256 `2c3df427bcaef4117f6cd593db85b9f4cd86d7ad9ae723946e2c60633894d1d8` | First invocation completed the smoke run; the second verified and skipped the same run. Query artifact contained both multiplicity columns. Formal runs remain blocked until this source is committed and redeployed from a Git bundle. |

| 2026-09-01 | Committed and transferred the formal expansion source | commit `52beca5c0ded61dc071c410ccf9cbd05691ac92e` | Git bundle `/root/autodl-tmp/riskcal_tkg_expansion_52beca5.bundle` | Bundle SHA-256 `d33ec6a3ef3dce6ff17531bd9f663d4ae47468f2b438dfa4d2970fe342fe2a33`; `git bundle verify` passed locally and server checksum matched | Bundle contains complete Git history and the answer-multiplicity/matrix-controller changes; no credential was persisted. |
| 2026-09-01 | Established and verified the clean formal worktree | commit `52beca5`; official source archive SHA-256 `2a993856622981535067a5ba54a5c649e7b50bf6ba0cb2197c17b2e9c069d25e` | `/root/autodl-tmp/riskcal_expansion_formal_52beca5` | `git status --short` empty; full suite `112 passed` with one pre-existing trusted-checkpoint warning; prepared file hashes match prior preparation | ICEWS14 and ICEWS05-15 were regenerated from the verified archive inside the formal worktree. GPU was idle before launch. |
| 2026-09-01 | Launched the frozen formal E1-E6 matrix | commit `52beca5`; default config list in `scripts/run_expansion_matrix.py` | worktree above; PID `11314`; log `expansion_matrix_52beca5.log`; progress `results/expansion_formal/matrix_progress.json` | Progress records Git commit `52beca5c0ded61dc071c410ccf9cbd05691ac92e`; first config SHA-256 `b8159bc51883a7ddc1d0ef1d362605d278c2966ec9c73e746780273b912c8f62` | Background process started with E1 `icews14_distmult_filtered`; manuscript remains locked. Completion, resource use, and anomalies will be monitored from the process, log, condition markers, and atomic progress record. |

| 2026-09-01 | Added and independently validated the post-matrix expansion audit exporter | uncommitted child of `52beca5`; validation archive SHA-256 `410f15fc226b822285db8f3b9c599e12681addc6a0f29cbab802428ac64a5f04` | validation only in `/root/autodl-tmp/riskcal_expansion_staging_v5`; formal worktree unchanged | Focused tests `3 passed`; full suite `115 passed` with one pre-existing trusted-checkpoint warning | Exporter refuses incomplete/non-Git matrices and inconsistent answer multiplicities, supports relocated results, and writes condition, deletion, method, sampling, multi-answer, manifest, and checksum artifacts. It will run only after E1-E6 complete. |
| 2026-09-01 | Initial formal-matrix health check | commit `52beca5`; no code change | formal PID `11314`; E1 run `results/expansion_formal/icews14_distmult_filtered/20260831T183333122638Z-7584b3c257f7` | Seven complete condition markers after 42 minutes | Process alive with sustained CPU/GPU use and low memory; no retry, failure, or stalled condition observed. |
| 2026-09-01 | Added and server-validated the predeclared expansion timestamp-block bootstrap exporter | uncommitted child of `356dee7`; transfer archive SHA-256 `8cd27b1db2780fcab2c8bc42e0db30beb637cbd9e31af1a507623ad86d1b8b8e` | isolated validation worktree `/root/autodl-tmp/riskcal_expansion_staging_v5`; formal worktree unchanged | Focused tests `4 passed`; full suite `119 passed` with the same pre-existing trusted-checkpoint warning | Exporter requires a complete matrix, validates provenance and paired support, and covers method, deletion, negative-sampling, and multi-answer contrasts with 20,000-iteration circular timestamp-block bootstrap at block lengths 3, 7, 14, and 21. It writes CSV, manifest, and SHA-256 records only after E1-E6 finish. |
| 2026-09-01 | Formal E1 health check after bootstrap-tool validation | commit `52beca5`; no code change | formal PID `11314`; same E1 run | Fourteen of 20 E1 condition markers complete after 1 h 21 min | Process remained alive at about 644% CPU; GPU reported 48% utilization, 1,681 MiB memory, 42 C, and 178 W. No failure, retry, or stale marker observed. |

## Decision log

| Time | Decision | Evidence/rationale | Consequence |
| --- | --- | --- | --- |
| 2026-08-31 | Complete and verify the experiment batch before manuscript revision | User instruction; avoids repeated formatting and claim drift | Manuscript remains locked until final evidence audit |
| 2026-08-31 | Treat non-monotonic or null deletion effects as reportable findings | Reviewer concern and anti-HARKing rule | Motivation will be narrowed if the data require it |
| 2026-08-31 | Do not persist SSH credentials | Reproducibility does not require secrets | Authentication is interactive only |
| 2026-08-31 | Use ICEWS05-15 as the added dataset | Same official TKBC bundle as ICEWS14; 5.08x more facts and 11-year horizon; direct source and checksum available | Cross-dataset evidence will use a matched chronological protocol rather than paper-reported official random-split scores |
| 2026-08-31 | Add a continuous-time ComplEx scorer | TComplEx provides a materially different asymmetric complex-valued scoring family; a continuous time map is required by the prequential future-timestamp protocol | The scorer will be described as a matched-protocol continuous-time ComplEx implementation, not as an exact reproduction of the official TComplEx optimizer |
| 2026-08-31 | Filter only training positives during negative sampling | Predeclared sensitivity scope; prevents false negatives without exposing calibration/test labels | Uniform and filtered training runs share evaluation filtering and all other settings |
| 2026-08-31 | Keep the old margin methods but stop calling them KGCP | Method-identity audit | Expansion outputs use explicit `static_margin` and `rolling_margin`; KGCP names are reserved for the published score transforms |
| 2026-09-01 | Record answer multiplicity in the primary query artifact rather than infer it after aggregation | Reviewer concern about multi-answer degradation; the filtered truth map is available at evaluation time | Every formal method/query row can be stratified by single- versus multi-answer status without reconstructing labels from a separate data split |
| 2026-09-01 | Launch formal E1-E6 only from a clean clone of a committed Git bundle | The uncommitted transfer archive cannot supply a Git commit in `matrix_progress.json` | Formal matrix provenance must include a non-null commit and clean worktree before launch |

## Completion checklist

- [x] Current 2026-08-31 source snapshot frozen in Git without build/QA clutter.
- [x] Published KGCP method and code audited against the current baseline.
- [x] Added dataset selected from a primary source; reproducible preparation script pending.
- [x] Added model implemented and unit-tested; experiment documentation awaits final evidence.
- [x] Filtered negative sampling implemented and unit-tested.
- [x] Conformal KG baseline(s) implemented and unit-tested.
- [x] Pilot matrix passes leakage, metric, and runtime checks.
- [ ] Full expansion matrix completed with logs and checkpoints.
- [ ] Multi-answer and deletion-interaction diagnostics verified.
- [ ] All output manifests and checksums verified locally.
- [ ] Manuscript unlocked and revised once from the verified evidence table.
- [ ] Submission PDF, figures, and statements pass final QA.
- [ ] Clean-clone reproduction check passes.
- [ ] Public branch updated and self-describing release/tag created.
