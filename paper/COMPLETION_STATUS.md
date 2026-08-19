# RiskCal-TKG Paper Completion and Evidence Status

**Supersession note (2026-08-16).** This register was written before the
five-seed ICEWS14 server run completed. The current manuscript now uses the
formal confirmatory artifacts from
`results/final_confirmatory/20260815T110014908249Z-eb736dbf6658`, where the
machine-readable gate reports `supported=true`. Treat the older statements below
about missing confirmatory evidence and MVP-only methodology as historical audit
notes, not as the current paper state.

Last boundary review: 2026-08-12.

## Frozen method identity

The implemented MVP is **globally selected fixed half-life weighted empirical
calibration**. It evaluates candidate half-lives on a pre-test calibration-time
split, chooses one half-life once, and uses that fixed value for every test
timestamp batch. The implementation does **not** compute the manuscript's drift
feature vector and does **not** choose a batch-specific half-life.

The batch-adaptive drift-feature selector remains a paper-facing method
specification. Until its implementation and leakage tests exist, it cannot be
named as the method responsible for any implemented or local empirical value.

## Evidence classes

| Class | Permitted content | Claim boundary | Current status |
|---|---|---|---|
| Unit/smoke verification | Deterministic tests of parsing, temporal splitting, calibration primitives, batching, artifact writing, and toy end-to-end execution | Verifies software behavior only; it is not dataset-level empirical evidence | Available as code/tests; results must be reported only with the exact verification command and date |
| Full-data descriptive audit | Dataset counts, temporal ranges, fingerprints, source hashes, split membership, deletion-mask summaries, and leakage checks on the complete local data | Descriptive/provenance evidence only; it supports no ranking, calibration, or comparative-performance claim | Complete for local ICEWS14 and reconciled to the copied immutable dataset manifest; server-extension datasets remain missing |
| Local CPU diagnostic | Reduced-budget runs on local CPU, including the existing one-seed frequency-biased subset engineering pilot and the three-seed full-ICEWS14 diagnostic | Non-confirmatory engineering evidence. It may diagnose pipeline behavior but cannot fill confirmatory tables, the abstract result sentence, the conclusion result sentence, or a preregistered gate | Both diagnostics are incorporated only in explicitly labeled non-confirmatory subsections; the full-data run completed six conditions and its hashes/aggregates are under `paper/data` |
| Future confirmatory server evidence | Frozen paper-facing implementation, required backbones and baselines, additional dataset, five seeds, immutable artifacts, and preregistered inference | The only class permitted to populate confirmatory result tables and paper-level empirical claims | Missing |

The four classes must remain separate in prose, tables, figures, artifact paths,
and summaries. In particular, the subset pilot and the completed full-data local CPU
diagnostic must not be pooled with one another or with server runs.

The older pilot used one seed and a frequency-biased 109-entity subset; the
2026-08-12 diagnostic used three seeds, all 7,128 ICEWS14 entities, and all
90,730 facts. Their candidate spaces, sampling designs, and training conditions
are different, so no pooled estimate or improvement comparison is admissible.

The versioned exporter `scripts/export_paper_diagnostic.py` is the sole
paper-facing transformation for the full-data local diagnostic. The diagnostic
manifest records its exact invocation and SHA-256. Source metrics keep
zero-based internal timestamp IDs; copied manifests name those fields with an
`_id` suffix, while tables and figures use the original ICEWS14 labels 293--365.

## Components blocked from local completion

The table accounts for every currently incomplete manuscript component whose
completion depends on work or decisions outside the local evidence boundary.
Components that can be completed locally (for example, descriptive manifests,
theory scoped to the implemented empirical threshold, reference maintenance,
and document checks) are not listed as external blockers.

| Manuscript component | Missing dependency | What is required before completion |
|---|---|---|
| Paper-facing drift feature vector, per-half-life predictive models, batch-adaptive selector, deterministic tie rule, and selector leakage tests | Method implementation | Implement and validate the specified batch-adaptive method. Until then, all implemented-method prose must say globally selected fixed half-life weighting. |
| Four disjoint pre-test roles and a final calibration pool with no half-life-selection reuse | Method implementation | Replace the MVP's exploratory calibration reuse and add protocol-to-code tests. |
| Feedback-delay sensitivities and additional controlled history-deletion mechanisms, including any source/relation-family mapping | Method implementation; additional dataset | Implement the mechanisms and freeze dataset-specific metadata and mappings. |
| RE-GCN primary backbone and optional TiRGN neural extension | Method implementation; server compute | Implement or integrate the backbones, validate the shared scoring interface, and train them under frozen budgets. |
| Frequency, Repeat/Relaxed-Repeat, static KGCP, predicate-conditional KGCP, ACI, and the selected beyond-exchangeability/non-exchangeable risk-control comparison | Baseline code; server compute | Integrate audited baseline implementations and run them on shared immutable score tensors and configurations. |
| ICEWS05-15/GDELT extension and at least one structurally distinct non-event temporal graph, including dataset-table cells and source metadata | Additional dataset; server compute | Select, acquire, normalize, hash, and run the frozen protocol on the additional data. |
| Confirmatory inference configuration tied to the final implementation, including the fifth seed, block-length rule, resampling settings, multiplicity correction, and machine-readable gates | Method implementation; five-seed result | Freeze the protocol before inspecting complete confirmatory outputs, then validate the summarizer against all five seeds. |
| Server hardware/software manifest, exact training budgets, measured runtime/memory/scaling results, and any exact-versus-ANN comparison | Server compute | Record the execution environment and produce measurements from immutable server runs. |
| Dataset ranking table and its interpretation against Frequency/Repeat | Baseline code; server compute; five-seed result | Complete all five seeds for the neural and non-neural methods and calculate the preregistered intervals. |
| Coverage-over-time figure, primary calibration table, H1/H2/primary-gate prose, and confidence bounds | Method implementation; baseline code; server compute; five-seed result | Generate immutable per-window and per-seed confirmatory artifacts with the frozen summarizer. |
| Reliability--utility curves, group-wise reliability figure, success/failure cases, ablation table, selector-frequency/feature analyses, score and delay sensitivities, missingness comparisons, efficiency figures, and supplemental per-seed tables | Method implementation; baseline code; server compute; additional dataset; five-seed result | Run the relevant frozen confirmatory and secondary conditions; local pilots cannot fill these components. |
| Result-dependent Discussion branch, empirical contribution sentence, Abstract result sentence, and Conclusion result sentence | Five-seed result | Select wording only after the immutable confirmatory gate and trade-off outputs exist. No local diagnostic value is admissible here. |
| Author affiliation, postal address, ORCID, corresponding email, acknowledgments, funding, and conflict-of-interest declarations | Author metadata | Obtain and approve final author-provided metadata. |
| Official journal template, final front matter, venue-specific declarations, hardware/carbon wording, artifact/archival requirements, and submission formatting | Venue decision | Select the target venue and apply its current requirements. |

## Placement rule for local values

The existing engineering-pilot values may appear only in the subsection
`Non-Confirmatory Engineering Pilot, Not Scientific Evidence` and in this
register as provenance
when needed. They must not appear in the abstract, conclusion, confirmatory
result tables, confirmatory figures, hypothesis gates, or paper-level empirical
claims. Future local CPU diagnostic values inherit the same restriction.

## 2026 literature-delta evidence register

The novelty statement was re-audited on 2026-08-12. The following records are
direct comparisons and must remain visible in any submission version.

| Work | Status and identifier | Task boundary | Consequence for RiskCal-TKG |
|---|---|---|---|
| Hu et al., *Conformal Event Prediction with Temporal Knowledge Graph* (CFEP) | Peer-reviewed, Findings of ACL 2026, July 2026, DOI `10.18653/v1/2026.findings-acl.258`; [ACL Anthology](https://aclanthology.org/2026.findings-acl.258/) | Predicts sets of co-occurring event/relation types from TKG history; it does not use the present paper's all-entity subject/object link-forecasting interface | Closest peer-reviewed temporal-KG comparison. The paper may not claim that conformal TKG prediction is an empty field. A server study needs a CFEP-derived entity-forecasting comparison or a documented incompatibility analysis. |
| Wang et al., *Non-exchangeable Conformal Prediction for Temporal Graph Neural Networks* (NCPNET) | Peer-reviewed, KDD 2025; arXiv submitted 2 July 2025, DOI `10.1145/3711896.3737064`, arXiv `2507.02151`; [record](https://arxiv.org/abs/2507.02151) | Temporal-graph/GNN tasks and node-oriented benchmarks rather than TKG all-entity completion | Mandatory temporal conformal design comparison; its guarantee and task semantics cannot be transferred to the MVP. |
| Lam et al., *Non-Exchangeable Conformal Path Reasoning over Temporal Knowledge Graphs* | Non-peer-reviewed concurrent SSRN preprint, posted 3 July 2026, DOI `10.2139/ssrn.7050285`; [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7050285) | Multi-hop path reasoning and relation-conditional coverage on ICEWS14 rather than single-hop entity forecasting | Concurrent evidence only. Its negative fixed-recency result strengthens the need for conditional-coverage and volatility-aware analysis but is not treated as established peer-reviewed evidence. |

The remaining defensible novelty target is the combination of all-entity
subject/object forecasting, controlled training-history deletion, separation of
scorer degradation from calibration drift, timestamp-batched prequential
updates, and coverage--set-size--ranking auditing. That target is provisional
until the paper-facing adaptive selector and confirmatory comparisons are
implemented.
