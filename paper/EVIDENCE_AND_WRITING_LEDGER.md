# Evidence and Writing Ledger

Last updated: 2026-08-19

This file records writing decisions and claim-control rules for the RiskCal-TKG
paper. It is intentionally separated from `manuscript.tex` so that experiments
can continue without repeatedly disturbing the submission draft.

The consolidated pre-rewrite synthesis is recorded in
`paper/RESULT_SYNTHESIS_FOR_REVISION.md`. Use that file as the main guide for
the next unified manuscript rewrite.

## Locked Writing Decisions

1. Do not revise the full manuscript after every small experiment.
   Future experiments should first produce scripts, outputs, checksums, and a
   short result note. The manuscript should be updated in a consolidated writing
   pass after a group of experiments is complete.

2. Treat the current manuscript as an evidence-bearing draft, not the final
   submission layout. Large blank regions caused by LaTeX two-column floats are
   acceptable during research drafting, but must be cleaned during the final
   journal-template or Word-template pass.

3. Do not place local absolute paths, server paths, or operational archive lists
   in the professional main text. Examples that should not appear in the final
   main text include `/Users/...`, `/root/...`, and long local command blocks.

4. Keep reproducibility details, commands, checksums, archive names, and run IDs
   in one of these places:
   - appendix, if the target journal welcomes detailed reproducibility sections;
   - supplementary material;
   - repository README;
   - Zenodo/GitHub artifact description after DOI or repository URL is ready.

5. The final paper should use a concise Data and Code Availability statement in
   the main text. Detailed command-level reproduction should not read like a
   local operations log.

6. Every scientific claim in the manuscript must be tied to a specific evidence
   file, table, figure, manifest, or checksum record. Unsupported claims should
   be removed or explicitly framed as future work.

7. Negative results should not be hidden. If an experiment shows a limitation,
   write it as a diagnostic limitation instead of turning it into an unsupported
   method claim.

## Current Claim Register

| Claim type | Current evidence | Writing status |
| --- | --- | --- |
| Static calibration under-covers the later ICEWS14 stream | `paper/data/final_confirmatory/main_calibration_table.csv`, `timestamp_block_bootstrap*.json/csv` | Core claim, keep |
| Rolling prequential calibration improves observed-label reliability | `summary_by_deletion_method.csv`, `timestamp_block_bootstrap_summary.csv` | Core claim, keep |
| The improvement has a large utility cost | `main_calibration_table.csv`, `query_level_paper_table.csv` | Core claim, keep |
| Query-level full-set coverage improves but many sets remain full vocabulary | `query_level_summary.csv`, `query_level_paper_table.csv` | Diagnostic support, keep |
| Relation-side groups improve on average but the weakest groups still under-cover | `relation_slice_summary.csv`, `relation_worst_group_paper_table.csv` | Diagnostic limitation, keep |
| Window/half-life audits show the default pool is short and half-life selection is weakly identified | `window_ablation_summary.csv`, `half_life_selection_summary.csv`, `pool_diagnostics_summary.csv` | Diagnostic support, keep |
| Delayed feedback weakens rolling coverage but does not eliminate the reported positive reduction | `delay_feedback_summary.csv`, `delay_feedback_effects.csv` | Stress-test support, keep |
| Simple set-size caps reduce candidate load but sharply reduce answer rate and unconditional recall | `set_size_utility_paper_table.csv`, `set_size_utility_effects.csv` | Negative diagnostic result, keep as limitation |
| Rank-threshold rolling conformal shortlists reduce candidate-set size with little observed-label coverage change, but full-set query coverage drops slightly | `shortlist_calibration_paper_table.csv`, `shortlist_calibration_effects.csv`, `shortlist_calibration_success_gate.json`, `shortlist_calibration_manifest.json` | Candidate improvement claim; write with the full-set coverage trade-off |
| Score-adaptive APS/RAPS shortlists further reduce mean and median candidate-set size while preserving near-target observed-label coverage, but do not improve tail size over rank-threshold shortlists | `score_adaptive_shortlist_paper_table.csv`, `score_adaptive_shortlist_effects.csv`, `score_adaptive_shortlist_success_gate.json`, `score_adaptive_shortlist_manifest.json` | Candidate improvement claim; write with the full-set coverage and p90-size trade-offs |
| Score-adaptive shortlist gains are stable for 0.88 and 0.90 target coverage, but not supported as a margin-beating size reduction at 0.92 | `score_adaptive_sensitivity_paper_table.csv`, `score_adaptive_sensitivity_effects.csv`, `score_adaptive_sensitivity_success_table.csv`, `score_adaptive_sensitivity_manifest.json` | Robustness and boundary-condition claim; do not write as universal robustness |
| The paper is not a ranking-superiority claim | `history_baseline_summary.csv` | Important limitation, keep |

## Active Experiment Queue

| Experiment | Status | Evidence rule |
| --- | --- | --- |
| Coverage-preserving shortlist calibration by rank-threshold conformal sets | Completed on 2026-08-18 with 20 conditions, 5 seeds, 4 deletion rates, CUDA, 20,000 timestamp-block bootstrap iterations; output archive SHA-256 `758eb5dfae2c8249502b1252056847bc2552645a69a899615b96808a40c14ef9` | At deletion 0.3, `rank_rolling` observed-label coverage is 0.8992 vs 0.8998 for `margin_rolling`; mean set size is 3668.2 vs 4069.4, a reduction of 401.3 candidates with 95% CI [149.0, 651.7]. Full-set coverage decreases from 0.9020 to 0.8881, so do not describe this as a no-cost improvement. |
| Score-adaptive APS/RAPS shortlist calibration | Completed on 2026-08-18 with 20 conditions, 5 seeds, 4 deletion rates, CUDA, 21 APS/RAPS candidates, and 20,000 timestamp-block bootstrap iterations; output archive SHA-256 `03a049993ba5cc3ac71addc212bf981f7eb3416c7621666d40a2be4554c67341` | At deletion 0.3, `adaptive_mass_rolling` observed-label coverage is 0.8987, close to the 0.9 target. Mean set size is 3319.0 vs 4069.4 for `margin_rolling`, a reduction of 750.5 candidates with 95% CI [427.1, 1072.8], and 3319.0 vs 3668.2 for `rank_rolling`, a reduction of 349.2 candidates with 95% CI [246.0, 465.0]. Full-set coverage is 0.8874 vs 0.9020 for `margin_rolling` and 0.8881 for `rank_rolling`; p90 size is 4646.5 vs 3668.2 for `rank_rolling`, so do not claim uniform dominance over rank-threshold shortlists. |
| Score-adaptive shortlist target/tolerance sensitivity | Completed on 2026-08-19 with 20 conditions, 5 seeds, 4 deletion rates, 3 target coverages, 4 selection tolerances, CUDA, 21 APS/RAPS candidates, and 20,000 timestamp-block bootstrap iterations; output archive SHA-256 `4acb18d18065ddb763c688d6b0ddb86fbffea131272b543a325e313610f61692` | At deletion 0.3, `adaptive_mass_rolling` is supported versus `margin_rolling` and has positive mean-size reduction versus `rank_rolling` for all tolerances at target coverage 0.88 and 0.90 (8/12 grid cells). It is not supported versus `margin_rolling` at target coverage 0.92 (0/4 cells; relative mean-size reduction CI lower bounds are negative). Across all tested cells, p90 size is not better than `rank_rolling`, so the final paper should report stable average-size gains for 0.88/0.90 and a high-coverage/tail-size boundary condition. |

## Future Workflow

For each new experiment:

1. Add or update the script and tests.
2. Run or ask the user to run the server command.
3. Download and extract outputs.
4. Record the output files, key numbers, and SHA-256 hashes.
5. Add a short entry to this ledger or a dedicated result note.
6. Only after a batch of experiments is complete, update `manuscript.tex`.
7. Run layout checks only during consolidated writing passes or before delivery.

## Final Submission Cleanup Checklist

Before journal submission:

1. Replace local archive/file wording in the main text with a concise
   Data and Code Availability statement.
2. Move command blocks, run IDs, and checksums to supplementary material or a
   repository README.
3. Rebuild the paper in the selected journal template or Word template.
4. Remove large two-column float gaps where possible.
5. Verify that all claims in the abstract, results, discussion, and conclusion
   appear in the claim register above.
6. Make sure no local absolute paths remain in the final main manuscript.
