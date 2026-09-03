# Follow-up Calibration Evidence

This directory contains the fixed component-ablation and query-objective study
used by the revised manuscript.

`analysis/followup_by_timestamp.csv` is the authoritative paired timestamp
table. `analysis/followup_by_seed.csv` is its count-weighted seed summary, and
`analysis/query_objective_contrasts_by_seed.csv` isolates query-max minus
label-level calibration. The analysis manifest pins code commit `7f19fe7` and
the source checkpoint hashes.

`bootstrap/followup_bootstrap_summary.csv` reports 20,000-replicate circular
moving-block bootstrap intervals at block lengths 3, 7, 14, and 21. Its
manifest pins code commit `d6c9602`. Block length 7 is the primary analysis;
the remaining lengths are sensitivity checks.

Each subdirectory has an independent `SHA256SUMS.txt`. The original trained
checkpoints are not duplicated here; their hashes and source run locations are
recorded in the analysis manifest.
