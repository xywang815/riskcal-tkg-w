# Expansion Evidence Bundle

This directory contains the paper-facing outputs from the completed E1--E6
expansion matrix. The raw checkpoints and per-query Parquet files are not stored
in Git because of their size; their hashes and run identities are retained in
the manifests.

- `analysis/`: seed-level and aggregate CSV outputs plus their manifest.
- `bootstrap/`: 20,000-iteration timestamp-block-bootstrap outputs.
- `provenance/`: frozen formal/pilot configurations, completion ledger, external
  run audits, and checksums.
- `../../figures/expansion/`: PDF/PNG figures generated from the verified CSVs.

Run `shasum -a 256 -c SHA256SUMS.txt` inside each subdirectory to verify the
tracked files. The manuscript-facing interpretation and claim boundaries are in
`docs/EXPANSION_RESULT_SUMMARY_20260902.md` and
`docs/EXPANSION_CLAIM_LEDGER_20260901.md`.

Method names are intentionally precise. `static_margin` is not KGCP;
`kgcp_negscore_static`, `kgcp_minmax_static`, and `kgcp_softmax_static` use the
published KGCP score transformations under the matched static calibration
protocol. `continuous_tcomplex` is a matched-protocol continuous-time
complex-valued scorer, not an exact reproduction of the official TComplEx
optimizer.
