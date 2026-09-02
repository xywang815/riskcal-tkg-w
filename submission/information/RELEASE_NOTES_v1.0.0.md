# RiskCal-TKG v1.0.0 MDPI Information Submission Snapshot

This release binds the final submission manuscript to the completed E1--E6 expansion evidence.

## Included Evidence

- 90/90 completed seed--deletion conditions.
- ICEWS14 and ICEWS05-15.
- Temporal DistMult and a matched-protocol continuous-time complex scorer.
- Filtered and uniform training-negative sensitivity pairs.
- Static margin, rolling margin, and three published KGCP score transformations.
- 20,000-replicate equal-seed circular timestamp-block bootstrap with block length 7 and sensitivity lengths 3, 14, and 21.
- Query-level multi-answer diagnostics.

## Interpretation Boundary

The evidence supports a prequential reliability--utility diagnostic. It does not establish universal superiority, an arbitrary-drift distribution-free guarantee, an exact reproduction of the KGCP or TComplEx authors' complete pipelines, or a deletion-driven mechanism for static miscalibration.

## Artifact Integrity

Paper-facing CSV files and figures are checksum-bound by their manifests. A fresh server-side clone of candidate `ff5b959` passed all 124 project tests. The post-manuscript audit then verified all 90 condition markers, tracked publication files, matrix ancestry, and analysis, bootstrap, and figure checksum chains. The release metadata commit adds that audit report without changing the manuscript or empirical outputs.
