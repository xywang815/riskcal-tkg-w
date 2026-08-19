# Manuscript Restructuring Map

Last updated: 2026-08-19

This map guides the consolidated rewrite of `paper/manuscript.tex`. It is not a
paper section.

## Rewrite Priorities

1. Keep the central thesis narrow: reliability-utility trade-off for temporal
   KG prediction sets on ICEWS14.
2. Remove operational text from the main paper: no local paths, terminal
   commands, archive filenames, run IDs, screenshots, or server instructions.
3. Add the completed rank-shortlist and adaptive-mass shortlist experiments to
   the main results.
4. Keep negative results visible: hard caps lose many answerable queries, APS
   alone increases mean set size, adaptive mass does not dominate rank on tail
   size, and target 0.92 is not supported as a margin-beating utility gain.
5. Shorten the abstract and conclusion so that they state the main evidence
   without reading like a JSON summary.
6. Reduce float pressure by shortening or removing oversized appendix tables.

## Section-Level Plan

### Title and Abstract

Use a title that signals calibration and reliability-utility trade-off. The
abstract should report only the main static-vs-rolling evidence and the main
shortlist trade-off. It should not list every diagnostic.

### Introduction

Keep the motivation and four contributions, but change the final contribution
from "adaptive diagnostic" to "shortlist and robustness diagnostics." Do not
promise broader datasets, backbones, or formal drift guarantees.

### Problem and Method

Keep the closed-vocabulary query formulation and prequential information
boundary. Add a concise description of rank and adaptive-mass shortlisting after
the margin threshold. Clarify that APS/RAPS and adaptive mass are empirical
diagnostics, not conformal guarantees under arbitrary drift.

### Experimental Design

Keep ICEWS14-only scope. Replace references to "server run" with "confirmatory
run" or "archived run." Keep hardware/runtime details short and place artifact
hash details outside the main text.

### Results

Recommended order:

1. Ranking sanity check and scope boundary.
2. Static undercoverage and rolling reliability repair.
3. Query-level reliability and large-set utility cost.
4. Hard-cap negative diagnostic.
5. Rank and adaptive-mass shortlist calibration.
6. Robustness and boundary conditions: sensitivity grid, relation slices,
   window/half-life, delayed feedback.

### Discussion

Emphasize that the contribution is operational calibration evidence, not a new
TKG ranking model. Discuss the three utility boundaries: full vocabulary sets,
full-set coverage loss under shortlists, and tail-size non-dominance.

### Reproducibility

Replace the command/archive list with a concise Data and Code Availability
statement. Keep exact commands and archive names in repository README or
supplement, not in the main manuscript.

### Appendix

Shorten the appendix. Keep only compact tables that support limitations:
history baselines and block-length sensitivity. Remove the oversized diagnostic
run summary table from the manuscript body unless a journal explicitly asks for
it in supplementary material.

## Required Main-Text Evidence

- Static vs rolling at deletion rate 0.3:
  - Static coverage 0.8235, rolling coverage 0.8998.
  - Undercoverage reduction 0.0723, 95% CI [0.0535, 0.0938].
- Query-level rolling at deletion rate 0.3:
  - Full-set coverage 0.9020, partial recall 0.9174.
  - Full-vocabulary set rate 0.3647.
- Rank shortlist at deletion rate 0.3:
  - Mean size 3668.2 vs margin rolling 4069.4.
  - Full-set coverage 0.8881 vs 0.9020.
- Adaptive mass shortlist at deletion rate 0.3:
  - Mean size 3319.0, median size 2924.6.
  - Observed-label coverage 0.8987.
  - Full-set coverage 0.8874.
  - p90 size 4646.5, worse than rank rolling p90 3668.2.
- Target sensitivity:
  - Supported at 0.88 and 0.90 targets.
  - Not supported as margin-beating at 0.92.
- Ranking scope:
  - Prequential Repeat MRR 0.3631 exceeds frozen scorer MRR 0.3105.

## Claims To Keep Out Of Main Text

- SOTA ranking claims.
- Universal adaptive superiority claims.
- Arbitrary temporal drift guarantees.
- Local machine/server paths.
- Shell commands.
- Archive filenames and SHA strings in prose.
- Screenshots or screenshot-derived evidence.

