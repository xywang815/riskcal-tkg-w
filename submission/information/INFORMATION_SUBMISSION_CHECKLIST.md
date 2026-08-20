# MDPI Information Submission Checklist

Target journal: MDPI `Information`.

## Manuscript Status

- Title narrowed to include `An ICEWS14 Diagnostic Study`.
- The manuscript uses the MDPI `Information` LaTeX template.
- The method name in the manuscript is `validation-selected RAPS rolling`.
- RAPS selection is described as calibration-only validation, not test-period online selection.
- The APS/RAPS candidate grid is stated:
  - APS baseline candidate;
  - RAPS `k` grid: `{50, 100, 250, 500, 1000}`;
  - RAPS `lambda` grid: `{1e-4, 5e-4, 1e-3, 2e-3}`;
  - softmax temperature `T=1`;
  - selection tolerance `0.02`;
  - final selected candidate in the main 0.90 setting: `k=50`, `lambda=1e-4`.
- The abstract is in the requested 230--245 word range.
- A separate `Limitations and Practical Implications` section is included.
- The back matter includes Author Contributions, Funding, IRB statement, Informed Consent statement, Data Availability Statement, and Conflicts of Interest.
- The manuscript is framed as a diagnostic reliability-utility study under strict prequential ordering.
- The manuscript does not claim:
  - a new distribution-free coverage theorem under temporal drift;
  - state-of-the-art temporal KG ranking;
  - guaranteed conditional coverage;
  - conclusions beyond the stated ICEWS14 protocol.

## Repository Status

- Public GitHub repository: `https://github.com/xywang815/riskcal-tkg-w`.
- License: MIT License.
- Included artifacts: code, configurations, tests, MDPI manuscript source, derived paper-facing result tables, and figures.
- Excluded artifacts: raw ICEWS14 files, checkpoints, logs, build products, transfer archives, and server access details.
- Stable Git tag created: `v1.0-information-submission`.
- Optional but stronger: create a GitHub Release from this tag, archive the release on Zenodo, and add the DOI to the Data Availability Statement.

## Files to Upload in the MDPI System

- Manuscript PDF: `riskcal_tkg_mdpi_information_submission_20260820.pdf`.
- LaTeX source package: `riskcal_tkg_mdpi_information_source_20260820.zip`.
- Cover letter: use `COVER_LETTER_DRAFT.md` as the text source.
- Repository URL for data/code availability: `https://github.com/xywang815/riskcal-tkg-w`.

## Author Confirmation Before Submission

- Confirm that the affiliation, email address, and funding grant number are correct.
- Confirm whether the manuscript is submitted as a regular Article or to a special issue.
- Confirm whether a Zenodo DOI will be created before first submission or only after review.
- Confirm APC/payment details in the MDPI submission system.
