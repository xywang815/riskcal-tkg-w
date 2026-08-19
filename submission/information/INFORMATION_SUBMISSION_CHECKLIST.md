# MDPI Information Submission Checklist

Target journal: MDPI `Information`.

## Manuscript Changes Required Before Submission

- Change title to a narrower title with `An ICEWS14 Diagnostic Study`.
- Replace `adaptive mass rolling` with `validation-selected RAPS rolling`.
- State clearly that RAPS selection uses calibration-only validation batches, not test-period online selection.
- Specify the RAPS/APS candidate grid:
  - APS baseline candidate;
  - RAPS `k` grid: `{50, 100, 250, 500, 1000}`;
  - RAPS `lambda` grid: `{1e-4, 5e-4, 1e-3, 2e-3}`;
  - softmax temperature `T=1`;
  - selection tolerance `0.02`;
  - final selected candidate in the main 0.90 setting: `k=50`, `lambda=1e-4`.
- Remove any remaining `Route A`, `route-A`, or internal project terminology from the manuscript and supplement.
- Convert to the MDPI LaTeX template for `Information`.
- Add MDPI-required back matter:
  - Author Contributions;
  - Funding;
  - Institutional Review Board Statement, if required by template;
  - Informed Consent Statement, if required by template;
  - Data Availability Statement;
  - Conflicts of Interest;
  - Acknowledgments, if needed.
- Add a separate `Limitations and Practical Implications` section.
- Compress abstract to approximately 230--245 words.
- Ensure the manuscript never claims:
  - a new distribution-free coverage theorem;
  - state-of-the-art temporal KG ranking;
  - guaranteed conditional coverage;
  - performance beyond ICEWS14 without supporting experiments.

## Repository Requirements

- Public GitHub repository created: `https://github.com/xywang815/riskcal-tkg-w`.
- Selected code license: MIT License.
- Add a stable release before submission.
- Preferably archive the release on Zenodo and cite the DOI.
- Data Availability Statement must contain a real public URL or DOI, not `will be added`.

## Cover Letter Points

The cover letter should state:

- The paper is a diagnostic study of prequential answer-set calibration for temporal KG forecasting.
- The contribution is reliability-utility evaluation under strict temporal ordering.
- The study does not claim a new distribution-free theorem under temporal drift.
- The study does not claim state-of-the-art ranking performance.
- The paper provides reproducibility artifacts: code, configs, seeds, deletion masks or generation scripts, paper-facing result tables, and figures.

## Information-Specific Submission Notes

- Use the MDPI LaTeX template and select `Information` as the journal.
- Check whether the selected article type is `Article`.
- Confirm APC and any discount/waiver information in the MDPI submission system before final submission.
- Confirm whether the special issue, if any, has additional formatting or data-sharing requirements.
