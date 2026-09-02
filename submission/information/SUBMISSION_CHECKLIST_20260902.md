# MDPI Information Submission Checklist

## Files to Upload

- [ ] `manuscript.pdf` opens correctly and has no missing figures or references.
- [ ] `manuscript.tex`, `references.bib`, `Definitions/`, and `figures/` are uploaded together as the LaTeX source package.
- [ ] `cover_letter.docx` is uploaded or its text is pasted into the submission system.
- [ ] The author name, affiliation, email, and funding information entered in the submission system match the manuscript.
- [ ] The GitHub repository and tag are public before final submission.

## Manuscript Checks

- [x] Journal class is `information` and the source uses the official MDPI class supplied on 23 July 2026.
- [x] Compiled manuscript length is 10 pages, below the requested 15-page ceiling.
- [x] Abstract length is 175 words under whitespace tokenization.
- [x] Title does not contain the former dataset-specific subtitle.
- [x] Proposition 2 is bold.
- [x] The query symbol `?` is explicitly defined.
- [x] No pseudo-code, local filesystem path, archive filename, or terminal command appears in the manuscript.
- [x] Results disclose set-size cost, multi-answer degradation, and null or inconclusive stress tests.
- [x] KGCP and continuous complex scorer identities are bounded accurately.
- [x] Author Contributions, Funding, Institutional Review Board, Informed Consent, Data Availability, Generative AI disclosure, and Conflicts of Interest are present.

## Repository and Evidence Checks

- [x] `main` contains the final manuscript source, figures, source CSV files, checksums, and post-manuscript release audit.
- [x] Tag `v1.0.0-mdpi-information-submission` resolves to the release metadata commit immediately following audited candidate `ff5b959`.
- [x] A fresh server-side clone of audited candidate `ff5b959` passes all 124 project tests; the tag adds only the audit report and release metadata.
- [x] The public repository landing page describes both datasets, both scorers, and the bounded reliability--utility claim.
- [x] No password, token, private key, raw proprietary file, local absolute path, checkpoint, or transfer archive is tracked.

## Statements to Confirm in the Submission System

- [ ] The manuscript is original and not under consideration elsewhere.
- [ ] The single author has approved the submitted version.
- [ ] The funder had no role in study design, analysis, interpretation, writing, or publication decision.
- [ ] The Generative AI disclosure entered in the system matches the manuscript.
- [ ] Suggested and opposed reviewers, if requested, are selected by the author without conflicts of interest.

## Author-Only Reference Files

The Chinese translation, evidence ledger, result summary, and release-audit report are for author understanding and verification. They are not manuscript files unless an editor specifically requests them.
