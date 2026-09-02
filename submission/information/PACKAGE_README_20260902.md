# MDPI Information Submission Package

This package contains the final manuscript and the files needed to submit it to *Information*.

## Upload Ready

- `upload_ready/manuscript.pdf`: compiled manuscript for editorial review.
- `upload_ready/manuscript_source.zip`: LaTeX source, bibliography, official MDPI definitions, and figure files.
- `upload_ready/cover_letter.docx`: cover letter.

## Author Reference

- `author_reference/RiskCal-TKG_中文逐章翻译与解读.docx`: Chinese translation and reader-oriented explanation; do not upload as part of the English manuscript.
- `author_reference/EXPANSION_RESULT_SUMMARY_20260902.md`: result interpretation boundary.
- `author_reference/EXPANSION_CLAIM_LEDGER_20260901.md`: claim-to-evidence map.
- `author_reference/RELEASE_AUDIT_POST_MANUSCRIPT.json`: clean-clone release audit.

## Final Checks

Use `SUBMISSION_CHECKLIST.md` immediately before uploading. The repository URL and submission tag must be public and resolvable. `PACKAGE_SHA256SUMS.txt` records the integrity of every packaged file except the checksum file itself.

The package deliberately excludes raw ICEWS data, model checkpoints, terminal logs, local paths, transfer archives, tokens, and passwords.
