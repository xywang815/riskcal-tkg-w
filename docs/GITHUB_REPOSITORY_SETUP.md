# GitHub Repository Setup for RiskCal-TKG

This guide prepares a long-term GitHub repository that can host this paper and future experiments.

## Recommended Repository Strategy

Use one personal long-term repository:

- Repository name: `riskcal-tkg-w`
- Visibility for MDPI Information submission: public is acceptable unless the journal or special issue explicitly requests double-blind review.
- Owner: your GitHub account.
- Purpose: source code, configs, paper-facing derived tables, figures, reproducibility instructions, and archived releases.

Do not upload:

- server passwords, SSH commands, screenshots, local paths, or rental-platform information;
- raw ICEWS14 archives if the original license or dataset source expects users to obtain them separately;
- temporary patch archives, transfer archives, logs, checkpoints, or build files.

## Files That Should Be Included

Include:

- `README.md`
- `pyproject.toml`
- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `data/README.md`
- `paper/manuscript.tex`
- `paper/references.bib`
- `paper/sections/`
- `paper/figures/final_confirmatory/`
- `paper/data/final_confirmatory/` paper-facing CSV/JSON summaries
- `submission/information/` submission checklist and cover-letter draft

The repository can keep derived CSV/JSON results because they are needed for review and are much smaller than raw checkpoints. If a file is too large for GitHub, move it to a release asset or Zenodo record.

## GitHub Web Steps

1. Open GitHub and sign in.
2. Click the `+` button in the top right.
3. Choose `New repository`.
4. Repository name: `riskcal-tkg-w`.
5. Description: `Prequential calibration diagnostics for temporal knowledge graph forecasting`.
6. Choose `Public` for a normal MDPI Information submission.
7. Do not initialize with README, .gitignore, or license if this local folder already has those files.
8. Click `Create repository`.
9. Copy the repository URL, for example:
   `https://github.com/xywang815/riskcal-tkg-w.git`.

## Local Terminal Commands

Run these in your Mac Terminal after the repository is created on GitHub. Replace `<project-dir>` with the local project directory on your own computer:

```bash
cd <project-dir>
git init
git status
git add README.md LICENSE pyproject.toml src scripts configs tests data paper submission docs .gitignore
git status
git commit -m "Prepare RiskCal-TKG reproducibility repository"
git branch -M main
git remote add origin https://github.com/xywang815/riskcal-tkg-w.git
git push -u origin main
```

If Git asks who you are:

```bash
git config --global user.name "Xinyu Wang"
git config --global user.email "xywang68@iflytek.com"
```

## Release and DOI Plan

Before submission:

1. Push a stable repository version.
2. Create a GitHub release named `v1.0-information-submission`.
3. Archive that release on Zenodo.
4. Add the Zenodo DOI to the manuscript and Data Availability Statement.

If the DOI is not ready on the first submission day, at minimum use the public GitHub URL in the manuscript. Replace it with a DOI after Zenodo is created.
