# RiskCal-TKG Current Work State

Last updated: 2026-08-31 (Asia/Shanghai)

This is the authoritative handoff record for continuing the project. Read this file before resuming experiments or editing the manuscript. Do not infer the current state from older Desktop exports or earlier submission directories.

## 1. Current objective

- Target venue: MDPI *Information*.
- Quality objective: use an ICLR-level empirical standard for experimental rigor, while formatting the final submission with the official MDPI *Information* template.
- Scientific scope: reliability and utility diagnostics for prequential answer-set calibration in temporal knowledge graph forecasting.
- Immediate research priority: broaden the evidence beyond the current ICEWS14 and DistMult study before the next unified manuscript revision.
- Writing rule confirmed by the user: finish and verify the planned experiment batch first, then revise the manuscript once. Do not repeatedly rewrite and reformat the paper after each individual experiment.

## 2. Latest authoritative deliverables

Only the following three RiskCal-TKG exports are intentionally retained on the Desktop:

| Artifact | Desktop path | SHA-256 |
|---|---|---|
| Submission package | `/Users/wangxinyu/Desktop/MDPI_Information_ICLR_Revision_20260831.zip` | `c9c0cced6021e539d3ae28343c9e8093e96811208709c9446a400cfd29129209` |
| English manuscript PDF | `/Users/wangxinyu/Desktop/RiskCal-TKG_MDPI_Information_ICLR_Revision_20260831.pdf` | `b47d9bb467003a44dc96e876b36585b99476e0f59f0c58ac5522cf9d074c548d` |
| Chinese close translation | `/Users/wangxinyu/Desktop/RiskCal-TKG_MDPI_Information_最新版中文逐字翻译稿_20260831.docx` | `b4a155f8ae33c0116631b1534e9cddc368bd62175e1022e5ca1ac6feb9f83e5a` |

On 2026-08-31, thirteen older Desktop exports were deleted at the user's request. They occupied approximately 7.4 MB. The original source PDF `/Users/wangxinyu/Desktop/riskcal_tkg_manuscript.pdf` was not deleted.

Project copies and authoritative editable sources:

- Package directory: `submission/information/MDPI_Information_ICLR_Revision_20260831/`
- English PDF: `output/pdf/RiskCal-TKG_MDPI_Information_ICLR_Revision_20260831.pdf`
- Revised manuscript: `paper/manuscript_mdpi_information_iclr_revision.tex`
- Official 27 July 2026 MDPI template: `paper/Definitions/`
- Theory section: `paper/sections/theory_scope_iclr_revision.tex`
- Experimental design: `paper/sections/experimental_design_iclr_revision.tex`
- Results and discussion: `paper/sections/results_discussion_iclr_revision.tex`
- Revised figures: `paper/figures/iclr_revision/`
- Figure builder: `scripts/build_iclr_revision_figures.py`
- Package builder: `scripts/build_mdpi_information_iclr_package.py`
- Chinese translation builder: `scripts/build_chinese_translation_iclr_revision.py`

The superseded 22-page baseline and earlier manuscript sources were removed from the
working tree during the 2026-08-31 cleanup. They remain recoverable from Git history at
the recorded pre-cleanup commit and must not be restored as active manuscript sources.

## 3. Current manuscript status

- Title: *Prequential Answer-Set Calibration for Temporal Knowledge Graph Forecasting*.
- The suffix `An ICEWS14 Diagnostic Study` has been removed from the title.
- The abstract is 194 words and follows a problem-gap-method-evidence-implication structure.
- The query notation explicitly states that `?` is the predicted entity slot; it is not missing text. The notation `r^{-1}` denotes an inverse relation.
- Proposition 2 is bold.
- The five-step pseudocode is retained in a compact, more readable form.
- The incomplete earlier Figure 4 was replaced by two complete vector multi-panel figures generated from verified experiment tables.
- The former Sections 7-10 were consolidated into `Discussion and Conclusion`.
- The active manuscript contains no `route-A`, repository placeholder, local archive name, local command, local path, or unresolved TODO.
- Repository URL used in the manuscript: `https://github.com/xywang815/riskcal-tkg-w`.
- Required author, data availability, funding, and conflict declarations are present.
- The English PDF uses the official MDPI *Information* class released 27 July 2026 and currently compiles to 16 A4 pages.
- The compile log was checked: no undefined references, overfull/underfull boxes, or substantive LaTeX warnings were found.
- After consolidating the template at `paper/Definitions/`, a clean temporary build again produced a 16-page PDF without matching error, undefined-reference, overfull, or underfull diagnostics.
- The submission ZIP passed `unzip -t` and excludes build and QA directories.
- The Chinese DOCX renders as 24 pages and was visually checked page by page after installing a CJK-capable font in the bundled LibreOffice QA runtime.

The manuscript is deliberately limited to the verified ICEWS14 evidence. It does not claim cross-dataset or cross-backbone generalization.

## 4. Verified ICEWS14 evidence

### Dataset and protocol

- 7,128 entities, 230 relations, 90,730 facts, and 365 timestamps.
- Strict temporal split: timestamps 1-219 for training, 220-292 for validation/calibration, and 293-365 for testing.
- Corresponding facts: 52,993 training, 18,439 validation/calibration, and 19,298 test.
- Deletion rates: 0.0, 0.1, 0.2, and 0.3.
- Seeds: 17, 29, 43, 59, and 71.
- Primary condition: 30% deletion with target coverage 0.90.
- Timestamp-block bootstrap: 20,000 repetitions, default block length 7; sensitivity lengths 3, 14, and 21.
- Strict prequential rule: labels at the current test timestamp are not available before predictions for that timestamp are produced.

### Main results at 30% deletion

- DistMult MRR: 0.3105; frequency baseline MRR: 0.0925.
- Paired difference: 0.2180, 95% CI [0.2070, 0.2297], one-sided bootstrap p < 0.0001.
- Train-only repeat MRR: 0.2774; prequential repeat MRR: 0.3631.
- Static coverage: 0.8235; rolling coverage: 0.8998.
- Rolling undercoverage reduction versus static: 0.0723, 95% CI [0.0535, 0.0938].
- Fixed weighted coverage: 0.8988; validation-selected half-life coverage: 0.8989.

### Query and subgroup diagnostics

- 33,964 unique temporal queries and 38,596 answer labels per seed.
- 10.1% of queries have multiple answers; the maximum answer count is 17.
- Full-set coverage: 0.8267 to 0.9020; partial recall: 0.8463 to 0.9174.
- Label-weighted mean set size: 2,748.6 to 3,802.5; P90 set size: 7,128.
- Full-vocabulary prediction sets occur for 36.5% of queries under the relevant rolling analysis.
- Worst relation-side coverage: 0.6274 to 0.7625.
- Relation-side groups below target: 63.2% to 22.4%.

### Method and sensitivity diagnostics

- Margin rolling: mean set size 4,069.4; full-set coverage 0.9020.
- Rank rolling: mean set size 3,668.2; full-set coverage 0.8881; no full-vocabulary sets.
- APS mean set size: 4,911.8.
- Validation-selected RAPS uses softmax temperature T = 1, a validation selection tolerance of 0.02, and 21 fixed candidates.
- All 20 seed/deletion conditions selected `k = 50` and `lambda = 1e-4`.
- Validation-selected RAPS at the primary condition: observed-label coverage 0.8987; mean set size 3,319.0; median 2,924.6; P90 4,646.5; full-set coverage 0.8874.
- Window sensitivity coverage: expanding 0.8590; R250 0.9129; R500 0.9001; R1000 0.8998; R2000 0.8964.
- The R1000 calibration pool spans a median of two timestamps.
- Effective sample size for half-life 7: 997.6; all validation conditions selected equal weighting.
- Feedback delay of seven timestamps gives coverage 0.8884.
- Confirmatory runtime: 88.4 minutes on one RTX 4090.
- Mean training, calibration, and inference times per condition: 203.6, 13.0, and 45.4 seconds.
- Peak GPU memory: 1.05 GiB.

All numbers above must remain traceable to generated result artifacts. If a future rerun produces a different value, record the run ID and reconcile the discrepancy before changing the manuscript.

## 5. Non-negotiable research and writing constraints

- Do not fabricate data, results, references, implementation details, or generalization claims.
- Report negative and mixed results, including coverage-efficiency trade-offs.
- Keep the contribution framed as reliability diagnostics and strict prequential evaluation, not a new distribution-free coverage theorem.
- Use validation-only model or hyperparameter selection. Never select a method or parameter using test labels.
- Keep timestamp-batched prequential evaluation and prevent current-timestamp label leakage.
- Retain fixed seeds and paired timestamp-block bootstrap inference.
- Include query-level and subgroup diagnostics when extending the experiments.
- Do not place screenshots, local paths, archive names, terminal commands, or internal run identifiers in the paper body.
- In the reproducibility statement, report the software/hardware configuration, public repository URL, and data source concisely.
- Remove AI-like filler, inflated novelty claims, repetitive transitions, and generic promotional language.
- Do not update the manuscript after each run. Record results first; perform one consolidated manuscript revision only after the experiment batch passes QA.

## 6. Expansion experiment plan

The next experiment batch is intended to test whether the observed reliability-utility behavior is specific to ICEWS14 and DistMult.

1. Audit the current data loader, scorer, training loop, and configuration interfaces for multi-dataset and multi-backbone support.
2. Audit candidate temporal KG datasets for lawful availability, compatible timestamps, sufficient query multiplicity, and reproducible preprocessing. Candidate names must not be committed to the paper until this audit is complete.
3. Audit a second temporal KGE backbone for implementation stability and comparability with the current DistMult setup.
4. Select at least one additional dataset, preferably two if preprocessing and compute remain controlled, plus one non-DistMult backbone.
5. Add smoke tests and a single-seed pilot before launching the full matrix.
6. Run the controlled full matrix using the same prequential protocol, target coverage, seed policy, leakage checks, diagnostics, and paired timestamp-block bootstrap where applicable.
7. Produce machine-readable manifests, per-condition outputs, aggregate tables, figures, and a success/failure gate without editing the paper.
8. Audit every manuscript-ready number against the artifacts, then make one unified English manuscript revision and rebuild the Chinese translation.

Dataset and second-backbone choices are deliberately marked as pending audit. Do not invent support for ICEWS18, GDELT, or another model before verifying the local code and the data license/source.

## 7. Server access and secret policy

- Authorized SSH endpoint: `root@connect.bjb1.seetacloud.com`, port `14000`.
- The user has authorized direct access and experiment execution.
- Never write the server password to this file, the repository, scripts, shell history helpers, logs, or experiment manifests.
- Revalidate the environment at the beginning of the next server session; historical observations may no longer be current.
- Historical environment: `/root/riskcal_tkg_mvp`, RTX 4090 24 GB, PyTorch 2.5.1+cu124, CUDA 12.4 available, Python 3.12.x.
- Historical confirmatory run root: `/root/riskcal_tkg_mvp/results/final_confirmatory/20260815T110014908249Z-eb736dbf6658`.
- No new SSH session or expansion experiment was started on 2026-08-31 before this handoff record was written.

## 8. Repository state at handoff

- Local repository: `/Users/wangxinyu/Documents/Codex/riskcal_tkg_mvp`.
- GitHub repository: `https://github.com/xywang815/riskcal-tkg-w`.
- Git HEAD at recording time: `9992a10f9d1a6fbb951dddc899db847a70a84e37`.
- The working tree contains untracked generated output, revised paper sources, figure files, build scripts, and submission package directories. Do not run destructive cleanup or reset commands.
- The obsolete 2026-08-27 submission directories, old manuscript/PDF variants, render caches, LaTeX intermediates, and duplicate template were removed on 2026-08-31.
- The main project decreased from 367,892 KiB to 81,468 KiB. Old review QA caches decreased from 110,008 KiB to 408 KiB. A further approximately 46 MiB of obsolete PDF render/build output was removed from the 2026-08-11 legacy work area while its code and analysis data were retained. The cleanup released approximately 433 MiB in total.
- Protected material was retained: `data/`, `paper/data/`, `src/`, `scripts/`, `configs/`, `tests/`, experiment archives, patch archives, experiment logs, Git history, review originals, and all 2026-08-31 submission artifacts.

## 9. Resume checklist

1. Read this file and confirm the three Desktop hashes if artifact integrity matters.
2. Inspect `git status --short`; preserve all existing user and generated changes.
3. Rebuild or inspect the latest English PDF only if needed; do not modify the manuscript yet.
4. Audit the local code paths for dataset and backbone extensibility.
5. Connect to the authorized server without persisting credentials.
6. Revalidate GPU, CUDA, PyTorch, Python, disk space, repository path, and current background processes.
7. Synchronize only the required code/config changes and run tests plus a smoke experiment.
8. Record each new run ID, config hash, code commit/hash, environment, artifact checksums, and any failures.
9. Complete and QA the expansion batch before revising the paper.
10. After the unified revision, rebuild the official MDPI package and the Chinese close translation, then update this handoff record.
