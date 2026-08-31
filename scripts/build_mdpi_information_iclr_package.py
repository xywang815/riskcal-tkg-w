from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "submission"
    / "information"
    / "MDPI_Information_ICLR_Revision_20260831"
)


def remove_inactive_blocks(text: str) -> str:
    r"""Remove TeX \iffalse ... \fi blocks, including nested inactive blocks."""
    token = re.compile(r"\\iffalse\b|\\fi\b")
    output: list[str] = []
    cursor = 0
    while True:
        start = re.search(r"\\iffalse\b", text[cursor:])
        if start is None:
            output.append(text[cursor:])
            break
        block_start = cursor + start.start()
        output.append(text[cursor:block_start])
        depth = 0
        block_end = None
        for match in token.finditer(text, block_start):
            if match.group() == r"\iffalse":
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    block_end = match.end()
                    break
        if block_end is None:
            raise ValueError("Unterminated \\iffalse block in manuscript source")
        cursor = block_end
    return "".join(output)


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "sections").mkdir(exist_ok=True)
    (PACKAGE / "figures" / "iclr_revision").mkdir(parents=True, exist_ok=True)

    source = ROOT / "paper" / "manuscript_mdpi_information_iclr_revision.tex"
    clean = remove_inactive_blocks(source.read_text(encoding="utf-8"))
    clean = clean.replace(
        "% MDPI Information submission draft based on the 27 July 2026 class release.",
        "% MDPI Information submission source based on the 27 July 2026 class release.",
        1,
    )
    (PACKAGE / "manuscript.tex").write_text(clean, encoding="utf-8")

    copy_tree(ROOT / "paper" / "Definitions", PACKAGE / "Definitions")
    shutil.copy2(ROOT / "paper" / "references.bib", PACKAGE / "references.bib")

    for name in (
        "theory_scope_iclr_revision.tex",
        "experimental_design_iclr_revision.tex",
        "results_discussion_iclr_revision.tex",
    ):
        shutil.copy2(ROOT / "paper" / "sections" / name, PACKAGE / "sections" / name)

    for name in ("empirical_overview.pdf", "utility_robustness.pdf"):
        shutil.copy2(
            ROOT / "paper" / "figures" / "iclr_revision" / name,
            PACKAGE / "figures" / "iclr_revision" / name,
        )

    readme = """# MDPI Information submission package

This directory contains the clean submission source for **Prequential Answer-Set
Calibration for Temporal Knowledge Graph Forecasting**. It uses the MDPI class
release dated 27 July 2026 supplied with the journal template.

## Contents

- `manuscript.tex`: clean manuscript source with no disabled legacy sections.
- `Definitions/`: official MDPI class and bibliography files.
- `sections/`: the three active modular sections.
- `figures/iclr_revision/`: the two vector figures used in the manuscript.
- `references.bib`: bibliography database.
- `manuscript.pdf`: compiled submission PDF (added by the release build).
- `COVER_LETTER.md`: draft cover letter for *Information*.
- `SUBMISSION_CHECKLIST.md`: author-side checks before upload.

## Reproducibility

Code, fixed configurations, seeds, and paper-facing derived artifacts are public
at <https://github.com/xywang815/riskcal-tkg-w>. ICEWS14 must be obtained from
its original public source according to its distribution terms.

## Local compilation

Run `pdflatex manuscript.tex`, `bibtex manuscript`, and `pdflatex manuscript.tex`
twice from this directory. The study is empirical and does not claim a new
distribution-free conformal coverage theorem.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")

    cover_letter = """Dear Editors of *Information*,

Please consider our manuscript, **“Prequential Answer-Set Calibration for
Temporal Knowledge Graph Forecasting,”** for publication as an Article in
*Information*.

Temporal knowledge graph forecasting is commonly assessed with ranking metrics,
which do not reveal whether a deployed model can return answer sets that cover
observed correct entities at a stated rate. The manuscript studies this problem
under a strict timestamp-batched prequential protocol: predictions for a test
timestamp are emitted before any label from that timestamp is used for
calibration. On ICEWS14, five-seed controlled history-deletion experiments show
that recent-history calibration repairs the undercoverage of a fixed threshold,
but often produces large answer sets and does not remove relation-side or
feedback-delay limitations. Validation-selected RAPS provides a smaller average
operating point, with a measurable query-level coverage trade-off.

The contribution is a reliability diagnostic and leakage-controlled empirical
evaluation, not a claim of a new distribution-free coverage theorem. The paper
reports negative and subgroup diagnostics alongside average performance and
provides code, fixed configurations, seeds, and paper-facing artifacts at
<https://github.com/xywang815/riskcal-tkg-w>.

This manuscript is original, is not under consideration elsewhere, and has been
approved by the author. The author declares no conflicts of interest.

Sincerely,

Xinyu Wang  
Anhui Institute of Information Technology  
xywang68@iflytek.com
"""
    (PACKAGE / "COVER_LETTER.md").write_text(cover_letter, encoding="utf-8")

    checklist = """# Submission checklist

- [x] Journal class set to `information` using the official 27 July 2026 MDPI template.
- [x] Title does not contain a dataset-specific subtitle.
- [x] Abstract is one paragraph and no longer than 200 words.
- [x] Present-timestamp labels are excluded from calibration before prediction.
- [x] RAPS grid, temperature, selection tolerance, and selected parameters are stated.
- [x] Proposition 2 is typeset in bold.
- [x] Result claims are tied to ICEWS14 evidence and five fixed seeds.
- [x] Figures are complete vector graphics; the former incomplete Figure 4 is absent.
- [x] Discussion and conclusion are merged and include limitations and future work.
- [x] Author Contributions, Funding, Data Availability, and Conflicts of Interest are present.
- [x] Public code URL is present and resolves to the author-owned repository.
- [ ] Confirm the public repository matches the exact submitted source revision.
- [ ] Enter any ORCID requested by the submission system.
- [ ] Upload the raw `.tex`, `Definitions/`, figures, and bibliography with the PDF.
- [ ] Confirm the manuscript is not simultaneously submitted elsewhere.
"""
    (PACKAGE / "SUBMISSION_CHECKLIST.md").write_text(checklist, encoding="utf-8")

    print(PACKAGE)


if __name__ == "__main__":
    main()
