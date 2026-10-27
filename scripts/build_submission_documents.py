#!/usr/bin/env python3
"""Build the submission .docx files from the Markdown sources.

The response document is relabelled into the conference's Reviewer Concern / Author Response
form on the way through. Bare "---" horizontal rules are stripped: pandoc silently drops
subsequent content, including whole tables, when they appear in these documents, so the
sources avoid them and this script asserts they are absent.

Pandoc's output is then passed through the deterministic Word layout repair in
`timeentropy.reporting.docx_layout`, which removes duplicate image captions, applies
content-aware column widths, repeats header rows, and keeps captions with their content.
Content gates run before conversion: an unresolved placeholder, a missing bibliography, or a
missing declaration fails the build rather than reaching a reviewer.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from timeentropy.reporting.docx_layout import build_reference, highlight_revisions, postprocess

WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
PANDOC_FORMAT = "markdown+pipe_tables-yaml_metadata_block"
PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")
MARKED_NOTE = (
    "MARKED COPY. Yellow marks every paragraph, table, and figure that is new or rewritten "
    "relative to the submitted manuscript. Because the experiment was reconstructed in full "
    "after the original code and predictions were found to be unrecoverable, almost all text is "
    "new; unchanged material is left unmarked. The response letter gives the reviewer comment "
    "number for each table and figure."
)
REQUIRED_MANUSCRIPT_SECTIONS = (
    "## References",
    "## Conflict of Interest",
    "## Funding Information",
    "## Ethical Statement",
    "## Acknowledgement",
    "## Data and code availability",
)


def relabel_response(text: str) -> str:
    """Rewrite item headings into the conference response template's labels."""

    lines = []
    for line in text.split("\n"):
        heading = re.match(r"^### (R\d+(?:-EV)?\.\d+)\s+(.*)$", line)
        if heading:
            lines.append(f"**Reviewer Concern ({heading.group(1)}):** {heading.group(2).strip()}")
            continue
        lines.append(re.sub(r"^\*\*Author response\.\*\*", "**Author Response:**", line))
    joined = "\n".join(lines)
    return re.sub(
        r"^\*\*(R1-EV\.\d+) — (.+?)\*\*",
        r"**Reviewer Concern (\1):** \2\n\n**Author Response:**",
        joined,
        flags=re.MULTILINE,
    )


def check_source(text: str, source: Path, *, is_manuscript: bool, allow_placeholders: bool) -> list[str]:
    """Reject sources that are not submission-ready before spending a conversion on them."""

    failures = []
    rules = [i + 1 for i, line in enumerate(text.split("\n")) if line == "---"]
    if rules:
        failures.append(f"{source} contains bare '---' rules at lines {rules}")

    placeholders = sorted(set(PLACEHOLDER.findall(text)))
    if placeholders and not allow_placeholders:
        failures.append(
            f"{source} still contains unresolved placeholders {placeholders}; "
            "resolve them or rebuild with --allow-placeholders for a draft"
        )

    if is_manuscript:
        missing = [
            heading for heading in REQUIRED_MANUSCRIPT_SECTIONS if heading not in text
        ]
        if missing:
            failures.append(f"{source} is missing required sections {missing}")
        cited = {int(number) for number in re.findall(r"(?<![\w])\[(\d+)\]", text)}
        listed = {int(match.group(1)) for match in re.finditer(r"^\[(\d+)\]", text, re.MULTILINE)}
        uncited = sorted(listed - cited)
        unlisted = sorted(number for number in cited - listed if number > 0)
        if unlisted:
            failures.append(f"{source} cites references with no bibliography entry: {unlisted}")
        if uncited:
            failures.append(f"{source} lists references that are never cited: {uncited}")
    return failures


def answered_items(response: str) -> tuple[set[int], set[int]]:
    """Tables and figures the response names as answering a numbered reviewer comment."""

    tables: set[int] = set()
    figures: set[int] = set()
    for line in response.split("\n"):
        if not line.startswith("**Manuscript location:**"):
            continue
        for kind, span in re.findall(
            r"(Tables?|Figures?)\s+((?:\d+)(?:\s*(?:–|,|and)\s*\d+)*)", line
        ):
            target = tables if kind.startswith("Table") else figures
            for piece in re.split(r",|\band\b", span):
                piece = piece.strip()
                if "–" in piece:
                    first, last = (int(value) for value in piece.split("–"))
                    target.update(range(first, last + 1))
                elif piece.isdigit():
                    target.add(int(piece))
    return tables, figures


def convert(markdown: str, output: Path, reference: Path) -> None:
    """Render through pandoc, taking page geometry and typography from the venue's file."""

    if not reference.exists():
        raise SystemExit(f"reference document {reference} is absent; the venue layout is required")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(markdown)
        source = Path(handle.name)
    try:
        subprocess.run(
            [
                "pandoc",
                str(source),
                f"--from={PANDOC_FORMAT}",
                "--to=docx",
                f"--reference-doc={reference}",
                "-o",
                str(output),
            ],
            check=True,
        )
    finally:
        source.unlink()


def describe(path: Path) -> tuple[int, int]:
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    paragraphs = [
        "".join(node.text or "" for node in p.iter(f"{WORD}t")) for p in root.iter(f"{WORD}p")
    ]
    return len(list(root.iter(f"{WORD}tbl"))), len("\n".join(paragraphs))


def venue_reference(workspace: Path, template: Path) -> Path:
    """Produce the reference DOCX pandoc renders against, in the venue's own format."""

    default = workspace / "pandoc-default-reference.docx"
    default.write_bytes(
        subprocess.run(
            ["pandoc", "--print-default-data-file", "reference.docx"],
            check=True,
            capture_output=True,
        ).stdout
    )
    return build_reference(default, workspace / "venue-reference.docx", template)


def build(arguments: argparse.Namespace, reference: Path) -> list[str]:
    failures: list[str] = []
    sources = [arguments.manuscript, arguments.response]
    if arguments.supplement.is_file():
        sources.append(arguments.supplement)
    for source in sources:
        is_manuscript = source == arguments.manuscript
        is_supplement = source == arguments.supplement
        text = source.read_text()
        failures.extend(
            check_source(
                text,
                source,
                is_manuscript=is_manuscript,
                allow_placeholders=arguments.allow_placeholders,
            )
        )
        if is_supplement:
            labels = re.findall(r"^\*\*Table (S\d+)\.", text, re.MULTILINE)
            expected = [f"S{i}" for i in range(1, len(labels) + 1)]
            if labels != expected:
                failures.append(f"{source} supplementary labels are not sequential: {labels}")
        expected = text.count("\n|---")
        markdown = text if is_manuscript or is_supplement else relabel_response(text)
        if is_manuscript:
            name = "Time_Entropy_Dysarthric_Speech_Classification_REVISED.docx"
        elif is_supplement:
            name = "Supplementary_Material.docx"
        else:
            name = "Response_to_Reviewers.docx"
        output = arguments.output_dir / name
        convert(markdown, output, reference)
        try:
            removed, derived = postprocess(
                output,
                output,
                strict=True,
                require_all_tables=False,
                venue=arguments.original_manuscript if is_manuscript else None,
            )
        except ValueError as error:
            failures.append(f"{output.name} layout pass failed:\n{error}")
            removed, derived = [], []
        tables, characters = describe(output)
        status = "ok" if tables == expected else "TABLE LOSS"
        print(
            f"[{status:>10}] {output.name}: {tables}/{expected} tables, "
            f"{characters:,} characters, {len(removed)} duplicate captions removed, "
            f"{len(derived)} derived column layouts"
        )
        if tables != expected:
            failures.append(f"{output.name} rendered {tables} of {expected} tables")

        if is_manuscript and not arguments.no_marked_copy:
            if not arguments.original_manuscript.exists():
                failures.append(
                    f"{arguments.original_manuscript} is absent; the marked copy needs it to diff"
                )
            else:
                tables, figures = answered_items(arguments.response.read_text())
                marked = output.with_name(output.stem + "_MARKED.docx")
                counts = highlight_revisions(
                    output,
                    marked,
                    original_path=arguments.original_manuscript,
                    tables=tables,
                    figures=figures,
                    note=MARKED_NOTE,
                )
                share = 100.0 * counts["changed_paragraphs"] / max(counts["total_paragraphs"], 1)
                print(
                    f"[{'ok':>10}] {marked.name}: {counts['changed_paragraphs']}/"
                    f"{counts['total_paragraphs']} paragraphs changed ({share:.0f}%), "
                    f"{counts['tables']} tables and {counts['figures']} figures marked"
                )
                if counts["tables"] != len(tables) or counts["figures"] != len(figures):
                    failures.append(
                        f"{marked.name} marked {counts['tables']}/{len(tables)} tables and "
                        f"{counts['figures']}/{len(figures)} figures named by the response"
                    )

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=Path("docs/revised_manuscript.md"))
    parser.add_argument("--response", type=Path, default=Path("docs/reviewer_response.md"))
    parser.add_argument(
        "--supplement", type=Path, default=Path("docs/supplementary_material.md")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--original-manuscript",
        type=Path,
        default=Path("Time_Entropy_Dysarthric_Speech_Classification_IRO_Formatted.docx"),
        help="Submitted manuscript used to produce the highlighted marked copy.",
    )
    parser.add_argument(
        "--no-marked-copy",
        action="store_true",
        help="Skip the highlighted revision copy the venue may require alongside the clean file.",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Build a draft whose sources still contain unresolved {{PLACEHOLDER}} tokens.",
    )
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as workspace:
        reference = venue_reference(Path(workspace), arguments.original_manuscript)
        failures = build(arguments, reference)

    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit("submission documents did not build cleanly")

if __name__ == "__main__":
    main()
