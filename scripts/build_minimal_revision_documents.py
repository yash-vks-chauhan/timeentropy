#!/usr/bin/env python3
"""Build the restrained reviewer-revision submission set.

The original IRO manuscript controls page geometry and typography. The marked copy uses
paragraph alignment plus token-level differences for similar paragraphs, avoiding the blanket
paragraph/table highlighting used by the earlier full-reconstruction document.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from copy import deepcopy
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

from timeentropy.reporting.docx_layout import (
    build_reference,
    document_paragraphs,
    document_usable_twips,
    ensure_caption_styles,
    highlight_runs,
    normalize,
    postprocess,
    set_table_widths,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "Time_Entropy_Dysarthric_Speech_Classification_IRO_Formatted.docx"
MANUSCRIPT = ROOT / "docs/minimal_revised_manuscript.md"
SUPPLEMENT = ROOT / "docs/minimal_supplementary_material.md"
RESPONSE = ROOT / "docs/minimal_reviewer_response.md"

CLEAN = ROOT / "Time_Entropy_Dysarthric_Speech_Classification_MINIMAL_REVISED.docx"
MARKED = ROOT / "Time_Entropy_Dysarthric_Speech_Classification_MINIMAL_REVISED_MARKED.docx"
SUPPLEMENT_OUT = ROOT / "Supplementary_Material_MINIMAL_REVISION.docx"
RESPONSE_OUT = ROOT / "Response_to_Reviewers_MINIMAL_REVISION.docx"

PANDOC_FORMAT = "markdown+pipe_tables-yaml_metadata_block"
TOKEN = re.compile(r"\s+|[\w]+|[^\w\s]", re.UNICODE)
SUPPLEMENT_CAPTION = re.compile(r"^Table\s+S\d+\.\s+\S")
MARKED_NOTE = (
    "MARKED COPY. Yellow identifies new or corrected text, numerical values, tables, and "
    "captions. Language retained from the submitted manuscript is unmarked. Detailed "
    "reviewer-requested evidence is provided in the accompanying Supplementary Material."
)


def convert(source: Path, output: Path, reference: Path) -> None:
    subprocess.run(
        [
            "pandoc",
            str(source),
            f"--from={PANDOC_FORMAT}",
            "--to=docx",
            f"--reference-doc={reference}",
            "--resource-path=.",
            "-o",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )


def prepare_reference(workspace: Path) -> Path:
    default = workspace / "pandoc-default-reference.docx"
    default.write_bytes(
        subprocess.run(
            ["pandoc", "--print-default-data-file", "reference.docx"],
            check=True,
            capture_output=True,
        ).stdout
    )
    return build_reference(default, workspace / "venue-reference.docx", ORIGINAL)


def paragraph_text(node) -> str:
    parts: list[str] = []
    for element in node.iter():
        if element.tag in {qn("w:t"), qn("m:t")}:
            parts.append(element.text or "")
    return normalize("".join(parts))


def add_highlight(run) -> None:
    properties = run.find(qn("w:rPr"))
    if properties is None:
        properties = OxmlElement("w:rPr")
        run.insert(0, properties)
    highlight = properties.find(qn("w:highlight"))
    if highlight is None:
        highlight = OxmlElement("w:highlight")
        properties.append(highlight)
    highlight.set(qn("w:val"), "yellow")


def clear_highlight(node) -> None:
    """Remove revision highlighting while preserving all text and run formatting."""

    for properties in node.iter(qn("w:rPr")):
        for highlight in list(properties.findall(qn("w:highlight"))):
            properties.remove(highlight)


def changed_ranges(old: str, new: str) -> list[tuple[int, int]]:
    old_tokens = [(match.group(), match.start(), match.end()) for match in TOKEN.finditer(old)]
    new_tokens = [(match.group(), match.start(), match.end()) for match in TOKEN.finditer(new)]
    matcher = SequenceMatcher(
        a=[token.casefold() for token, _, _ in old_tokens],
        b=[token.casefold() for token, _, _ in new_tokens],
        autojunk=False,
    )
    ranges: list[tuple[int, int]] = []
    for tag, _a0, _a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal" or b0 == b1:
            continue
        start = new_tokens[b0][1]
        end = new_tokens[b1 - 1][2]
        ranges.append((start, end))
    return ranges


def simple_text_run(run) -> str | None:
    allowed = {qn("w:rPr"), qn("w:t")}
    if any(child.tag not in allowed for child in run):
        return None
    texts = list(run.iter(qn("w:t")))
    if not texts:
        return None
    return "".join(text.text or "" for text in texts)


def set_run_text(run, text: str) -> None:
    for child in list(run):
        if child.tag != qn("w:rPr"):
            run.remove(child)
    node = OxmlElement("w:t")
    if text[:1].isspace() or text[-1:].isspace():
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = text
    run.append(node)


def highlight_character_ranges(paragraph, ranges: list[tuple[int, int]]) -> None:
    """Highlight only intersecting characters in simple Word runs.

    Complex runs containing drawings, fields, breaks, or tabs are left structurally intact and
    highlighted as a whole when they intersect a changed range.
    """

    offset = 0
    for run in list(paragraph.iter(qn("w:r"))):
        text = "".join(node.text or "" for node in run.iter(qn("w:t")))
        if not text:
            continue
        start, end = offset, offset + len(text)
        offset = end
        local: list[tuple[int, int]] = []
        for first, last in ranges:
            left, right = max(first, start), min(last, end)
            if left < right:
                local.append((left - start, right - start))
        if not local:
            continue
        simple = simple_text_run(run)
        if simple is None or simple != text:
            add_highlight(run)
            continue

        boundaries = {0, len(text)}
        for first, last in local:
            boundaries.update((first, last))
        points = sorted(boundaries)
        parent = run.getparent()
        position = parent.index(run)
        parent.remove(run)
        for first, last in pairwise(points):
            if first == last:
                continue
            clone = deepcopy(run)
            set_run_text(clone, text[first:last])
            if any(first < hi and last > lo for lo, hi in local):
                add_highlight(clone)
            parent.insert(position, clone)
            position += 1


def make_marked(clean: Path, marked: Path) -> dict[str, int]:
    original = Document(ORIGINAL)
    revised = Document(clean)
    original_nodes = document_paragraphs(original)
    revised_nodes = document_paragraphs(revised)
    original_texts = [paragraph_text(node) for node in original_nodes]
    original_nonempty = [text for text in original_texts if text]
    exact = set(original_nonempty)

    full = 0
    partial = 0
    unchanged = 0
    for node in revised_nodes:
        new = paragraph_text(node)
        if not new:
            continue
        if new in exact:
            unchanged += 1
            continue

        plausible = [
            old
            for old in original_nonempty
            if min(len(old), len(new)) / max(len(old), len(new)) >= 0.45
        ]
        best = ""
        ratio = 0.0
        for old in plausible:
            score = SequenceMatcher(a=old, b=new, autojunk=False).ratio()
            if score > ratio:
                best, ratio = old, score
        if ratio >= 0.58 and not node.xpath(".//m:oMath | .//w:drawing"):
            ranges = changed_ranges(best, new)
            if ranges:
                highlight_character_ranges(node, ranges)
                partial += 1
            else:
                unchanged += 1
        else:
            highlight_runs(node)
            full += 1

    # Author metadata is unchanged. Superscript/run differences between the two Word files can
    # otherwise produce false token-level marks here, which obscures the substantive revision.
    for paragraph in revised.paragraphs[:7]:
        clear_highlight(paragraph._p)

    # These retained passages differ from the source only in typographic punctuation emitted by
    # pandoc (curly apostrophe and spacing around a slash), not in wording or meaning.
    for paragraph in revised.paragraphs:
        text = paragraph.text.strip()
        if text.startswith("Dysarthria is a motor speech disorder") or text == (
            "Conflict of Interest / Competing Interests"
        ):
            clear_highlight(paragraph._p)

    # References 1–25 are retained sources. Minor bibliography-punctuation normalization is
    # not a scientific change; only the seven newly added sources remain marked.
    in_references = False
    for paragraph in revised.paragraphs:
        text = paragraph.text.strip()
        if text == "References":
            in_references = True
            clear_highlight(paragraph._p)
            continue
        if not in_references:
            continue
        match = re.match(r"^\[(\d+)\]", text)
        if match and int(match.group(1)) <= 25:
            clear_highlight(paragraph._p)

    first = next((p for p in revised.paragraphs if p.text.strip()), None)
    if first is not None:
        note = first.insert_paragraph_before(MARKED_NOTE)
        note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        highlight_runs(note._p)

    revised.save(marked)
    return {"full": full, "partial": partial, "unchanged": unchanged}


def style_supplement_captions(path: Path) -> None:
    document = Document(path)
    ensure_caption_styles(document)
    for paragraph in document.paragraphs:
        if SUPPLEMENT_CAPTION.match(paragraph.text.strip()):
            paragraph.style = "Table Caption"
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.keep_together = True
    document.save(path)


def compact_references(path: Path) -> None:
    """Keep the expanded bibliography on two full pages without changing page geometry."""

    document = Document(path)
    in_references = False
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == "References":
            in_references = True
            continue
        if not in_references or not re.match(r"^\[\d+\]", paragraph.text.strip()):
            continue
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.0
        for run in paragraph.runs:
            run.font.size = Pt(10.5)
    document.save(path)


def tune_main_tables(path: Path) -> None:
    """Prevent the fold-summary label from breaking inside the word 'Mean'."""

    document = Document(path)
    usable_twips = document_usable_twips(document)
    for table in document.tables:
        header = tuple(cell.text.strip() for cell in table.rows[0].cells)
        if len(header) == 7 and header[:2] == ("Fold", "n") and "Mean observation" in header:
            set_table_widths(
                table,
                (10, 10, 14, 15, 18, 21, 12),
                usable_twips,
                points=9.0,
            )
    document.save(path)


def tune_supplement_tables(path: Path) -> None:
    """Tune dense validation tables without changing the supplement's page geometry."""

    document = Document(path)
    usable_twips = document_usable_twips(document)
    for table in document.tables:
        header = tuple(cell.text.strip() for cell in table.rows[0].cells)
        if len(header) == 9 and header[:2] == ("Fold", "Fixed-300 validation macro-F1"):
            set_table_widths(
                table,
                (8, 14, 12, 12, 8, 8, 13, 12, 13),
                usable_twips,
                points=6.5,
            )
        elif len(header) == 3 and header[:2] == ("Metric", "Pooled estimate"):
            set_table_widths(
                table,
                (38, 24, 38),
                usable_twips,
                points=9.0,
            )
        elif len(header) == 7 and header[:2] == ("Fold", "n"):
            set_table_widths(
                table,
                (12, 8, 13, 13, 18, 21, 15),
                usable_twips,
                points=7.5,
            )
        elif len(header) == 7 and header[:2] == ("Policy", "Accuracy"):
            set_table_widths(
                table,
                (27, 10, 10, 13, 15, 10, 15),
                usable_twips,
                points=7.0,
            )
    for paragraph in document.paragraphs:
        if paragraph.text.startswith("Entropy-only and maximum-softmax policies"):
            paragraph.paragraph_format.space_before = Pt(6)
        elif paragraph.text.startswith("Panel B. Pooled estimates"):
            paragraph.paragraph_format.keep_with_next = True
    document.save(path)


def main() -> None:
    if not ORIGINAL.is_file():
        raise SystemExit(f"Missing original manuscript: {ORIGINAL}")
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp/minimal_revision") as temporary:
        reference = prepare_reference(Path(temporary))

        convert(MANUSCRIPT, CLEAN, reference)
        postprocess(CLEAN, CLEAN, strict=True, require_all_tables=True, venue=ORIGINAL)
        tune_main_tables(CLEAN)
        compact_references(CLEAN)

        convert(SUPPLEMENT, SUPPLEMENT_OUT, reference)
        postprocess(SUPPLEMENT_OUT, SUPPLEMENT_OUT, strict=False, require_all_tables=False, venue=None)
        style_supplement_captions(SUPPLEMENT_OUT)
        tune_supplement_tables(SUPPLEMENT_OUT)

        convert(RESPONSE, RESPONSE_OUT, reference)
        postprocess(RESPONSE_OUT, RESPONSE_OUT, strict=False, require_all_tables=False, venue=None)

    counts = make_marked(CLEAN, MARKED)
    print(f"Built {CLEAN.name}")
    print(f"Built {MARKED.name}: {counts}")
    print(f"Built {SUPPLEMENT_OUT.name}")
    print(f"Built {RESPONSE_OUT.name}")


if __name__ == "__main__":
    main()
