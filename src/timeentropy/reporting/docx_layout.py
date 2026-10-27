"""Deterministic Word layout pass for the two submission DOCX files.

Pandoc emits a structurally correct document but leaves three defects that make the files
unfit to submit: every `![](path)` image gains a bare `Figure N` caption beside the authored
one, every table gets equal column widths regardless of content, and nothing is marked
keep-with-next, so captions orphan from their figures and tables split across pages.

Column widths are derived from what each column actually holds, including inline code and
rendered equations, which `Table.text` reports as empty. A table whose widest unbreakable
content still cannot fit the line has its type reduced, within a bound, rather than having
words split mid-letter.

This module repairs layout only. It never renumbers a caption: prose cross-references are
written against the Markdown sources, so a numbering defect is raised for the author to fix
at the source rather than silently rewritten here.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Sequence
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentObject
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

BARE_FIGURE = re.compile(r"^Figure\s+([1-9]\d*)$")
FULL_FIGURE = re.compile(r"^Figure\s+([1-9]\d*)\.\s+\S")
FULL_TABLE = re.compile(r"^Table\s+([1-9]\d*(?:[a-z])?)\.\s+\S")
EMU_PER_TWIP = 635

MIN_COLUMN_PERCENT = 4
HEADER_WEIGHT = 0.65
BODY_PERCENTILE = 0.9
BODY_WEIGHT_CAP = 60
# The venue sets table text at the body size with a bordered grid and centred cells.
TABLE_POINTS = 12.0
MIN_TABLE_POINTS = 7.0
TABLE_STYLE = "Table Grid"
TABLE_BORDER_EIGHTHS = 8
TABLE_CELL_MARGIN_DXA = 100
# Approximate advance of one Times character per point of type size, used to reserve enough
# width for a column's longest unbreakable unit so Word cannot split it mid-letter. Header
# words are largely capitals, which run wider than the average character, so this is
# deliberately generous: under-reserving breaks a word, over-reserving only shrinks the type.
CHAR_TWIPS_PER_POINT = 11.2
CELL_PADDING_TWIPS = 200
# Pandoc styles inline code with `VerbatimChar`, a monospace face whose characters are wider
# than the proportional body font, so a code identifier needs more room than its length.
CODE_STYLE = "VerbatimChar"
CODE_WIDTH_FACTOR = 1.3
# Rendered mathematics is wider than its plain-text characters: brackets, operators, and
# sub/superscripts all take space the concatenated `m:t` text does not account for.
MATH_WIDTH_FACTOR = 1.25
MAX_FLOOR_SHARE = 0.94
BREAK_OPPORTUNITY = re.compile(r"[\s/\-\u2010-\u2015]+")

# Explicit overrides for tables whose ideal layout is not the one `derive_ratios` computes
# from their content. Keys are normalized header rows and values are integer percentages
# summing to 100. This is deliberately empty: an override pins a layout to one revision of a
# table, so add one only when a specific table renders badly and stays fixed thereafter.
WIDTHS: dict[tuple[str, ...], tuple[int, ...]] = {}


def normalize(text: str) -> str:
    """Normalize layout-only variation while retaining semantically meaningful text."""

    return " ".join(
        text.replace("θ", "theta").replace("ε", "epsilon").replace("Δ", "delta").split()
    ).casefold()


def iter_blocks(document: DocumentObject):
    """Yield top-level paragraphs and tables in document order."""

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def set_on_off(parent, tag: str, enabled: bool = True) -> None:
    node = parent.find(qn(tag))
    if enabled and node is None:
        node = OxmlElement(tag)
        parent.append(node)
    elif not enabled and node is not None:
        parent.remove(node)


def remove_paragraph(paragraph: Paragraph) -> None:
    parent = paragraph._p.getparent()
    if parent is not None:
        parent.remove(paragraph._p)


def ensure_caption_styles(document: DocumentObject) -> None:
    styles = document.styles
    try:
        caption = next(style for style in styles if style.name == "Caption")
    except StopIteration:
        caption = styles.add_style("Caption", WD_STYLE_TYPE.PARAGRAPH)
        caption.base_style = next(style for style in styles if style.name == "Normal")
    # Typography comes from the venue's caption style; only pagination behaviour is set here.
    caption.paragraph_format.keep_together = True

    try:
        table_caption = next(style for style in styles if style.name == "Table Caption")
    except StopIteration:
        table_caption = styles.add_style("Table Caption", WD_STYLE_TYPE.PARAGRAPH)
        table_caption.base_style = caption
    table_caption.paragraph_format.keep_with_next = True
    table_caption.paragraph_format.keep_together = True


def clean_and_style_captions(document: DocumentObject) -> list[str]:
    """Remove only provable duplicate image-alt captions and style authored captions."""

    ensure_caption_styles(document)
    removed: list[str] = []
    blocks = list(iter_blocks(document))
    for index, block in enumerate(blocks):
        if not isinstance(block, Paragraph):
            continue
        text = block.text.strip()
        if FULL_TABLE.match(text):
            block.style = next(style for style in document.styles if style.name == "Table Caption")
            block.paragraph_format.keep_with_next = True
            block.paragraph_format.keep_together = True
            continue
        if FULL_FIGURE.match(text):
            # Style the authored caption whether or not pandoc also emitted a bare one, so
            # figure and table captions read the same way throughout the document.
            block.style = next(style for style in document.styles if style.name == "Caption")
            block.paragraph_format.keep_together = True
            block.paragraph_format.keep_with_next = False
            continue
        has_drawing = bool(block._p.xpath(".//w:drawing"))
        if not has_drawing:
            continue

        block.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        block.paragraph_format.keep_with_next = True
        block.paragraph_format.space_after = Pt(0)

        if index + 1 >= len(blocks) or not isinstance(blocks[index + 1], Paragraph):
            continue
        bare = blocks[index + 1]
        bare_match = BARE_FIGURE.match(bare.text.strip())
        if not bare_match:
            continue
        # Delete only when an authored full caption with the same number immediately follows.
        if index + 2 >= len(blocks) or not isinstance(blocks[index + 2], Paragraph):
            continue
        full = blocks[index + 2]
        full_match = FULL_FIGURE.match(full.text.strip())
        if not full_match or full_match.group(1) != bare_match.group(1):
            continue
        remove_paragraph(bare)
        removed.append(f"Figure {bare_match.group(1)}")
        full.style = next(style for style in document.styles if style.name == "Caption")
        full.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        full.paragraph_format.keep_together = True
        full.paragraph_format.keep_with_next = False
    return removed


FRONT_MATTER_STYLES = ("IROAuthors", "IROAffiliation", "IROMetadata", "IROMetadata", "IROMetadata")


def restyle_front_matter(document: DocumentObject, venue: Path | None) -> int:
    """Give the author, affiliation, and contact lines the venue's front-matter formatting.

    Pandoc has no notion of an author block in a plain Markdown body, so those lines arrive as
    ordinary paragraphs and pick up the body style's justification and first-line indent. The
    venue sets each of them differently, so they are restyled from the template by position:
    everything between the title and the first section heading.
    """

    if venue is None or not venue.is_file():
        return 0
    template = Document(venue)
    direct = representative_formatting(template)
    styles = {node.get(qn("w:styleId")): node for node in template.styles.element.findall(qn("w:style"))}

    paragraphs = list(document.paragraphs)
    heading = next(
        (
            index
            for index, paragraph in enumerate(paragraphs)
            if paragraph.style.name in {"Title", "Heading 1"} and paragraph.text.strip()
        ),
        None,
    )
    if heading is None:
        return 0
    block = []
    for paragraph in paragraphs[heading + 1 :]:
        if paragraph.style.name in {"Title"} or paragraph.style.name.startswith("Heading"):
            break
        if paragraph.text.strip():
            block.append(paragraph)

    restyled = 0
    for index, paragraph in enumerate(block):
        style_id = FRONT_MATTER_STYLES[min(index, len(FRONT_MATTER_STYLES) - 1)]
        if style_id not in styles:
            continue
        properties = paragraph._p.find(qn("w:pPr"))
        if properties is not None:
            paragraph._p.remove(properties)
        source = direct.get(style_id)
        if source is None:
            source = styles[style_id].find(qn("w:pPr"))
        if source is not None:
            copied = deepcopy(source)
            for reference in copied.findall(qn("w:pStyle")):
                copied.remove(reference)
            paragraph._p.insert(0, copied)
        run_properties = styles[style_id].find(qn("w:rPr"))
        if run_properties is not None:
            for run in paragraph.runs:
                existing = run._r.find(qn("w:rPr"))
                if existing is not None:
                    run._r.remove(existing)
                run._r.insert(0, deepcopy(run_properties))
        restyled += 1
    return restyled


def caption_labels(document: DocumentObject, pattern: re.Pattern[str]) -> list[str]:
    labels = []
    for paragraph in document.paragraphs:
        match = pattern.match(paragraph.text.strip())
        if match:
            labels.append(match.group(1))
    return labels


def validate_caption_order(document: DocumentObject, *, require_all_tables: bool) -> None:
    """Fail rather than silently renumbering captions and breaking prose references."""

    errors = []
    figures = caption_labels(document, FULL_FIGURE)
    expected_figures = [str(i) for i in range(1, len(figures) + 1)]
    if figures != expected_figures:
        errors.append(f"figure captions are not sequential: {figures}; expected {expected_figures}")

    tables = caption_labels(document, FULL_TABLE)
    expected_tables = [str(i) for i in range(1, len(tables) + 1)]
    if tables != expected_tables:
        errors.append(f"table captions are not sequential: {tables}; expected {expected_tables}")

    if require_all_tables:
        blocks = list(iter_blocks(document))
        missing = []
        for index, block in enumerate(blocks):
            if not isinstance(block, Table):
                continue
            previous = blocks[index - 1] if index else None
            if not isinstance(previous, Paragraph) or not FULL_TABLE.match(previous.text.strip()):
                missing.append(" | ".join(cell.text.strip() for cell in block.rows[0].cells))
        if missing:
            errors.append(
                "manuscript tables without an immediately preceding numbered caption: "
                + repr(missing)
            )
        if len(tables) != len(document.tables):
            errors.append(
                f"manuscript has {len(document.tables)} tables but {len(tables)} numbered captions"
            )

    if errors:
        raise ValueError("\n".join(errors))


def table_header(table: Table) -> tuple[str, ...]:
    return tuple(normalize(cell.text) for cell in table.rows[0].cells)


def set_val(parent, tag: str, value: str, attr: str = "w:val") -> None:
    node = parent.find(qn(tag))
    if node is None:
        node = OxmlElement(tag)
        parent.append(node)
    node.set(qn(attr), value)


def document_usable_twips(document: DocumentObject) -> int:
    """Derive usable line width from the reference DOCX section geometry.

    The supplied files currently have one usable width. Fail on mixed section widths until the
    caller resolves each table to its governing section; silently using the first section would
    be unsafe.
    """

    widths = {
        round((section.page_width - section.left_margin - section.right_margin) / EMU_PER_TWIP)
        for section in document.sections
    }
    if len(widths) != 1:
        raise ValueError(
            "document has multiple usable section widths; resolve each table to its section: "
            f"{sorted(widths)}"
        )
    usable_twips = widths.pop()
    if usable_twips <= 0:
        raise ValueError(f"invalid usable section width: {usable_twips} twips")
    return usable_twips


def set_table_widths(
    table: Table, ratios: tuple[int, ...], usable_twips: int, points: float = TABLE_POINTS
) -> None:
    if len(ratios) != len(table.columns) or sum(ratios) != 100:
        raise ValueError(f"invalid column ratios {ratios} for {len(table.columns)} columns")

    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    try:
        table.style = TABLE_STYLE
    except KeyError:
        pass
    table_properties = table._tbl.tblPr
    set_val(table_properties, "w:tblW", "5000")
    table_properties.find(qn("w:tblW")).set(qn("w:type"), "pct")
    set_val(table_properties, "w:tblLayout", "fixed", "w:type")

    # A ruled grid on every edge, as the venue's own tables use.
    borders = table_properties.find(qn("w:tblBorders"))
    if borders is not None:
        table_properties.remove(borders)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), str(TABLE_BORDER_EIGHTHS))
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), "000000")
        borders.append(node)
    table_properties.append(borders)

    margins = table_properties.find(qn("w:tblCellMar"))
    if margins is None:
        margins = OxmlElement("w:tblCellMar")
        table_properties.append(margins)
    for side in ("top", "left", "bottom", "right"):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(TABLE_CELL_MARGIN_DXA))
        node.set(qn("w:type"), "dxa")

    widths = [round(usable_twips * ratio / 100) for ratio in ratios]
    widths[-1] += usable_twips - sum(widths)
    grid = table._tbl.tblGrid
    for old in list(grid):
        grid.remove(old)
    for width in widths:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(width))
        grid.append(column)

    for row_index, row in enumerate(table.rows):
        row_properties = row._tr.get_or_add_trPr()
        set_on_off(row_properties, "w:cantSplit", True)
        set_on_off(row_properties, "w:tblHeader", row_index == 0)
        for cell, width in zip(row.cells, widths):
            properties = cell._tc.get_or_add_tcPr()
            cell_width = properties.find(qn("w:tcW"))
            if cell_width is None:
                cell_width = OxmlElement("w:tcW")
                properties.append(cell_width)
            cell_width.set(qn("w:w"), str(width))
            cell_width.set(qn("w:type"), "dxa")
            set_val(properties, "w:vAlign", "center")
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                paragraph.paragraph_format.first_line_indent = Pt(0)
                paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.keep_together = True
                for run in paragraph.runs:
                    run.font.size = Pt(points)
                    # The venue bolds the header row and leaves body rows regular.
                    if row_index == 0:
                        run.font.bold = True


def longest_unbreakable(text: str) -> int:
    """Length of the longest run Word cannot break, treating hyphens and slashes as breaks."""

    return max((len(part) for part in BREAK_OPPORTUNITY.split(text) if part), default=1)


def code_runs(cell) -> list[str]:
    """Text of each inline-code run in a cell, which needs a wider allowance per character."""

    runs = []
    for run in cell._tc.iter(qn("w:r")):
        properties = run.find(qn("w:rPr"))
        style = None if properties is None else properties.find(qn("w:rStyle"))
        if style is not None and style.get(qn("w:val")) == CODE_STYLE:
            runs.append("".join(node.text or "" for node in run.iter(qn("w:t"))))
    return runs


def math_runs(cell) -> list[str]:
    """Text of each OMML equation in a cell.

    Word renders an equation as one object and will not wrap it, so each equation is an
    unbreakable unit for width purposes. `cell.text` omits equation text entirely, which
    would otherwise make a formula column look empty and collapse to its header width.
    """

    return [
        "".join(node.text or "" for node in equation.iter(qn("m:t")))
        for equation in cell._tc.xpath(".//m:oMath")
    ]


def column_profile(table: Table) -> tuple[list[float], list[int]]:
    """Return each column's content weight and its longest unbreakable unit.

    Pandoc gives every column an equal share, which squeezes numeric columns and wraps prose
    columns into unreadable slivers. The weight drives the proportional split; the longest
    unbreakable unit becomes a hard floor, so a column holding `recovered_offline_peak` or a
    threshold formula keeps it on one line instead of splitting or overflowing it.
    """

    weights: list[float] = []
    tokens: list[int] = []
    for index in range(len(table.columns)):
        cells = [row.cells[index] for row in table.rows]
        contents = [cell.text.strip() for cell in cells]
        equations = [
            len(equation) * MATH_WIDTH_FACTOR
            for cell in cells
            for equation in math_runs(cell)
        ]
        code = [
            longest_unbreakable(text) * CODE_WIDTH_FACTOR
            for cell in cells
            for text in code_runs(cell)
        ]
        header, body = contents[0], contents[1:]
        lengths = sorted(len(text) for text in body) or [0]
        typical = max(
            lengths[round(BODY_PERCENTILE * (len(lengths) - 1))],
            max(equations, default=0),
        )
        # A header word must never break inside itself, so it is reserved whole rather than
        # only up to its first hyphen or slash: "Post/pre" reads badly as "Post/pr e".
        header_word = max((len(word) for word in header.split()), default=1)
        token = round(
            max(
                [longest_unbreakable(text) for text in contents]
                + [float(header_word)]
                + equations
                + code
            )
        )
        tokens.append(token)
        weights.append(
            float(max(token, min(BODY_WEIGHT_CAP, max(HEADER_WEIGHT * len(header), typical))))
        )
    return weights, tokens


def fitting_points(tokens: list[int], usable_twips: int) -> float:
    """Shrink a table's type only as far as its widest unbreakable content requires.

    A table whose longest tokens cannot fit the line at the standard size would otherwise
    have words split mid-letter. Reducing the size is preferable to that, and preferable to
    turning the page: it is bounded, so a table that still does not fit is reported instead
    of silently becoming unreadable.
    """

    required = sum(tokens) * CHAR_TWIPS_PER_POINT * TABLE_POINTS + len(tokens) * CELL_PADDING_TWIPS
    if required <= usable_twips:
        return TABLE_POINTS
    available = max(usable_twips - len(tokens) * CELL_PADDING_TWIPS, 1)
    return max(MIN_TABLE_POINTS, available / (sum(tokens) * CHAR_TWIPS_PER_POINT))


def derive_ratios(table: Table, usable_twips: int, points: float) -> tuple[int, ...]:
    """Turn column weights into integer percentages that sum to exactly 100.

    Each column first reserves the width of its longest unbreakable token; whatever remains is
    split in proportion to content. When the reserved widths would themselves overflow the
    line, they are scaled back together rather than starving one column.
    """

    columns = len(table.columns)
    weights, tokens = column_profile(table)
    char_twips = CHAR_TWIPS_PER_POINT * points
    floors = [
        max(MIN_COLUMN_PERCENT, 100 * (token * char_twips + CELL_PADDING_TWIPS) / usable_twips)
        for token in tokens
    ]
    if sum(floors) > MAX_FLOOR_SHARE * 100:
        scale = MAX_FLOOR_SHARE * 100 / sum(floors)
        floors = [floor * scale for floor in floors]

    spare = 100 - sum(floors)
    total = sum(weights)
    shares = [floor + spare * weight / total for floor, weight in zip(floors, weights)]
    ratios = [int(share) for share in shares]
    remainders = sorted(
        range(columns), key=lambda index: shares[index] - ratios[index], reverse=True
    )
    for index in remainders[: 100 - sum(ratios)]:
        ratios[index] += 1
    return tuple(ratios)


def format_tables(document: DocumentObject, *, strict_specs: bool = True) -> list[tuple[str, ...]]:
    """Apply an explicit width specification where one exists, otherwise derive one."""

    derived = []
    usable_twips = document_usable_twips(document)
    for table in document.tables:
        header = table_header(table)
        ratios = WIDTHS.get(header)
        points = TABLE_POINTS
        if ratios is None:
            _, tokens = column_profile(table)
            points = fitting_points(tokens, usable_twips)
            ratios = derive_ratios(table, usable_twips, points)
            derived.append(header)
        if strict_specs and min(ratios) < 1:
            raise ValueError(
                f"column widths {ratios} collapse a column for header " + " | ".join(header)
            )
        set_table_widths(table, ratios, usable_twips, points)
    return derived


def keep_final_response_table_together(document: DocumentObject) -> None:
    """Keep the short final limitations table, its heading, and intro from orphaning."""

    blocks = list(iter_blocks(document))
    for index, block in enumerate(blocks):
        if not isinstance(block, Table) or table_header(block) != ("request", "status", "reason"):
            continue
        # The final table fits on one page. Chain all rows; do not use this on long tables.
        for row_index, row in enumerate(block.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = row_index < len(block.rows) - 1
                    paragraph.paragraph_format.keep_together = True
        # Keep the immediately preceding explanatory paragraph and heading with the table.
        for prior in blocks[max(0, index - 2) : index]:
            if isinstance(prior, Paragraph):
                prior.paragraph_format.keep_with_next = True
                prior.paragraph_format.keep_together = True
        return


def highlight_runs(element) -> None:
    """Paint every run under an element with Word's yellow highlight.

    Equation runs are painted too. A display equation is an `m:r` element with no `w:r` of its
    own, so a text-only pass would leave equations white between marked paragraphs; the run
    properties inside `m:r` still take a `w:highlight`.
    """

    for tag in (qn("w:r"), qn("m:r")):
        for run in element.iter(tag):
            properties = run.find(qn("w:rPr"))
            if properties is None:
                properties = OxmlElement("w:rPr")
                run.insert(0, properties)
            highlight = properties.find(qn("w:highlight"))
            if highlight is None:
                highlight = OxmlElement("w:highlight")
                properties.append(highlight)
            highlight.set(qn("w:val"), "yellow")


def document_paragraphs(document: DocumentObject) -> list:
    """Every paragraph in document order, including those inside table cells."""

    return list(document.element.body.iter(qn("w:p")))


def paragraph_text(paragraph) -> str:
    """Readable text of a paragraph, including any equation it contains."""

    parts = [node.text or "" for node in paragraph.iter(qn("w:t"))]
    parts += [node.text or "" for node in paragraph.iter(qn("m:t"))]
    return normalize("".join(parts))


UNCHANGED_RATIO = 0.90


def changed_paragraphs(original: list[str], revised: list[str]) -> set[int]:
    """Indices of revised paragraphs that are new or rewritten relative to the original.

    Exact matches are found by sequence alignment. Within each replaced span the paragraphs are
    compared pairwise as well, so a paragraph that survived with only trivial edits is not
    reported as a revision.
    """

    changed: set[int] = set()
    matcher = SequenceMatcher(a=original, b=revised, autojunk=False)
    for tag, first_start, first_end, second_start, second_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            continue
        candidates = original[first_start:first_end]
        for index in range(second_start, second_end):
            text = revised[index]
            if not text:
                continue
            best = max(
                (SequenceMatcher(a=text, b=other, autojunk=False).ratio() for other in candidates),
                default=0.0,
            )
            if best < UNCHANGED_RATIO:
                changed.add(index)
    return changed


def highlight_revisions(
    clean_path: Path,
    marked_path: Path,
    *,
    original_path: Path,
    tables: set[int],
    figures: set[int],
    note: str,
) -> dict[str, int]:
    """Write a yellow-marked copy showing what changed against the submitted manuscript.

    Every paragraph that is new or rewritten relative to the submitted version is highlighted,
    which is what a reader comparing the two documents needs. The tables and figures the
    response letter names as answering a numbered comment are highlighted as well, so that a
    reviewer following a comment number always lands on marked content.
    """

    original = Document(original_path)
    document = Document(clean_path)
    original_text = [paragraph_text(node) for node in document_paragraphs(original)]
    revised_nodes = document_paragraphs(document)
    revised_text = [paragraph_text(node) for node in revised_nodes]

    changed = changed_paragraphs(original_text, revised_text)
    for index in changed:
        highlight_runs(revised_nodes[index])

    blocks = list(iter_blocks(document))
    marked_tables, marked_figures = 0, 0
    for index, block in enumerate(blocks):
        if not isinstance(block, Paragraph):
            continue
        text = block.text.strip()
        table_match = FULL_TABLE.match(text)
        if table_match and table_match.group(1).isdigit() and int(table_match.group(1)) in tables:
            highlight_runs(block._p)
            following = blocks[index + 1] if index + 1 < len(blocks) else None
            if isinstance(following, Table):
                highlight_runs(following._tbl)
            marked_tables += 1
            continue
        figure_match = FULL_FIGURE.match(text)
        if figure_match and int(figure_match.group(1)) in figures:
            highlight_runs(block._p)
            marked_figures += 1

    heading = next(
        (block for block in blocks if isinstance(block, Paragraph) and block.text.strip()), None
    )
    if heading is not None:
        highlight_runs(heading.insert_paragraph_before(note)._p)

    document.save(marked_path)
    return {
        "changed_paragraphs": len(changed),
        "total_paragraphs": sum(1 for text in revised_text if text),
        "tables": marked_tables,
        "figures": marked_figures,
    }


def postprocess(
    source: Path,
    output: Path,
    *,
    strict: bool,
    require_all_tables: bool,
    venue: Path | None = None,
) -> tuple[list[str], list[tuple[str, ...]]]:
    document = Document(source)
    restyle_front_matter(document, venue)
    removed = clean_and_style_captions(document)
    unknown = format_tables(document, strict_specs=strict)
    keep_final_response_table_together(document)
    if strict:
        validate_caption_order(document, require_all_tables=require_all_tables)
    document.save(output)
    return removed, unknown




# The venue's template defines its own paragraph styles rather than restyling Word's built-ins,
# so pandoc's style names are mapped onto the venue's. Every mapped style takes the venue's
# fonts, sizes, weights, colours, and spacing; unmapped pandoc styles keep their own formatting
# but inherit the venue's document defaults.
STYLE_MAP = {
    "Title": "IROTitle",
    "Author": "IROAuthors",
    "Subtitle": "IROAffiliation",
    "Date": "IROMetadata",
    "AbstractTitle": "IROAbstractHeading",
    "Abstract": "IROAbstractBody",
    "Normal": "IROBody",
    "BodyText": "IROBody",
    "FirstParagraph": "IROBody",
    "Compact": "IROTableText",
    "BlockText": "IROBody",
    "Heading1": "IROTitle",
    "Heading2": "IROMainHeading",
    "Heading3": "IROSubheading",
    "Heading4": "IROSubheading",
    "Heading5": "IROSubheading",
    "Heading6": "IROSubheading",
    "Caption": "IROCaption",
    "ImageCaption": "IROCaption",
    "TableCaption": "IROCaption",
    "Bibliography": "IROReference",
    "FootnoteText": "IROMetadata",
}
# Parts copied wholesale so theme-resolved fonts and colours match the venue's file, and so
# Word opens the result in its own compatibility mode rather than falling back to
# Compatibility Mode, which changes justification, table sizing, and hyphenation.
COPIED_PARTS = ("word/theme/theme1.xml", "word/fontTable.xml", "word/settings.xml")


def style_elements(document: DocumentObject) -> dict[str, object]:
    """Map styleId to its `w:style` element."""

    return {
        node.get(qn("w:styleId")): node
        for node in document.styles.element.findall(qn("w:style"))
        if node.get(qn("w:styleId"))
    }


def representative_formatting(document: DocumentObject) -> dict[str, object]:
    """The direct paragraph formatting each style is actually used with in the venue's file.

    The template sets alignment, indent, and spacing on the paragraphs rather than in the style
    definitions, so copying style definitions alone loses the justification, first-line indent,
    and heading spacing that give the document its look.
    """

    seen: dict[str, object] = {}
    for paragraph in document.paragraphs:
        if not paragraph.text.strip():
            continue
        style_id = paragraph.style.style_id
        if style_id in seen:
            continue
        properties = paragraph._p.find(qn("w:pPr"))
        if properties is not None:
            copied = deepcopy(properties)
            for reference in copied.findall(qn("w:pStyle")):
                copied.remove(reference)
            seen[style_id] = copied
    return seen


def copy_formatting(source, target, direct=None) -> None:
    """Replace a style's paragraph and run formatting with the source style's."""

    for tag in (qn("w:pPr"), qn("w:rPr")):
        existing = target.find(tag)
        if existing is not None:
            target.remove(existing)
        incoming = direct if tag == qn("w:pPr") and direct is not None else source.find(tag)
        if incoming is not None:
            copied = deepcopy(incoming)
            # A mapped style must not drag the venue style's own inheritance along with it.
            for inherited in copied.findall(qn("w:pStyle")):
                copied.remove(inherited)
            target.append(copied)


def replace_parts(package: Path, template: Path, parts: Sequence[str]) -> None:
    """Swap named parts of a DOCX with the template's, leaving every other part untouched."""

    with zipfile.ZipFile(template) as source:
        available = {name: source.read(name) for name in parts if name in source.namelist()}
    if not available:
        return
    rebuilt = package.with_name(package.name + ".rebuilt")
    with zipfile.ZipFile(package) as old, zipfile.ZipFile(
        rebuilt, "w", zipfile.ZIP_DEFLATED
    ) as new:
        for item in old.infolist():
            new.writestr(item, available.get(item.filename, old.read(item.filename)))
    rebuilt.replace(package)


def build_reference(source: Path, output: Path, template: Path | None = None) -> Path:
    """Give pandoc's own reference document the venue's page setup and typography.

    Pandoc needs its own named styles (`Compact`, `Body Text`, `Image Caption`, the table
    styles) to lay a document out at all, so the venue's file cannot be handed to pandoc as a
    reference directly: its tables collapse into a single stacked column. This keeps pandoc's
    style inventory and pours the venue's formatting into it, style by style, so the rendered
    result matches the submitted manuscript.
    """

    document = Document(source)

    if template is not None and template.is_file():
        venue = Document(template)
        venue_section = venue.sections[0]
        section = document.sections[0]
        for attribute in (
            "page_width",
            "page_height",
            "left_margin",
            "right_margin",
            "top_margin",
            "bottom_margin",
            "header_distance",
            "footer_distance",
            "orientation",
        ):
            value = getattr(venue_section, attribute, None)
            if value is not None:
                setattr(section, attribute, value)

        venue_defaults = venue.styles.element.find(qn("w:docDefaults"))
        if venue_defaults is not None:
            current = document.styles.element.find(qn("w:docDefaults"))
            if current is not None:
                document.styles.element.remove(current)
            document.styles.element.insert(0, deepcopy(venue_defaults))

        venue_styles = style_elements(venue)
        pandoc_styles = style_elements(document)
        direct = representative_formatting(venue)
        for pandoc_id, venue_id in STYLE_MAP.items():
            if pandoc_id in pandoc_styles and venue_id in venue_styles:
                copy_formatting(
                    venue_styles[venue_id], pandoc_styles[pandoc_id], direct.get(venue_id)
                )

    document.save(output)
    if template is not None and template.is_file():
        replace_parts(output, template, COPIED_PARTS)
    return output
