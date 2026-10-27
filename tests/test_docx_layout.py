from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, Twips

from timeentropy.reporting.docx_layout import (
    MIN_TABLE_POINTS,
    TABLE_POINTS,
    build_reference,
    derive_ratios,
    fitting_points,
    format_tables,
    longest_unbreakable,
    validate_caption_order,
)

# A4 with one-inch margins, the geometry `build_reference` writes into the reference DOCX.
USABLE_TWIPS = 11906 - 2 * 1440


def _document_with_table(rows: list[list[str]]) -> Document:
    document = Document()
    section = document.sections[0]
    section.page_width = Twips(11906)
    section.page_height = Twips(16838)
    section.left_margin = Twips(1440)
    section.right_margin = Twips(1440)
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    for row, values in zip(table.rows, rows):
        for cell, value in zip(row.cells, values):
            cell.text = value
    return document


def test_longest_unbreakable_treats_hyphens_and_slashes_as_break_points() -> None:
    assert longest_unbreakable("recovered_offline_peak") == 22
    assert longest_unbreakable("speaker-independent") == len("independent")
    assert longest_unbreakable("control / dysarthric") == len("dysarthric")
    assert longest_unbreakable("") == 1


def test_derive_ratios_sum_to_one_hundred_and_favour_the_wider_column() -> None:
    document = _document_with_table(
        [
            ["Exact operation", "n"],
            ["Conv2d 1 to 24, 3x3, padding 1; BatchNorm2d; SiLU; MaxPool 2x1", "264"],
            ["Mean over 64 remaining frequency bins; transpose", "0"],
        ]
    )
    ratios = derive_ratios(document.tables[0], USABLE_TWIPS, TABLE_POINTS)

    assert sum(ratios) == 100
    assert ratios[0] > ratios[1]
    assert min(ratios) >= 1


def test_fitting_points_shrinks_only_when_content_cannot_fit() -> None:
    assert fitting_points([8, 6, 6], USABLE_TWIPS) == TABLE_POINTS

    crowded = fitting_points([15] * 8, USABLE_TWIPS)
    assert MIN_TABLE_POINTS <= crowded < TABLE_POINTS


def test_fitting_points_never_falls_below_the_readable_floor() -> None:
    assert fitting_points([40] * 12, USABLE_TWIPS) == MIN_TABLE_POINTS


def test_format_tables_replaces_pandocs_equal_column_default() -> None:
    document = _document_with_table(
        [
            ["Speaker protocol", "n"],
            ["Speaker-independent stratified 10-fold cross-validation", "5"],
        ]
    )
    derived = format_tables(document, strict_specs=True)

    assert derived == [("speaker protocol", "n")]
    widths = [
        int(column.get(qn("w:w"))) for column in document.tables[0]._tbl.tblGrid
    ]
    assert sum(widths) == USABLE_TWIPS
    assert widths[0] > widths[1]

    first_row = document.tables[0].rows[0]
    assert first_row._tr.trPr.find(qn("w:tblHeader")) is not None
    assert first_row._tr.trPr.find(qn("w:cantSplit")) is not None


def test_validate_caption_order_reports_a_gap_rather_than_renumbering() -> None:
    document = Document()
    document.add_paragraph("Table 1. First.")
    document.add_paragraph("Table 3. Third.")

    with pytest.raises(ValueError, match="table captions are not sequential"):
        validate_caption_order(document, require_all_tables=False)


def test_validate_caption_order_requires_every_manuscript_table_to_be_captioned() -> None:
    document = _document_with_table([["Mode", "Threshold"], ["causal", "0.003"]])

    with pytest.raises(ValueError, match="without an immediately preceding numbered caption"):
        validate_caption_order(document, require_all_tables=True)


def _venue_template(path: Path) -> Path:
    """A stand-in for the journal's file: its own named styles, used on real paragraphs."""

    venue = Document()
    section = venue.sections[0]
    section.page_width = Twips(11906)
    section.page_height = Twips(16838)
    section.top_margin = Twips(1503)

    body = venue.styles.add_style("IRO Body", WD_STYLE_TYPE.PARAGRAPH)
    body.font.name = "Times New Roman"
    body.font.size = Pt(12)
    title = venue.styles.add_style("IRO Title", WD_STYLE_TYPE.PARAGRAPH)
    title.font.size = Pt(24)
    title.font.bold = True

    written = venue.add_paragraph("Paper title", style=title)
    written.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    written = venue.add_paragraph("Body sentence long enough to matter.", style=body)
    written.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    written.paragraph_format.first_line_indent = Twips(720)
    venue.save(path)
    return path


def test_build_reference_copies_the_venue_page_setup_and_typography(tmp_path: Path) -> None:
    template = _venue_template(tmp_path / "venue.docx")
    source = tmp_path / "source.docx"
    Document().save(source)

    reference = Document(build_reference(source, tmp_path / "reference.docx", template))
    built = reference.sections[0]

    assert built.page_width == Twips(11906)
    assert built.page_height == Twips(16838)
    assert built.top_margin == Twips(1503)

    body = reference.styles["Body Text"]
    assert body.font.name == "Times New Roman"
    assert body.font.size == Pt(12)
    heading = reference.styles["Heading 1"]
    assert heading.font.size == Pt(24)
    assert heading.font.bold is True


def test_build_reference_carries_the_venue_paragraph_layout(tmp_path: Path) -> None:
    """Alignment and indent live on the venue's paragraphs, not its style definitions."""

    template = _venue_template(tmp_path / "venue.docx")
    source = tmp_path / "source.docx"
    Document().save(source)

    reference = Document(build_reference(source, tmp_path / "reference.docx", template))

    assert reference.styles["Body Text"].paragraph_format.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY
    assert reference.styles["Body Text"].paragraph_format.first_line_indent == Twips(720)
    assert reference.styles["Heading 1"].paragraph_format.alignment == WD_ALIGN_PARAGRAPH.CENTER


def test_build_reference_without_a_template_leaves_pandoc_defaults(tmp_path: Path) -> None:
    source = tmp_path / "source.docx"
    Document().save(source)

    assert build_reference(source, tmp_path / "reference.docx").is_file()
