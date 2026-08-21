#!/usr/bin/env python3
"""Build the submission .docx files from the Markdown sources.

The response document is relabelled into the conference's Reviewer Concern / Author Response
form on the way through. Bare "---" horizontal rules are stripped: pandoc silently drops
subsequent content, including whole tables, when they appear in these documents, so the
sources avoid them and this script asserts they are absent.
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

WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
PANDOC_FORMAT = "markdown+pipe_tables-yaml_metadata_block"


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


def convert(markdown: str, output: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(markdown)
        source = Path(handle.name)
    try:
        subprocess.run(
            ["pandoc", str(source), f"--from={PANDOC_FORMAT}", "--to=docx", "-o", str(output)],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=Path("docs/revised_manuscript.md"))
    parser.add_argument("--response", type=Path, default=Path("docs/reviewer_response.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    arguments = parser.parse_args()

    failures = []
    for source in (arguments.manuscript, arguments.response):
        text = source.read_text()
        rules = [i + 1 for i, line in enumerate(text.split("\n")) if line == "---"]
        if rules:
            failures.append(f"{source} contains bare '---' rules at lines {rules}")
        expected = text.count("\n|---")
        markdown = relabel_response(text) if source == arguments.response else text
        output = arguments.output_dir / (
            "Response_to_Reviewers.docx"
            if source == arguments.response
            else "Time_Entropy_Dysarthric_Speech_Classification_REVISED.docx"
        )
        convert(markdown, output)
        tables, characters = describe(output)
        status = "ok" if tables == expected else "TABLE LOSS"
        print(f"[{status:>10}] {output.name}: {tables}/{expected} tables, {characters:,} characters")
        if tables != expected:
            failures.append(f"{output.name} rendered {tables} of {expected} tables")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit("submission documents did not build cleanly")


if __name__ == "__main__":
    main()
