#!/usr/bin/env python3
"""Build only the supplement and response files for the compact validation addendum."""

from __future__ import annotations

import tempfile
from pathlib import Path

from build_minimal_revision_documents import (
    ORIGINAL,
    RESPONSE,
    SUPPLEMENT,
    convert,
    postprocess,
    prepare_reference,
    style_supplement_captions,
    tune_supplement_tables,
)
from build_strict_minimal_revision import build_response

ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT_OUT = ROOT / "Supplementary_Material_MINIMAL_REVISION_VALIDATION_ADDED.docx"
RESPONSE_OUT = ROOT / "Response_to_Reviewers_STRICT_MINIMAL_REVISION_VALIDATION_ADDED.docx"


def main() -> None:
    if not ORIGINAL.is_file():
        raise SystemExit(f"Missing original manuscript: {ORIGINAL}")

    temporary_root = ROOT / "tmp/validation_additions"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
        workspace = Path(temporary)
        reference = prepare_reference(workspace)

        convert(SUPPLEMENT, SUPPLEMENT_OUT, reference)
        postprocess(
            SUPPLEMENT_OUT,
            SUPPLEMENT_OUT,
            strict=False,
            require_all_tables=False,
            venue=None,
        )
        style_supplement_captions(SUPPLEMENT_OUT)
        tune_supplement_tables(SUPPLEMENT_OUT)

        response_source = workspace / "response-minimal.docx"
        convert(RESPONSE, response_source, reference)
        postprocess(
            response_source,
            response_source,
            strict=False,
            require_all_tables=False,
            venue=None,
        )
        build_response(response_source, RESPONSE_OUT)

    print(SUPPLEMENT_OUT)
    print(RESPONSE_OUT)


if __name__ == "__main__":
    main()
