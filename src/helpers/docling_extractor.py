
# ============================================================
# FILE: src/helpers/docling_extractor.py
# PURPOSE: Convert resumes into Markdown using Docling
#          (pipeline stage 3; 01_data_extraction.ipynb, cell 2)
#
# Each PDF becomes <resume stem>.md in the output folder. Table
# structure is preserved so the LLM sees the original layout.
# A conversion failure is recorded per file and does not stop
# the remaining resumes.
# ============================================================

import logging
from pathlib import Path

import pandas as pd
from docling.document_converter import DocumentConverter

from config import (
    MARKDOWN_RESUMES_DIR,
    SUPPORTED_RESUME_EXTENSIONS,
)


logger = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "file_name",
    "status",
    "character_count",
    "extracted_text",
]


# ============================================================
# 1. CREATE THE CONVERTER
# ============================================================

def create_document_converter():
    """Create a Docling converter (loads the layout models)."""

    try:
        return DocumentConverter()

    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize the Docling converter: {exc}"
        ) from exc


# ============================================================
# 2. CONVERT ONE RESUME
# ============================================================

def convert_resume_to_markdown(file_path, output_folder, converter):
    """
    Convert a single resume to Markdown and save it.

    Returns:
        (output_path, markdown_text)
    """

    result = converter.convert(str(file_path))

    # Export document while preserving table structure
    markdown_text = result.document.export_to_markdown()

    output_path = Path(output_folder) / f"{Path(file_path).stem}.md"

    try:
        output_path.write_text(markdown_text, encoding="utf-8")

    except OSError as exc:
        raise OSError(
            f"Could not write Markdown file {output_path}: {exc}"
        ) from exc

    return output_path, markdown_text


# ============================================================
# 3. CONVERT ALL RESUMES
# ============================================================

def extract_resumes_to_markdown(
    resume_paths,
    output_folder=MARKDOWN_RESUMES_DIR,
    converter=None,
):
    """
    Convert each resume into Markdown, preserving table structure.

    Returns:
        pandas.DataFrame with file_name, status, character_count
        and extracted_text for every processed resume.
    """

    output_folder = Path(output_folder)

    try:
        output_folder.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create Markdown folder {output_folder}: {exc}"
        ) from exc

    if converter is None:
        converter = create_document_converter()

    results = []

    total = len(resume_paths)

    for index, file_path in enumerate(resume_paths, start=1):
        file_path = Path(file_path)

        # Step 1: skip anything that is not a supported resume file.
        if not file_path.is_file():
            logger.warning("Resume not found, skipped: %s", file_path)
            continue

        if file_path.suffix.lower() not in SUPPORTED_RESUME_EXTENSIONS:
            logger.info("Unsupported format, skipped: %s", file_path.name)
            continue

        logger.info("[%d/%d] Converting: %s", index, total, file_path.name)

        # Step 2: convert; a failure is recorded, not raised.
        try:
            _, markdown_text = convert_resume_to_markdown(
                file_path, output_folder, converter
            )

            results.append({
                "file_name": file_path.name,
                "status": "Success",
                "character_count": len(markdown_text),
                "extracted_text": markdown_text,
            })

            logger.info(
                "Processed: %s (%d characters)",
                file_path.name, len(markdown_text),
            )

        except Exception as exc:
            logger.exception(
                "Markdown extraction failed for %s.", file_path.name
            )

            results.append({
                "file_name": file_path.name,
                "status": f"Failed: {exc}",
                "character_count": 0,
                "extracted_text": "",
            })

    # Step 3: summary.
    df = pd.DataFrame(results, columns=RESULT_COLUMNS)

    success_count = int(df["status"].eq("Success").sum())

    logger.info(
        "Markdown conversion complete: %d processed, %d succeeded, %d failed.",
        len(df), success_count, len(df) - success_count,
    )

    return df
