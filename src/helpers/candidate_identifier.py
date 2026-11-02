
# ============================================================
# FILE: src/helpers/candidate_identifier.py
# PURPOSE: Generate and maintain stable candidate IDs
#          (pipeline stage 2)
#
# Every extracted resume gets a CAND_000001-style ID recorded in
# candidate_manifest.csv together with its relative path and
# SHA-256 fingerprint. Re-running a job keeps existing IDs.
# ============================================================

import hashlib
import logging
import re
from pathlib import Path

import pandas as pd

from config import (
    SOURCE_RESUMES_DIR,
    CANDIDATE_MANIFEST_PATH,
)
from src.helpers.csv_handler import (
    read_csv_file,
    save_csv_file,
    validate_candidate_ids,
)


logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB read chunks


# ============================================================
# 1. MANIFEST COLUMNS
# ============================================================

MANIFEST_COLUMNS = [
    "candidate_id",
    "source_relative_path",
    "source_sha256",
    "file_size_bytes",
]


# ============================================================
# 2. CALCULATE FILE CONTENT FINGERPRINT
# ============================================================

def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a resume file."""

    file_path = Path(file_path)

    sha256 = hashlib.sha256()

    try:
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
                sha256.update(chunk)

    except OSError as exc:
        raise OSError(
            f"Could not read resume for hashing: {file_path}: {exc}"
        ) from exc

    return sha256.hexdigest()


# ============================================================
# 3. GENERATE THE NEXT CANDIDATE ID
# ============================================================

def generate_candidate_id(existing_ids):
    """Generate the next sequential candidate ID (CAND_000001, ...)."""

    highest_number = 0

    for candidate_id in existing_ids:
        match = re.fullmatch(r"CAND_(\d+)", str(candidate_id))

        if match:
            highest_number = max(highest_number, int(match.group(1)))

    return f"CAND_{highest_number + 1:06d}"


# ============================================================
# 4. LOAD THE EXISTING CANDIDATE MANIFEST
# ============================================================

def load_candidate_manifest(manifest_path=CANDIDATE_MANIFEST_PATH):
    """Load the persistent manifest or create an empty one."""

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        logger.debug("No manifest yet; starting a new one.")
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    manifest = read_csv_file(manifest_path)

    missing_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            "Candidate manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if manifest["source_relative_path"].duplicated().any():
        raise ValueError(
            "Duplicate source paths found in candidate manifest."
        )

    validate_candidate_ids(manifest)

    logger.debug("Manifest loaded: %d records.", len(manifest))

    return manifest[MANIFEST_COLUMNS].copy()


# ============================================================
# 5. IDENTIFY A RESUME
# ============================================================

def identify_resume(resume_path, manifest, source_dir=SOURCE_RESUMES_DIR):
    """
    Return the stable candidate ID for a resume.

    The relative source path identifies the resume record.
    The SHA-256 fingerprint detects changes to its content.

    Returns:
        (candidate_id, manifest, source_hash)
    """

    resume_path = Path(resume_path).resolve()
    source_root = Path(source_dir).resolve()

    # Step 1: the resume must live inside the source folder.
    try:
        relative_path = resume_path.relative_to(source_root).as_posix()

    except ValueError as exc:
        raise ValueError(
            f"Resume is outside the source directory: {resume_path}"
        ) from exc

    if not resume_path.is_file():
        raise FileNotFoundError(
            f"Resume file not found: {resume_path}"
        )

    # Step 2: fingerprint the file.
    source_hash = calculate_file_hash(resume_path)
    file_size = resume_path.stat().st_size

    # Step 3: look for an existing record by normalized path.
    normalized_path = relative_path.casefold()

    matches = manifest[
        manifest["source_relative_path"].astype(str).str.casefold()
        == normalized_path
    ]

    if len(matches) > 1:
        raise ValueError(
            f"Multiple manifest entries found for {relative_path}"
        )

    if not matches.empty:
        row_index = matches.index[0]
        candidate_id = manifest.at[row_index, "candidate_id"]

        if manifest.at[row_index, "source_sha256"] != source_hash:
            logger.info(
                "Resume content changed for %s (%s).",
                candidate_id, relative_path,
            )

        # Preserve the ID while updating the current file fingerprint.
        manifest.at[row_index, "source_sha256"] = source_hash
        manifest.at[row_index, "file_size_bytes"] = file_size

        return candidate_id, manifest, source_hash

    # Step 4: allocate a new ID for a new source path.
    candidate_id = generate_candidate_id(manifest["candidate_id"].tolist())

    new_record = {
        "candidate_id": candidate_id,
        "source_relative_path": relative_path,
        "source_sha256": source_hash,
        "file_size_bytes": file_size,
    }

    manifest = pd.concat(
        [manifest, pd.DataFrame([new_record])],
        ignore_index=True,
    )

    return candidate_id, manifest, source_hash


# ============================================================
# 6. IDENTIFY ALL RESUMES
# ============================================================

def identify_candidates(
    resume_paths,
    source_dir=SOURCE_RESUMES_DIR,
    manifest_path=CANDIDATE_MANIFEST_PATH,
):
    """
    Assign stable IDs to all discovered resumes.

    The manifest is saved after each new or updated record so IDs
    survive interruptions.

    Returns:
        list[dict]: Candidate IDs and source file information.
    """

    manifest = load_candidate_manifest(manifest_path)

    source_root = Path(source_dir).resolve()

    candidates = []

    for resume_path in resume_paths:

        resume_path = Path(resume_path)

        candidate_id, manifest, source_hash = identify_resume(
            resume_path,
            manifest,
            source_dir,
        )

        # Persist progress so IDs survive interruptions.
        save_csv_file(manifest[MANIFEST_COLUMNS], manifest_path)

        candidates.append({
            "candidate_id": candidate_id,
            "file_name": resume_path.name,
            "source_path": str(resume_path),
            "source_relative_path": (
                resume_path.resolve().relative_to(source_root).as_posix()
            ),
            "source_sha256": source_hash,
        })

        logger.info("Identified: %s | %s", candidate_id, resume_path.name)

    logger.info(
        "Candidate identification complete: %d resumes, %d manifest records.",
        len(candidates), len(manifest),
    )

    return candidates
