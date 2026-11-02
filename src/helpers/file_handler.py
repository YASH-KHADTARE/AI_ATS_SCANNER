
# ============================================================
# FILE: src/helpers/file_handler.py
# PURPOSE: Input file handling and safe ZIP extraction
#          (pipeline stage 1)
#
#   validate_zip_file()     reject missing, empty or corrupt ZIPs
#   is_safe_zip_member()    block path traversal and symlinks
#   extract_resumes_zip()   extract supported resumes, skip the rest
#   discover_resumes()      list extracted resumes recursively
#   prepare_resumes()       extract + discover in one call
# ============================================================

import hashlib
import logging
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from config import (
    RESUMES_ZIP,
    SOURCE_RESUMES_DIR,
    MARKDOWN_RESUMES_DIR,
    RESULTS_DIR,
    SUPPORTED_RESUME_EXTENSIONS,
)


logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB read/write chunks


# ============================================================
# 1. CREATE NOTEBOOK OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():
    """Create the output directories used by the notebooks."""

    directories = [
        SOURCE_RESUMES_DIR,
        MARKDOWN_RESUMES_DIR,
        RESULTS_DIR,
    ]

    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create output directories: {exc}"
        ) from exc

    return directories


# ============================================================
# 2. VALIDATE RESUME ZIP FILE
# ============================================================

def validate_zip_file(zip_path=RESUMES_ZIP):
    """
    Validate the ZIP archive before extraction.

    Returns:
        Path: Validated ZIP file path.
    """

    zip_path = Path(zip_path)

    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Resume ZIP file not found: {zip_path}"
        )

    if zip_path.stat().st_size == 0:
        raise ValueError(
            f"Resume ZIP file is empty: {zip_path}"
        )

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(
            f"Invalid or corrupt ZIP archive: {zip_path}"
        )

    # Check every member's CRC before extracting anything.
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            bad_file = archive.testzip()

    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(
            f"Invalid or corrupt ZIP archive: {zip_path}: {exc}"
        ) from exc

    if bad_file is not None:
        raise ValueError(
            f"Corrupt file detected inside ZIP: {bad_file}"
        )

    logger.debug("ZIP validated: %s", zip_path)

    return zip_path


# ============================================================
# 3. CHECK WHETHER A ZIP MEMBER IS SAFE
# ============================================================

def is_safe_zip_member(member):
    """
    Reject absolute paths, path traversal, and symbolic links.
    """

    # ZIP paths normally use forward slashes, but also check
    # Windows-style paths to prevent unsafe extraction.
    posix_path = PurePosixPath(member.filename)
    windows_path = PureWindowsPath(member.filename)

    if posix_path.is_absolute() or windows_path.is_absolute():
        return False

    if windows_path.drive or windows_path.root:
        return False

    if ".." in posix_path.parts or ".." in windows_path.parts:
        return False

    # Reject Unix symbolic links.
    unix_mode = member.external_attr >> 16
    file_type = unix_mode & 0o170000

    if file_type == 0o120000:
        return False

    return True


# ============================================================
# 4. GENERATE A UNIQUE EXTRACTION PATH
# ============================================================

def get_unique_path(file_path):
    """
    Generate a unique path without overwriting an existing file.

    Example:
        resume.pdf
        resume__1.pdf
        resume__2.pdf
    """

    file_path = Path(file_path)

    if not file_path.exists():
        return file_path

    counter = 1

    while True:
        candidate_path = file_path.with_name(
            f"{file_path.stem}__{counter}{file_path.suffix}"
        )

        if not candidate_path.exists():
            return candidate_path

        counter += 1


# ============================================================
# 5. COMPARE A ZIP MEMBER WITH AN EXISTING FILE
# ============================================================

def is_same_content(archive, member, file_path):
    """Return True when a ZIP member matches an existing file."""

    file_path = Path(file_path)

    if (
        not file_path.is_file()
        or file_path.stat().st_size != member.file_size
    ):
        return False

    member_hash = hashlib.sha256()
    file_hash = hashlib.sha256()

    try:
        with archive.open(member, "r") as source:
            for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                member_hash.update(chunk)

        with file_path.open("rb") as existing:
            for chunk in iter(lambda: existing.read(CHUNK_SIZE), b""):
                file_hash.update(chunk)

    except OSError as exc:
        logger.warning(
            "Could not compare %s with %s: %s",
            member.filename, file_path, exc,
        )
        return False

    return member_hash.digest() == file_hash.digest()


# ============================================================
# 6. EXTRACT RESUME ZIP SAFELY
# ============================================================

def extract_resumes_zip(
    zip_path=RESUMES_ZIP,
    source_dir=SOURCE_RESUMES_DIR,
):
    """
    Safely extract supported resumes from the ZIP archive.

    Preserves nested directories and prevents different files
    from overwriting one another. Files already extracted with
    identical content are reused, so re-runs do not duplicate them.

    Returns:
        (list[Path], list[str]): Newly extracted resume paths and
        the names of ZIP members that were skipped.
    """

    # Step 1: validate the archive and prepare the target folder.
    zip_path = validate_zip_file(zip_path)

    extraction_root = Path(source_dir).resolve()

    try:
        extraction_root.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create extraction folder {extraction_root}: {exc}"
        ) from exc

    extracted_files = []
    unchanged_files = []
    skipped_files = []
    used_paths = set()

    with zipfile.ZipFile(zip_path, "r") as archive:

        for member in archive.infolist():

            if member.is_dir():
                continue

            # Step 2: skip unsupported formats (only PDF is processed).
            if (
                Path(member.filename).suffix.lower()
                not in SUPPORTED_RESUME_EXTENSIONS
            ):
                logger.info(
                    "Skipped unsupported file: %s", member.filename
                )
                skipped_files.append(member.filename)
                continue

            # Step 3: reject unsafe paths and symbolic links.
            if not is_safe_zip_member(member):
                logger.warning(
                    "Skipped unsafe ZIP member: %s", member.filename
                )
                skipped_files.append(member.filename)
                continue

            relative_path = Path(*PurePosixPath(member.filename).parts)

            destination = (extraction_root / relative_path).resolve()

            # Confirm the destination remains inside the
            # extraction directory.
            if not destination.is_relative_to(extraction_root):
                logger.warning(
                    "Skipped path outside extraction directory: %s",
                    member.filename,
                )
                skipped_files.append(member.filename)
                continue

            # Step 4: avoid overwriting; reuse identical files.
            normalized_path = str(destination).casefold()

            if (
                normalized_path not in used_paths
                and is_same_content(archive, member, destination)
            ):
                used_paths.add(normalized_path)
                unchanged_files.append(destination)
                continue

            if normalized_path in used_paths or destination.exists():
                destination = get_unique_path(destination)

                while str(destination).casefold() in used_paths:
                    destination = get_unique_path(destination)

            used_paths.add(str(destination).casefold())

            # Step 5: stream the member to disk.
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)

                with archive.open(member, "r") as source:
                    with destination.open("xb") as target:
                        for chunk in iter(
                            lambda: source.read(CHUNK_SIZE), b""
                        ):
                            target.write(chunk)

                extracted_files.append(destination)

            except Exception:
                # Remove an incomplete file, if one was created.
                if destination.exists():
                    try:
                        destination.unlink()
                    except OSError:
                        logger.warning(
                            "Could not remove partial file: %s",
                            destination,
                        )

                logger.exception(
                    "Failed to extract resume: %s", member.filename
                )
                skipped_files.append(member.filename)

    # Step 6: summary.
    logger.info("=" * 60)
    logger.info("RESUME ZIP EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info("ZIP file          : %s", zip_path)
    logger.info("Extracted resumes : %d", len(extracted_files))
    logger.info("Already extracted : %d", len(unchanged_files))
    logger.info("Skipped files     : %d", len(skipped_files))
    logger.info("Output folder     : %s", extraction_root)

    if not extracted_files and not unchanged_files:
        raise ValueError(
            "No PDF resumes were found in the ZIP archive."
        )

    return extracted_files, skipped_files


# ============================================================
# 7. DISCOVER RESUMES RECURSIVELY
# ============================================================

def discover_resumes(source_dir=SOURCE_RESUMES_DIR):
    """
    Discover supported resume files recursively.

    Returns:
        list[Path]: Sorted resume paths.
    """

    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Resume directory not found: {source_dir}"
        )

    resume_files = sorted(
        (
            path
            for path in source_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_RESUME_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )

    if not resume_files:
        raise ValueError(
            f"No resumes found in: {source_dir}"
        )

    logger.info("Resumes discovered: %d", len(resume_files))

    return resume_files


# ============================================================
# 8. PREPARE RESUMES FOR THE PIPELINE
# ============================================================

def prepare_resumes(
    zip_path=RESUMES_ZIP,
    source_dir=SOURCE_RESUMES_DIR,
):
    """
    Extract the ZIP into source_dir and discover its resumes.

    Returns:
        (list[Path], list[str]): Resume paths available for
        processing and the ZIP members that were skipped.
    """

    extracted_files, skipped_files = extract_resumes_zip(
        zip_path, source_dir
    )

    resume_files = discover_resumes(source_dir)

    logger.info("Newly extracted resumes: %d", len(extracted_files))
    logger.info("Total resumes available: %d", len(resume_files))

    return resume_files, skipped_files
