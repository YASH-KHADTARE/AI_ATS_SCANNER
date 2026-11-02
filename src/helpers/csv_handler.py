
# ============================================================
# FILE: src/helpers/csv_handler.py
# PURPOSE: CSV operations and checkpoint management
#
#   read_csv_file()            read with the project encoding
#   save_csv_file()            atomic write via a temp file
#   ensure_columns()           add missing columns
#   validate_candidate_ids()   candidate_id present and unique
#   upsert_candidate_records() merge records by identifier
#   save_checkpoint()          persist progress after each record
#   load_or_create_csv()       load an existing CSV or create it
# ============================================================

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

CSV_ENCODING = "utf-8-sig"


# ============================================================
# 1. READ CSV FILE
# ============================================================

def read_csv_file(file_path, **read_kwargs):
    """
    Read a CSV file using the project's standard encoding.

    Extra keyword arguments are passed to pandas.read_csv
    (for example dtype=...).
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    try:
        return pd.read_csv(
            file_path,
            encoding=CSV_ENCODING,
            **read_kwargs,
        )

    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"CSV file is empty: {file_path}"
        ) from exc

    except pd.errors.ParserError as exc:
        raise ValueError(
            f"CSV file could not be parsed: {file_path}: {exc}"
        ) from exc


# ============================================================
# 2. SAVE CSV FILE
# ============================================================

def save_csv_file(dataframe, file_path):
    """
    Save a DataFrame atomically.

    The temporary file is created in the same directory so
    os.replace() can replace the destination on the same drive.
    A failed write never leaves a half-written destination.
    """

    file_path = Path(file_path)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create directory {file_path.parent}: {exc}"
        ) from exc

    temp_path = None

    try:
        # Step 1: write to a temporary file next to the destination.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=CSV_ENCODING,
            newline="",
            suffix=".tmp",
            prefix=f"{file_path.stem}_",
            dir=file_path.parent,
            delete=False,
        ) as temp_file:

            temp_path = Path(temp_file.name)

            dataframe.to_csv(temp_file, index=False)

            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Step 2: atomically replace the destination.
        os.replace(temp_path, file_path)

    except Exception:
        # Remove the incomplete temporary file, if one was created.
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning(
                    "Could not remove temporary file: %s", temp_path
                )

        logger.exception("Failed to save CSV: %s", file_path)
        raise

    logger.debug("CSV saved: %s (%d rows)", file_path, len(dataframe))


# ============================================================
# 3. ENSURE REQUIRED COLUMNS EXIST
# ============================================================

def ensure_columns(dataframe, required_columns):
    """
    Add missing columns without removing or replacing existing
    columns or values.
    """

    for column in required_columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA

    return dataframe


# ============================================================
# 4. VALIDATE CANDIDATE IDENTIFIERS
# ============================================================

def validate_candidate_ids(dataframe):
    """Validate that candidate IDs exist, are non-blank and unique."""

    if "candidate_id" not in dataframe.columns:
        raise ValueError(
            "CSV must contain a candidate_id column."
        )

    candidate_ids = dataframe["candidate_id"]

    if candidate_ids.isna().any():
        raise ValueError(
            "Missing candidate IDs found in the CSV."
        )

    candidate_ids = candidate_ids.astype(str).str.strip()

    if candidate_ids.eq("").any():
        raise ValueError(
            "Blank candidate IDs found in the CSV."
        )

    if candidate_ids.duplicated().any():
        duplicates = candidate_ids[
            candidate_ids.duplicated(keep=False)
        ].tolist()

        raise ValueError(
            f"Duplicate candidate IDs found: {duplicates}"
        )

    return True


# ============================================================
# 5. UPSERT CANDIDATE RECORDS
# ============================================================

def upsert_candidate_records(
    existing_df,
    new_df,
    candidate_key="candidate_id",
):
    """
    Merge candidate records using a stable identifier.

    Existing records are updated by matching their identifier.
    New candidates are appended.

    Existing columns are preserved. New columns are added when
    necessary.
    """

    # Step 1: both frames must carry unique, non-missing identifiers.
    if candidate_key not in existing_df.columns:
        raise ValueError(
            f"Existing data is missing {candidate_key}."
        )

    if candidate_key not in new_df.columns:
        raise ValueError(
            f"New data is missing {candidate_key}."
        )

    if existing_df[candidate_key].isna().any():
        raise ValueError(
            f"Missing identifiers in existing data: {candidate_key}"
        )

    if new_df[candidate_key].isna().any():
        raise ValueError(
            f"Missing identifiers in new data: {candidate_key}"
        )

    if existing_df[candidate_key].duplicated().any():
        raise ValueError(
            "Existing data contains duplicate identifiers."
        )

    if new_df[candidate_key].duplicated().any():
        raise ValueError(
            "New data contains duplicate identifiers."
        )

    # Step 2: preserve the union of columns from both datasets.
    all_columns = list(existing_df.columns)

    for column in new_df.columns:
        if column not in all_columns:
            all_columns.append(column)

    existing = existing_df.reindex(columns=all_columns).copy()
    incoming = new_df.reindex(columns=all_columns).copy()

    existing = existing.set_index(candidate_key, drop=False)
    incoming = incoming.set_index(candidate_key, drop=False)

    # Step 3: update matching records and append new identifiers.
    existing.update(incoming)

    new_ids = incoming.index.difference(existing.index)

    if len(new_ids) > 0:
        existing = pd.concat([existing, incoming.loc[new_ids]])

    logger.debug(
        "Upsert: %d updated, %d appended.",
        len(incoming) - len(new_ids), len(new_ids),
    )

    return existing.reset_index(drop=True)


# ============================================================
# 6. SAVE CHECKPOINT
# ============================================================

def save_checkpoint(dataframe, file_path):
    """
    Persist the current processing state.

    Call after each successfully processed candidate and after
    recording a failure.
    """

    save_csv_file(dataframe=dataframe, file_path=file_path)


# ============================================================
# 7. LOAD OR CREATE CSV
# ============================================================

def load_or_create_csv(file_path, columns):
    """
    Load an existing CSV or create an empty one with the
    required columns.
    """

    file_path = Path(file_path)

    if file_path.exists():
        dataframe = read_csv_file(file_path)

        return ensure_columns(dataframe, columns)

    dataframe = pd.DataFrame(columns=columns)

    save_csv_file(dataframe, file_path)

    logger.info("Created new CSV: %s", file_path)

    return dataframe
