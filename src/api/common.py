
# ============================================================
# FILE: src/api/common.py
# PURPOSE: Helpers shared by the route modules (no routes here)
#
#   save_upload()           stream an UploadFile to disk
#   get_existing_job()      job_id -> (paths, status) or 404
#   require_completed_job() same, but 409 unless completed
# ============================================================

import logging
import shutil

from fastapi import HTTPException

from config import get_job_paths
from src.helpers.pipeline import read_status


logger = logging.getLogger(__name__)

EXCEL_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# ============================================================
# 1. SAVE THE UPLOADED FILE
# ============================================================

def save_upload(upload, destination):
    """Stream an uploaded file to disk."""

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as target:
            shutil.copyfileobj(upload.file, target)

    except OSError as exc:
        raise OSError(
            f"Could not save upload to {destination}: {exc}"
        ) from exc

    finally:
        try:
            upload.file.close()
        except Exception:
            pass

    logger.debug("Upload saved: %s", destination)


# ============================================================
# 2. RESOLVE AN EXISTING JOB
# ============================================================

def get_existing_job(job_id):
    """
    Resolve a job ID to its paths and status, or raise 404.

    Returns:
        (paths, status)
    """

    # Step 1: the ID must be well-formed (blocks path traversal).
    try:
        paths = get_job_paths(job_id)

    except ValueError:
        logger.warning("Rejected malformed job ID: %r", job_id)
        raise HTTPException(status_code=404, detail="Job not found.")

    # Step 2: the workspace must exist.
    if not paths["root"].is_dir():
        raise HTTPException(status_code=404, detail="Job not found.")

    # Step 3: the pipeline must have started at least once.
    try:
        status = read_status(paths)

    except (ValueError, OSError) as exc:
        logger.error("Unreadable status.json for %s: %s", job_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Job status file is unreadable.",
        )

    if status is None:
        raise HTTPException(
            status_code=404,
            detail="Job was created but never executed.",
        )

    return paths, status


def require_completed_job(job_id, resource="results"):
    """Resolve a job and raise 409 unless its pipeline completed."""

    paths, status = get_existing_job(job_id)

    if status.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job is {status.get('status')} ({status.get('stage')}); "
                f"{resource} are not available."
            ),
        )

    return paths, status
