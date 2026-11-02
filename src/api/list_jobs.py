
# ============================================================
# FILE: src/api/list_jobs.py
# PURPOSE: GET /jobs
#
# List every uploaded and executed job with its name (position),
# job ID and status, newest first. Folders whose pipeline never
# started, or whose status.json is unreadable, are skipped.
# ============================================================

import logging

from fastapi import APIRouter, HTTPException

from config import JOBS_DIR, get_job_paths
from src.helpers.pipeline import read_status


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ============================================================
# 1. HELPERS
# ============================================================

def summarize_job(job_dir):
    """
    Build the list entry for one job folder.

    Returns:
        dict | None: None when the folder is not a usable job.
    """

    try:
        paths = get_job_paths(job_dir.name)

    except ValueError:
        logger.debug("Ignoring non-job folder: %s", job_dir.name)
        return None

    try:
        status = read_status(paths)

    except (ValueError, OSError) as exc:
        logger.warning(
            "Skipping job %s: unreadable status.json (%s)",
            job_dir.name, exc,
        )
        return None

    # Skip folders whose pipeline never started.
    if status is None:
        return None

    return {
        "job_id": job_dir.name,
        "job_position": status.get("job_position"),
        "status": status.get("status"),
        "created_at": status.get("created_at"),
        "updated_at": status.get("updated_at"),
    }


# ============================================================
# 2. ROUTE
# ============================================================

@router.get(
    "",
    summary="List all uploaded and executed jobs",
)
def list_jobs():
    """Return every job with its name (position), ID and status."""

    if not JOBS_DIR.is_dir():
        return []

    try:
        job_dirs = sorted(
            (path for path in JOBS_DIR.iterdir() if path.is_dir()),
            reverse=True,
        )

    except OSError as exc:
        logger.exception("Could not list jobs folder %s", JOBS_DIR)
        raise HTTPException(
            status_code=500,
            detail=f"Could not read the jobs folder: {exc}",
        )

    jobs = []

    for job_dir in job_dirs:
        entry = summarize_job(job_dir)

        if entry is not None:
            jobs.append(entry)

    logger.info("Listed %d job(s).", len(jobs))

    return jobs
