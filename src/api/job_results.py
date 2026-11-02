
# ============================================================
# FILE: src/api/job_results.py
# PURPOSE: GET /jobs/{job_id}/results
#
# Return the ranked candidates for one completed job.
#   404  unknown job / never executed / job.json missing
#   409  job not completed (running or failed)
# ============================================================

import logging

from fastapi import APIRouter, HTTPException

from src.helpers.pipeline import (
    load_job_inputs_file,
    load_ranked_candidates,
)

from src.api.common import require_completed_job


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id}/results",
    summary="Ranked candidates for a job",
)
def job_results(job_id: str):
    """Return the ranked candidates of a completed job."""

    # Step 1: the job must exist and be completed.
    paths, status = require_completed_job(job_id, resource="results")

    # Step 2: the saved inputs are needed for the skill columns.
    try:
        inputs = load_job_inputs_file(paths)

    except (ValueError, OSError) as exc:
        logger.error("Unreadable job.json for %s: %s", job_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Job inputs file is unreadable.",
        )

    if inputs is None:
        raise HTTPException(
            status_code=404,
            detail="Job inputs (job.json) not found.",
        )

    # Step 3: read the ranked CSV.
    try:
        candidates = load_ranked_candidates(paths, inputs)

    except Exception as exc:
        logger.exception("Could not load results for %s", job_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not load ranked candidates: {exc}",
        )

    return {
        "job_id": job_id,
        "job_position": status.get("job_position"),
        "status": status.get("status"),
        "candidates": candidates,
    }
