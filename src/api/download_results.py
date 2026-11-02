
# ============================================================
# FILE: src/api/download_results.py
# PURPOSE: GET /jobs/{job_id}/results/download
#
# Download the ranked candidates of a completed job as an Excel
# workbook (<job_id>_ranked_candidates.xlsx). The file is created
# from the ranked CSV if an older job predates the Excel export.
#   404  unknown job / never executed / no ranked results
#   409  job not completed
# ============================================================

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.helpers.pipeline import ensure_ranked_excel

from src.api.common import EXCEL_MEDIA_TYPE, require_completed_job


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id}/results/download",
    summary="Download the ranked candidates as an Excel file",
)
def download_results(job_id: str):
    """Serve the ranked Excel workbook for a completed job."""

    # Step 1: the job must exist and be completed.
    paths, _ = require_completed_job(job_id, resource="ranked Excel")

    # Step 2: locate (or build) the Excel file.
    try:
        excel_path = ensure_ranked_excel(paths)

    except Exception as exc:
        logger.exception("Could not prepare ranked Excel for %s", job_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not prepare the ranked Excel file: {exc}",
        )

    if excel_path is None:
        raise HTTPException(
            status_code=404,
            detail="Ranked candidates file not found.",
        )

    logger.info("Serving ranked Excel for %s: %s", job_id, excel_path)

    return FileResponse(
        excel_path,
        media_type=EXCEL_MEDIA_TYPE,
        filename=f"{job_id}_ranked_candidates.xlsx",
    )
