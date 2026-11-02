
# ============================================================
# FILE: src/api/screen_job.py
# PURPOSE: POST /jobs/screen
#
# Upload a ZIP of PDF resumes with the job details, run the full
# pipeline synchronously and return the ranked candidates.
#
#   A. validate the form        -> 400 on bad input
#   B. create jobs/<job_id>/    save ZIP + job.json
#   C. run the pipeline         -> 400 input problems, 500 failures
#   D. return summary + ranked candidates
# ============================================================

import logging
import shutil

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import (
    build_job_inputs,
    create_job_directories,
    generate_job_id,
    get_job_paths,
)
from src.helpers.file_handler import validate_zip_file
from src.helpers.pipeline import (
    load_ranked_candidates,
    run_pipeline,
    save_job_inputs,
)

from src.api.common import save_upload


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ============================================================
# 1. HELPERS
# ============================================================

def discard_workspace(paths, reason):
    """Remove a job folder that never reached the pipeline."""

    shutil.rmtree(paths["root"], ignore_errors=True)

    logger.warning(
        "Discarded workspace for %s: %s", paths["job_id"], reason
    )


# ============================================================
# 2. ROUTE
# ============================================================

@router.post(
    "/screen",
    summary="Upload resumes and run the full screening pipeline",
)
def screen_job(
    resumes: UploadFile = File(
        ..., description="ZIP archive of PDF resumes."
    ),
    job_position: str = Form(..., description="Job title, e.g. AI/ML Engineer"),
    job_description: str = Form(..., description="Full job description text"),
    key_skills: str = Form(
        ..., description="Comma-separated skills, e.g. SQL, ML, Python"
    ),
    skill_threshold: str = Form(
        ...,
        description=(
            "Minimum skill match: a fraction (2/3), a ratio (0.5) "
            "or a count of skills (2)."
        ),
    ),
    min_experience_years: int = Form(..., ge=0),
    min_experience_months: int = Form(..., ge=0, le=11),
    max_experience_years: int = Form(..., ge=0),
    max_experience_months: int = Form(..., ge=0, le=11),
):
    """
    Run every pipeline stage synchronously and return the ranked
    candidates. The request blocks until the ranked files are written.

    All job data is stored in jobs/<job_id>/.
    """

    # ----------------------------------------------------
    # A. Validate the form inputs
    # ----------------------------------------------------
    try:
        inputs = build_job_inputs(
            job_position=job_position,
            job_description=job_description,
            key_skills=key_skills,
            minimum_experience_years=min_experience_years,
            minimum_experience_months=min_experience_months,
            maximum_experience_years=max_experience_years,
            maximum_experience_months=max_experience_months,
            skill_threshold=skill_threshold,
        )

    except ValueError as exc:
        logger.warning("Rejected screening request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    if not (resumes.filename or "").lower().endswith(".zip"):
        logger.warning(
            "Rejected upload with non-ZIP name: %r", resumes.filename
        )
        raise HTTPException(
            status_code=400,
            detail="resumes must be a .zip file.",
        )

    # ----------------------------------------------------
    # B. Create the job workspace and store the inputs
    # ----------------------------------------------------
    job_id = generate_job_id()

    try:
        paths = create_job_directories(get_job_paths(job_id))

    except OSError as exc:
        logger.exception("Could not create workspace for %s", job_id)
        raise HTTPException(
            status_code=500,
            detail="Could not create the job workspace.",
        )

    try:
        save_upload(resumes, paths["resumes_zip"])
        validate_zip_file(paths["resumes_zip"])

    except ValueError as exc:
        # Nothing has run yet: discard the half-created workspace.
        discard_workspace(paths, str(exc))

        # Do not echo server paths back to the client.
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty, corrupt or not a ZIP archive.",
        )

    except OSError as exc:
        discard_workspace(paths, str(exc))

        logger.exception("Could not store upload for %s", job_id)
        raise HTTPException(
            status_code=500,
            detail="Could not store the uploaded file.",
        )

    try:
        save_job_inputs(paths, inputs)

    except OSError as exc:
        discard_workspace(paths, str(exc))

        logger.exception("Could not save job.json for %s", job_id)
        raise HTTPException(
            status_code=500,
            detail="Could not save the job inputs.",
        )

    logger.info(
        "Job %s created | position=%r | skills=%s | folder=%s",
        job_id, inputs["job_position"], inputs["key_skills"], paths["root"],
    )

    # ----------------------------------------------------
    # C. Run the pipeline (blocking)
    # ----------------------------------------------------
    try:
        summary = run_pipeline(paths, inputs)

    except (ValueError, FileNotFoundError) as exc:
        # Input problems surfaced while processing (e.g. no PDFs).
        # The workspace is kept so the failure can be inspected.
        raise HTTPException(
            status_code=400,
            detail={"job_id": job_id, "error": str(exc)},
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "job_id": job_id,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    # ----------------------------------------------------
    # D. Respond with the summary and ranked candidates
    # ----------------------------------------------------
    try:
        candidates = load_ranked_candidates(paths, inputs)

    except Exception:
        # The pipeline finished; a read-back problem should not
        # turn a successful run into an error response.
        logger.exception("Could not load ranked candidates for %s", job_id)
        candidates = []

    return {
        "job_id": job_id,
        "job_folder": str(paths["root"]),
        "inputs": inputs,
        "summary": summary,
        "candidates": candidates,
    }
