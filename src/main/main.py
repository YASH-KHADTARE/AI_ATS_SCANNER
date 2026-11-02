
# ============================================================
# FILE: src/main/main.py
# PURPOSE: FastAPI application for the AI ATS Scanner
#
# Run from the project root:
#   uvicorn src.main.main:app --reload
#
# Interactive docs: http://127.0.0.1:8000/docs
#
# Routers (one file per route in src/api/):
#   POST /jobs/screen                      screen_job.py
#   GET  /jobs                             list_jobs.py
#   GET  /jobs/{job_id}/results            job_results.py
#   GET  /jobs/{job_id}/results/download   download_results.py
# ============================================================

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config import APP_LOG, JOBS_DIR, load_api_key
from src.helpers.logging_config import setup_logging

from src.api.screen_job import router as screen_job_router
from src.api.list_jobs import router as list_jobs_router
from src.api.job_results import router as job_results_router
from src.api.download_results import router as download_results_router


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 1. STARTUP / SHUTDOWN
# ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    """
    Configure logging, prepare the jobs folder and fail fast if the
    Gemini API key is missing. Runs once when the server starts.
    """

    # Step 1: logging to console + data/logs/app.log.
    setup_logging(log_file=APP_LOG)

    # Step 2: the jobs folder must be writable.
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        logger.critical("Cannot create jobs folder %s: %s", JOBS_DIR, exc)
        raise

    # Step 3: the API key must be present before accepting requests.
    try:
        load_api_key()

    except ValueError as exc:
        logger.critical("Startup aborted: %s", exc)
        raise

    logger.info("GOOGLE_API_KEY loaded successfully.")
    logger.info("Job workspaces: %s", JOBS_DIR)
    logger.info("AI ATS Scanner API ready.")

    yield

    logger.info("AI ATS Scanner API shutting down.")


# ------------------------------------------------------------
# 2. APPLICATION
# ------------------------------------------------------------

app = FastAPI(
    title="AI ATS Scanner",
    description=(
        "Upload a ZIP of PDF resumes with the job details, run the "
        "extraction, filtering and scoring pipeline, and retrieve the "
        "ranked candidates."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# One router per route file.
app.include_router(screen_job_router)
app.include_router(list_jobs_router)
app.include_router(job_results_router)
app.include_router(download_results_router)


@app.get("/", tags=["health"], summary="Health check")
def health():
    """Liveness check; does not touch Gemini or the file system."""
    return {"status": "ok", "service": "AI ATS Scanner"}


# ------------------------------------------------------------
# 3. ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.main.main:app", host="127.0.0.1", port=8000, reload=True
    )
