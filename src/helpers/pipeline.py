
# ============================================================
# FILE: src/helpers/pipeline.py
# PURPOSE: Orchestrate the AI ATS Scanner pipeline for one job
#
# Stages (mirroring the notebooks):
#   1. prepare_resumes             unzip, keep PDFs only
#   2. identify_candidates         stable candidate IDs
#   3. extract_resumes_to_markdown Docling PDF -> Markdown
#   4. analyze_resumes             Gemini: contact, experience, skills
#   5. shortlist_candidates        experience + skills filters
#   6. score_candidates            Gemini scoring + ranking
#   7. finalize                    Excel export, shortlisted PDFs
#
# Every path comes from config.get_job_paths(job_id), so each job
# runs inside its own jobs/<job_id>/ workspace. Progress is
# written to status.json after every stage.
# ============================================================

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import SUPPORTED_RESUME_EXTENSIONS, create_job_directories
from src.helpers.csv_handler import read_csv_file, save_csv_file
from src.helpers.file_handler import prepare_resumes
from src.helpers.candidate_identifier import identify_candidates
from src.helpers.docling_extractor import extract_resumes_to_markdown
from src.helpers.resume_analyzer import analyze_resumes, get_contact_dtypes
from src.helpers.experience_filter import filter_by_experience
from src.helpers.skills_filter import filter_by_skills
from src.helpers.candidate_scoring import score_candidates, SCORING_COLUMNS


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 1. JOB STATUS AND INPUT FILES
# ------------------------------------------------------------

def _now():
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path):
    """Read a JSON file, raising a clear error on corruption."""

    try:
        with Path(path).open("r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    except OSError as exc:
        raise OSError(f"Could not read {path}: {exc}") from exc


def _write_json(path, data):
    """Write a JSON file, creating the parent folder if needed."""

    path = Path(path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

    except OSError as exc:
        raise OSError(f"Could not write {path}: {exc}") from exc


def read_status(paths):
    """Return the job's status.json contents, or None if missing."""

    if not paths["status_path"].is_file():
        return None

    return _read_json(paths["status_path"])


def write_status(paths, status, stage, **fields):
    """
    Update status.json, keeping fields written by earlier stages.

    status: "running" | "completed" | "failed"
    stage : human-readable description of the current stage
    """

    current = read_status(paths) or {
        "job_id": paths["job_id"],
        "created_at": _now(),
    }

    current.update(fields)
    current.update({
        "status": status,
        "stage": stage,
        "updated_at": _now(),
    })

    _write_json(paths["status_path"], current)

    logger.debug("Status -> %s | %s", status, stage)

    return current


def save_job_inputs(paths, inputs):
    """Persist the form inputs next to the uploaded ZIP."""

    _write_json(paths["job_json"], inputs)


def load_job_inputs_file(paths):
    """Return the job.json saved for this job, or None if missing."""

    if not paths["job_json"].is_file():
        return None

    return _read_json(paths["job_json"])


# ------------------------------------------------------------
# 2. SHORTLIST CANDIDATES (02_filtering.ipynb)
# ------------------------------------------------------------

def shortlist_candidates(inputs, paths):
    """
    Apply experience and skills filters to the resume analysis CSV
    and save the shortlisted candidates.

    Returns:
        pandas.DataFrame: Shortlisted candidates (may be empty).
    """

    analysis_path = paths["resume_analysis_path"]
    shortlisted_path = paths["shortlisted_candidates_path"]

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {analysis_path}. "
            "Check whether the analysis stage completed."
        )

    # Step 1: load the analysis CSV (contact details stay text).
    df = read_csv_file(analysis_path, dtype=get_contact_dtypes())

    logger.info("Total candidates loaded: %d", len(df))

    skill_columns = list(inputs["skill_columns"].values())

    # Step 2: validate required columns.
    required_columns = [
        "file_name",
        "experience_years",
        "experience_months",
        *skill_columns,
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    # Step 3: numeric conversion; drop rows with unusable values.
    for col in required_columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=required_columns).copy()

    if len(df) < before:
        logger.warning(
            "%d candidate(s) dropped for missing required values.",
            before - len(df),
        )

    # Step 4: apply both filters.
    if df.empty:
        shortlisted_df = df
    else:
        experience_filter = filter_by_experience(
            df,
            minimum_experience_years=inputs["minimum_experience_years"],
            maximum_experience_years=inputs["maximum_experience_years"],
        )

        skills_filter = filter_by_skills(
            df,
            skill_columns=skill_columns,
            minimum_matched_skills=inputs["minimum_matched_skills"],
            minimum_skill_match_ratio=inputs["minimum_skill_match_ratio"],
        )

        shortlisted_df = df[experience_filter & skills_filter].copy()

        shortlisted_df = shortlisted_df.sort_values(
            by=["total_experience_years", "skills_matched"],
            ascending=[True, False],
        )

    # Step 5: save.
    save_csv_file(shortlisted_df, shortlisted_path)

    logger.info("=" * 60)
    logger.info("SHORTLISTING SUMMARY")
    logger.info("=" * 60)
    logger.info("Candidates evaluated  : %d", len(df))
    logger.info("Candidates shortlisted: %d", len(shortlisted_df))
    logger.info("Output saved to       : %s", shortlisted_path)

    return shortlisted_df


# ------------------------------------------------------------
# 3. FINALIZE RESULTS
# ------------------------------------------------------------

def save_ranked_excel(ranked_df, paths):
    """Save the ranked candidates as an Excel workbook."""

    excel_path = paths["ranked_candidates_excel"]

    try:
        excel_path.parent.mkdir(parents=True, exist_ok=True)

        ranked_df.to_excel(
            excel_path,
            index=False,
            sheet_name="Ranked Candidates",
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not write ranked Excel {excel_path}: {exc}"
        ) from exc

    logger.info("Ranked Excel saved to: %s", excel_path)

    return excel_path


def ensure_ranked_excel(paths):
    """
    Return the ranked Excel path, creating it from the ranked CSV
    when a job predates the Excel export.

    Returns:
        Path | None: None when no ranked results exist.
    """

    excel_path = paths["ranked_candidates_excel"]

    if excel_path.is_file():
        return excel_path

    if not paths["ranked_candidates_path"].is_file():
        return None

    logger.info("Ranked Excel missing; generating from CSV.")

    ranked_df = read_csv_file(
        paths["ranked_candidates_path"], dtype=get_contact_dtypes()
    )

    return save_ranked_excel(ranked_df, paths)


def copy_shortlisted_resumes(paths, candidates_df):
    """
    Copy the source PDF of every shortlisted candidate into
    shortlisted_resumes/ so HR can open them directly.

    Returns:
        list[str]: Names of the copied resume files.
    """

    folder = paths["shortlisted_resumes_dir"]

    try:
        folder.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create folder {folder}: {exc}"
        ) from exc

    # Step 1: index the extracted PDFs by stem. file_name in the CSVs
    # is the Markdown name (resume.md); the stem links the two.
    source_index = {}

    for path in paths["source_resumes_dir"].rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_RESUME_EXTENSIONS
        ):
            source_index.setdefault(path.stem.lower(), path)

    # Step 2: copy each shortlisted candidate's PDF.
    copied_files = []

    for file_name in candidates_df["file_name"].astype(str):

        source = source_index.get(Path(file_name).stem.lower())

        if source is None:
            logger.warning(
                "Source resume not found for shortlisted candidate: %s",
                file_name,
            )
            continue

        destination = folder / source.name

        try:
            shutil.copy2(source, destination)
            copied_files.append(destination.name)

        except OSError as exc:
            logger.error(
                "Could not copy %s to %s: %s", source, destination, exc
            )

    logger.info("=" * 60)
    logger.info("SHORTLISTED RESUMES")
    logger.info("=" * 60)
    logger.info("Copied resumes : %d", len(copied_files))
    logger.info("Output folder  : %s", folder)

    return copied_files


def finalize_results(paths, ranked_df):
    """Excel export + shortlisted PDFs. Returns the copied file names."""

    logger.info("Finalizing: Excel export and shortlisted resumes.")

    save_ranked_excel(ranked_df, paths)

    return copy_shortlisted_resumes(paths, ranked_df)


# ------------------------------------------------------------
# 4. RUN THE PIPELINE
# ------------------------------------------------------------

def run_pipeline(paths, inputs):
    """
    Execute the complete screening and ranking workflow for one job.

    The uploaded ZIP must already be saved at paths["resumes_zip"].
    On failure, status.json records the error and the exception is
    re-raised for the caller.

    Returns:
        dict: The final status.json contents (summary of the run).
    """

    create_job_directories(paths)

    job_id = paths["job_id"]

    logger.info("=" * 65)
    logger.info("AI ATS SCANNER STARTED | job %s", job_id)
    logger.info("=" * 65)

    write_status(
        paths, "running", "Stage 1/6: Preparing resumes",
        job_position=inputs["job_position"],
    )

    try:
        # ----------------------------------------------------
        # Stage 1: Extract resumes from the uploaded ZIP
        # ----------------------------------------------------
        logger.info("Stage 1/6: Preparing resumes.")

        resume_paths, skipped_files = prepare_resumes(
            zip_path=paths["resumes_zip"],
            source_dir=paths["source_resumes_dir"],
        )

        write_status(
            paths, "running", "Stage 2/6: Identifying candidates",
            resumes_total=len(resume_paths),
            skipped_files=skipped_files,
        )

        # ----------------------------------------------------
        # Stage 2: Assign persistent candidate IDs
        # ----------------------------------------------------
        logger.info("Stage 2/6: Identifying candidates.")

        identify_candidates(
            resume_paths,
            source_dir=paths["source_resumes_dir"],
            manifest_path=paths["candidate_manifest_path"],
        )

        write_status(
            paths, "running", "Stage 3/6: Converting resumes to Markdown"
        )

        # ----------------------------------------------------
        # Stage 3: Convert resumes into Markdown
        # ----------------------------------------------------
        logger.info("Stage 3/6: Converting resumes to Markdown.")

        extraction_df = extract_resumes_to_markdown(
            resume_paths,
            output_folder=paths["markdown_resumes_dir"],
        )

        markdown_failed = extraction_df.loc[
            extraction_df["status"] != "Success", "file_name"
        ].tolist()

        if not extraction_df["status"].eq("Success").any():
            raise RuntimeError(
                "No resumes were successfully converted to Markdown."
            )

        write_status(
            paths, "running", "Stage 4/6: Analyzing resumes with Gemini",
            markdown_failed=markdown_failed,
        )

        # ----------------------------------------------------
        # Stage 4: Extract contact, experience and skills
        # ----------------------------------------------------
        logger.info("Stage 4/6: Analyzing resumes with Gemini.")

        analysis_df = analyze_resumes(
            skill_columns=inputs["skill_columns"],
            resume_folder=paths["markdown_resumes_dir"],
            output_csv=paths["resume_analysis_path"],
            error_log=paths["resume_error_log"],
        )

        converted = len(extraction_df) - len(markdown_failed)

        write_status(
            paths, "running", "Stage 5/6: Applying screening filters",
            analyzed=int(len(analysis_df)),
            analysis_failed=max(0, int(converted - len(analysis_df))),
        )

        # ----------------------------------------------------
        # Stage 5: Apply experience and skills filters
        # ----------------------------------------------------
        logger.info("Stage 5/6: Applying screening filters.")

        shortlisted_df = shortlist_candidates(inputs, paths)

        write_status(
            paths, "running", "Stage 6/6: Scoring and ranking candidates",
            shortlisted=int(len(shortlisted_df)),
        )

        if shortlisted_df.empty:
            logger.warning(
                "No candidates passed the configured screening filters. "
                "Scoring will not run."
            )

            # Write empty ranked files so every completed job has them.
            empty_ranked_df = pd.DataFrame(
                columns=list(shortlisted_df.columns) + SCORING_COLUMNS
            )

            save_csv_file(empty_ranked_df, paths["ranked_candidates_path"])

            finalize_results(paths, empty_ranked_df)

            return write_status(
                paths, "completed",
                "Completed: no candidates passed the filters",
                ranked=0,
                scoring_failed=0,
                shortlisted_resumes=0,
            )

        # ----------------------------------------------------
        # Stage 6: Score, rank and save shortlisted candidates
        # ----------------------------------------------------
        logger.info("Stage 6/6: Scoring and ranking candidates.")

        ranked_df = score_candidates(
            jd_text=inputs["jd_text"],
            input_csv=paths["shortlisted_candidates_path"],
            ranked_output_csv=paths["ranked_candidates_path"],
            resume_folder=paths["markdown_resumes_dir"],
            error_log=paths["scoring_error_log"],
        )

        # ----------------------------------------------------
        # Stage 7: Excel export and shortlisted resumes folder
        # ----------------------------------------------------
        copied_resumes = finalize_results(paths, ranked_df)

        logger.info("AI ATS SCANNER COMPLETED SUCCESSFULLY | job %s", job_id)

        return write_status(
            paths, "completed", "Completed",
            ranked=int(ranked_df["final_score"].notna().sum()),
            scoring_failed=int(ranked_df["final_score"].isna().sum()),
            shortlisted_resumes=len(copied_resumes),
        )

    except Exception as exc:
        logger.exception("Pipeline failed for job %s: %s", job_id, exc)

        # Recording the failure must never mask the original error.
        try:
            write_status(
                paths, "failed", "Failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            logger.exception("Could not record failure in status.json.")

        raise


# ------------------------------------------------------------
# 5. LOAD RANKED RESULTS
# ------------------------------------------------------------

def _clean_value(value):
    """Convert pandas/numpy values into JSON-friendly Python values."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def load_manifest_by_stem(manifest_path):
    """
    Map resume stem -> {candidate_id, source_file} from the manifest.

    The analysis CSV stores the Markdown name (resume.md) while the
    manifest stores the source PDF path (folder/resume.pdf); the
    stem is what links them.
    """

    manifest_by_stem = {}

    if not Path(manifest_path).is_file():
        logger.warning("Candidate manifest not found: %s", manifest_path)
        return manifest_by_stem

    manifest = read_csv_file(manifest_path)

    for _, row in manifest.iterrows():
        source_path = str(row["source_relative_path"])
        stem = Path(source_path).stem.lower()

        manifest_by_stem.setdefault(stem, {
            "candidate_id": str(row["candidate_id"]),
            "source_file": source_path,
        })

    return manifest_by_stem


def load_ranked_candidates(paths, inputs):
    """
    Read the ranked CSV and return one dict per candidate, joined
    with the candidate ID from the manifest.

    Returns:
        list[dict]: Candidates in ranked order (best first).
    """

    ranked_path = paths["ranked_candidates_path"]

    if not ranked_path.is_file():
        logger.warning("Ranked CSV not found: %s", ranked_path)
        return []

    # Step 1: load ranked results and the manifest.
    try:
        ranked_df = read_csv_file(ranked_path, dtype=get_contact_dtypes())

    except ValueError:
        # Empty CSV (no header) -> no candidates.
        return []

    if ranked_df.empty:
        return []

    manifest_by_stem = load_manifest_by_stem(
        paths["candidate_manifest_path"]
    )

    skill_columns = inputs["skill_columns"]

    # Step 2: build one JSON-friendly record per candidate.
    candidates = []

    for rank, (_, row) in enumerate(ranked_df.iterrows(), start=1):
        file_name = str(row["file_name"])

        record = manifest_by_stem.get(Path(file_name).stem.lower(), {})

        candidates.append({
            "rank": rank,
            "candidate_id": record.get("candidate_id"),
            "file_name": file_name,
            "source_file": record.get("source_file"),
            "candidate_name": _clean_value(row.get("candidate_name")),
            "email": _clean_value(row.get("email")),
            "phone": _clean_value(row.get("phone")),
            "experience_years": _clean_value(row.get("experience_years")),
            "experience_months": _clean_value(row.get("experience_months")),
            "total_experience_years": _clean_value(
                row.get("total_experience_years")
            ),
            "skills_matched": _clean_value(row.get("skills_matched")),
            "skills": {
                skill: _clean_value(row.get(column))
                for skill, column in skill_columns.items()
            },
            "resume_quality_score": _clean_value(
                row.get("resume_quality_score")
            ),
            "skills_projects_experience_score": _clean_value(
                row.get("skills_projects_experience_score")
            ),
            "jd_matching_score": _clean_value(row.get("jd_matching_score")),
            "final_score": _clean_value(row.get("final_score")),
        })

    logger.info("Loaded %d ranked candidate(s) for %s.", len(candidates), paths["job_id"])

    return candidates
