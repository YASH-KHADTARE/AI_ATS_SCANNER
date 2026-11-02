
# ============================================================
# FILE: config/config.py
# PURPOSE: Central configuration for the AI ATS Scanner
#
#   1. Project paths (notebook inputs/outputs, API job folders)
#   2. Model configuration and scoring weights
#   3. Skill column naming
#   4. Per-job paths (jobs/<job_id>/...)
#   5. Directory creation
#   6. Gemini API key loading
#   7. Validation of experience and skill criteria
#   8. Job inputs from files (notebooks) and from the API form
#
# Everything here is plain constants and functions; no classes.
# ============================================================

import json
import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# config/config.py -> project root is one level up.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# ------------------------------------------------------------
# 1A. Notebook inputs (data/input_data/)
#     Resumes.zip, JD.txt, skills.json, config.json
# ------------------------------------------------------------
INPUT_DIR = DATA_DIR / "input_data"

RESUMES_ZIP = INPUT_DIR / "Resumes.zip"
JD_PATH = INPUT_DIR / "JD.txt"
SKILLS_PATH = INPUT_DIR / "skills.json"
CONFIG_PATH = INPUT_DIR / "config.json"

# ------------------------------------------------------------
# 1B. Notebook outputs (data/output_data/)
#     source_resumes/      resumes unzipped from Resumes.zip
#     markdown_resumes/    resumes converted to Markdown
#     results/             analysis, shortlist and ranking CSVs
#     logs/                pipeline and error logs
# ------------------------------------------------------------
OUTPUT_DIR = DATA_DIR / "output_data"

SOURCE_RESUMES_DIR = OUTPUT_DIR / "source_resumes"
MARKDOWN_RESUMES_DIR = OUTPUT_DIR / "markdown_resumes"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

CANDIDATE_MANIFEST_PATH = OUTPUT_DIR / "candidate_manifest.csv"

RESUME_ANALYSIS_PATH = RESULTS_DIR / "1_resume_analysis.csv"
SHORTLISTED_CANDIDATES_PATH = RESULTS_DIR / "2_shortlisted_candidates.csv"
RANKED_CANDIDATES_PATH = RESULTS_DIR / "3_ranked_candidates.csv"

PIPELINE_LOG = LOGS_DIR / "pipeline.log"
RESUME_ERROR_LOG = LOGS_DIR / "resume_analysis_errors.log"
SCORING_ERROR_LOG = LOGS_DIR / "candidate_scoring_errors.log"

# ------------------------------------------------------------
# 1C. API job workspaces (jobs/<job_id>/), see get_job_paths()
# ------------------------------------------------------------
JOBS_DIR = BASE_DIR / "jobs"

# FastAPI application log
APP_LOG = DATA_DIR / "logs" / "app.log"

# Environment file holding GOOGLE_API_KEY
ENV_PATH = BASE_DIR / ".env"

# Only PDF resumes are processed; other formats in the ZIP are skipped.
SUPPORTED_RESUME_EXTENSIONS = {".pdf"}

# Server-generated job IDs, e.g. JOB_20261102_143512_A3F9
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

# Preserve the model configurations from the existing notebooks.
EXTRACTION_MODEL = "gemini-3.5-flash"
SCORING_MODEL = "gemini-3.8-flash"

LLM_TEMPERATURE = 0
REQUEST_DELAY_SECONDS = 1

# Scoring weights (must sum to 1.0)
RESUME_QUALITY_WEIGHT = 0.20
SKILLS_EXPERIENCE_WEIGHT = 0.40
JD_MATCHING_WEIGHT = 0.40


# ============================================================
# 3. SKILL COLUMN NAMES
# ============================================================

def skill_to_column(skill):
    """
    Convert a skill name into its CSV column name.

    Examples:
        "Machine Learning" -> "skill_machine_learning"
        "C++"              -> "skill_cplusplus"
        "C#"               -> "skill_csharp"
    """

    name = skill.strip().casefold()
    name = name.replace("+", "plus").replace("#", "sharp")
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")

    if not name:
        raise ValueError(
            f"Cannot create a column name for skill: {skill!r}"
        )

    return f"skill_{name}"


# ============================================================
# 4. PER-JOB PATHS
# ============================================================

def validate_job_id(job_id):
    """Reject job IDs that could escape the jobs folder."""

    if not JOB_ID_PATTERN.fullmatch(str(job_id)):
        raise ValueError(f"Invalid job ID: {job_id!r}")

    return str(job_id)


def get_job_paths(job_id):
    """
    Return every file and folder belonging to one screening job.

    jobs/<job_id>/
        input/                   resumes.zip and job.json as received
        resumes/                 PDFs extracted from the ZIP
        markdown/                resumes converted to Markdown
        results/                 analysis, shortlist and ranking CSV/XLSX
        shortlisted_resumes/     PDFs of the shortlisted candidates
        logs/                    per-stage error logs
        candidate_manifest.csv   stable candidate IDs
        status.json              pipeline progress and summary

    Returns:
        dict[str, Path]
    """

    job_id = validate_job_id(job_id)

    root = JOBS_DIR / job_id

    input_dir = root / "input"
    results_dir = root / "results"
    logs_dir = root / "logs"

    return {
        "job_id": job_id,
        "root": root,

        # Inputs as received from the API
        "input_dir": input_dir,
        "resumes_zip": input_dir / "resumes.zip",
        "job_json": input_dir / "job.json",

        # Working folders
        "source_resumes_dir": root / "resumes",
        "markdown_resumes_dir": root / "markdown",
        "shortlisted_resumes_dir": root / "shortlisted_resumes",
        "results_dir": results_dir,
        "logs_dir": logs_dir,

        # Job-level files
        "candidate_manifest_path": root / "candidate_manifest.csv",
        "status_path": root / "status.json",

        # Stage outputs
        "resume_analysis_path": results_dir / "1_resume_analysis.csv",
        "shortlisted_candidates_path": (
            results_dir / "2_shortlisted_candidates.csv"
        ),
        "ranked_candidates_path": results_dir / "3_ranked_candidates.csv",
        "ranked_candidates_excel": results_dir / "3_ranked_candidates.xlsx",

        # Per-stage error logs
        "resume_error_log": logs_dir / "resume_analysis_errors.log",
        "scoring_error_log": logs_dir / "candidate_scoring_errors.log",
    }


def create_job_directories(paths):
    """Create the job workspace folders."""

    directory_keys = [
        "input_dir",
        "source_resumes_dir",
        "markdown_resumes_dir",
        "results_dir",
        "logs_dir",
    ]

    try:
        for key in directory_keys:
            paths[key].mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create job workspace {paths['root']}: {exc}"
        ) from exc

    logger.debug("Job workspace ready: %s", paths["root"])

    return paths


def generate_job_id():
    """Create a new, unused job ID such as JOB_20261102_143512_A3F9."""

    while True:
        job_id = (
            f"JOB_{datetime.now():%Y%m%d_%H%M%S}_"
            f"{secrets.token_hex(2).upper()}"
        )

        if not (JOBS_DIR / job_id).exists():
            logger.debug("Generated job ID: %s", job_id)
            return job_id


# ============================================================
# 5. CREATE NOTEBOOK DIRECTORIES
# ============================================================

def create_directories():
    """Create the working and output directories used by the notebooks."""

    directories = [
        INPUT_DIR,
        OUTPUT_DIR,
        SOURCE_RESUMES_DIR,
        MARKDOWN_RESUMES_DIR,
        RESULTS_DIR,
        LOGS_DIR,
    ]

    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create project directories: {exc}"
        ) from exc


# ============================================================
# 6. LOAD GEMINI API KEY
# ============================================================

def load_api_key():
    """Load and validate the Google Gemini API key from .env."""

    load_dotenv(dotenv_path=ENV_PATH, override=True)

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            f"GOOGLE_API_KEY not found in {ENV_PATH}. "
            "Please check your .env file."
        )

    # Set the key explicitly for Google GenAI integrations
    os.environ["GOOGLE_API_KEY"] = api_key

    logger.debug("GOOGLE_API_KEY loaded from %s", ENV_PATH)

    return api_key


# ============================================================
# 7. VALIDATION HELPERS
# ============================================================

def load_json_file(file_path):
    """Read a JSON file and return its contents."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {file_path}: {exc}"
        ) from exc

    except OSError as exc:
        raise OSError(
            f"Could not read {file_path}: {exc}"
        ) from exc


def validate_experience_range(config):
    """
    Validate minimum and maximum experience in years/months.

    Returns:
        (minimum_years, maximum_years) as decimal years, matching
        total_experience_years in 02_filtering.ipynb.
    """

    required_keys = [
        "minimum_experience",
        "maximum_experience",
    ]

    # Step 1: both bounds must be dicts of non-negative integers.
    for key in required_keys:
        if key not in config or not isinstance(config[key], dict):
            raise ValueError(
                f"Missing or invalid configuration: {key}"
            )

        for unit in ["years", "months"]:
            value = config[key].get(unit)

            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{key}.{unit} must be a non-negative integer."
                )

        if config[key]["months"] > 11:
            raise ValueError(
                f"{key}.months must be between 0 and 11."
            )

    # Step 2: convert to decimal years.
    minimum = (
        config["minimum_experience"]["years"]
        + config["minimum_experience"]["months"] / 12
    )

    maximum = (
        config["maximum_experience"]["years"]
        + config["maximum_experience"]["months"] / 12
    )

    # Step 3: the range must be ordered.
    if minimum > maximum:
        raise ValueError(
            "Minimum experience cannot exceed maximum experience."
        )

    return minimum, maximum


def validate_skill_thresholds(skills_data, config):
    """
    Validate required skills and their matching thresholds.

    Returns:
        dict[str, str]: Skill name -> CSV column, in input order.
    """

    # Step 1: required_skills must be a non-empty list of strings.
    if not isinstance(skills_data, dict):
        raise ValueError(
            "skills.json must contain a 'required_skills' list."
        )

    skills = skills_data.get("required_skills")

    if not isinstance(skills, list) or not skills:
        raise ValueError(
            "skills.json must contain a non-empty "
            "'required_skills' list."
        )

    if any(not isinstance(skill, str) or not skill.strip()
           for skill in skills):
        raise ValueError(
            "Every required skill must be a non-empty string."
        )

    # Step 2: build unique column names.
    skill_columns = {}

    for skill in skills:
        column = skill_to_column(skill)

        if column in skill_columns.values():
            raise ValueError(
                f"required_skills contains duplicate skills: {skill!r}"
            )

        skill_columns[skill.strip()] = column

    # Step 3: at least one threshold, each within range.
    minimum_count = config.get("minimum_matched_skills")
    minimum_ratio = config.get("minimum_skill_match_ratio")

    if minimum_count is None and minimum_ratio is None:
        raise ValueError(
            "Configure minimum_matched_skills, "
            "minimum_skill_match_ratio, or both."
        )

    if minimum_count is not None:
        if type(minimum_count) is not int:
            raise ValueError(
                "minimum_matched_skills must be an integer."
            )

        if not 0 <= minimum_count <= len(skills):
            raise ValueError(
                "minimum_matched_skills must be between 0 "
                "and the number of required skills."
            )

    if minimum_ratio is not None:
        if (
            isinstance(minimum_ratio, bool)
            or not isinstance(minimum_ratio, (int, float))
            or not 0 <= minimum_ratio <= 1
        ):
            raise ValueError(
                "minimum_skill_match_ratio must be between 0 and 1."
            )

    # If both are configured, candidates must satisfy both.
    return skill_columns


# ============================================================
# 8A. JOB INPUTS FROM FILES (NOTEBOOKS)
# ============================================================

def load_job_inputs():
    """
    Load the current JD, skills and filtering configuration from
    data/input_data/. Used by the notebooks only; the API receives
    these values through the request form (see build_job_inputs).
    """

    # Step 1: all input files must exist.
    for file_path in [RESUMES_ZIP, JD_PATH, SKILLS_PATH, CONFIG_PATH]:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required input file not found: {file_path}"
            )

    if RESUMES_ZIP.stat().st_size == 0:
        raise ValueError("Resumes.zip is empty.")

    # Step 2: read the Job Description.
    try:
        jd_text = JD_PATH.read_text(encoding="utf-8").strip()

    except OSError as exc:
        raise OSError(f"Could not read {JD_PATH}: {exc}") from exc

    if not jd_text:
        raise ValueError("Job Description is empty.")

    # Step 3: load and validate skills and filtering configuration.
    skills_data = load_json_file(SKILLS_PATH)
    config = load_json_file(CONFIG_PATH)

    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object.")

    minimum_years, maximum_years = validate_experience_range(config)

    skill_columns = validate_skill_thresholds(skills_data, config)

    logger.info(
        "Job inputs loaded from files: %d skills, experience %.2f-%.2f years",
        len(skill_columns), minimum_years, maximum_years,
    )

    return {
        "jd_text": jd_text,
        # Skill name -> CSV column, e.g. "Python" -> "skill_python"
        "skill_columns": skill_columns,
        "minimum_experience_years": minimum_years,
        "maximum_experience_years": maximum_years,
        "minimum_matched_skills": config.get("minimum_matched_skills"),
        "minimum_skill_match_ratio": config.get("minimum_skill_match_ratio"),
    }


# ============================================================
# 8B. JOB INPUTS FROM THE API FORM
# ============================================================

def parse_key_skills(key_skills):
    """
    Split a comma-separated skills string into a clean list.

    "SQL, ML , Python" -> ["SQL", "ML", "Python"]
    """

    skills = [
        skill.strip()
        for skill in str(key_skills or "").split(",")
        if skill.strip()
    ]

    if not skills:
        raise ValueError(
            "key_skills must contain at least one skill, "
            "separated by commas."
        )

    return skills


def parse_skill_threshold(skill_threshold, skill_count):
    """
    Interpret the skill threshold supplied through the form.

    Accepted formats:
        "2/3"   -> at least 2 of every 3 skills (ratio 0.667)
        "0.5"   -> ratio between 0 and 1
        "2"     -> at least 2 matched skills (absolute count)

    Returns:
        (minimum_matched_skills, minimum_skill_match_ratio),
        exactly one of which is set.
    """

    text = str(skill_threshold or "").strip()

    if not text:
        raise ValueError("skill_threshold is required (e.g. 2/3).")

    # Format 1: fraction "a/b"
    if "/" in text:
        numerator, _, denominator = text.partition("/")

        try:
            numerator = int(numerator.strip())
            denominator = int(denominator.strip())

        except ValueError as exc:
            raise ValueError(
                "skill_threshold fraction must look like 2/3."
            ) from exc

        if denominator <= 0 or numerator < 0 or numerator > denominator:
            raise ValueError(
                "skill_threshold fraction must be between 0/1 and 1/1."
            )

        return None, numerator / denominator

    # Format 2 / 3: a number
    try:
        value = float(text)

    except ValueError as exc:
        raise ValueError(
            "skill_threshold must be a fraction (2/3), a ratio (0.5) "
            "or a whole number of skills (2)."
        ) from exc

    # Decimal -> ratio
    if value != int(value):
        if not 0 <= value <= 1:
            raise ValueError(
                "skill_threshold ratio must be between 0 and 1."
            )

        return None, value

    # Whole number -> count of skills
    count = int(value)

    if not 0 <= count <= skill_count:
        raise ValueError(
            "skill_threshold count must be between 0 and the number "
            f"of key skills ({skill_count})."
        )

    return count, None


def build_job_inputs(
    job_position,
    job_description,
    key_skills,
    minimum_experience_years,
    minimum_experience_months,
    maximum_experience_years,
    maximum_experience_months,
    skill_threshold,
):
    """
    Validate API form values and return the same inputs structure
    that load_job_inputs() produces from the input files.
    """

    # Step 1: text fields.
    job_position = str(job_position or "").strip()

    if not job_position:
        raise ValueError("job_position is required.")

    jd_text = str(job_description or "").strip()

    if not jd_text:
        raise ValueError("job_description is required.")

    # Step 2: skills and threshold.
    skills = parse_key_skills(key_skills)

    minimum_count, minimum_ratio = parse_skill_threshold(
        skill_threshold, len(skills)
    )

    # Step 3: reuse the file-based validators by shaping the form
    # values like config.json / skills.json.
    config = {
        "minimum_experience": {
            "years": minimum_experience_years,
            "months": minimum_experience_months,
        },
        "maximum_experience": {
            "years": maximum_experience_years,
            "months": maximum_experience_months,
        },
        "minimum_matched_skills": minimum_count,
        "minimum_skill_match_ratio": minimum_ratio,
    }

    minimum_years, maximum_years = validate_experience_range(config)

    skill_columns = validate_skill_thresholds(
        {"required_skills": skills},
        config,
    )

    logger.info(
        "Job inputs validated: position=%r, %d skills, threshold=%s, "
        "experience %.2f-%.2f years",
        job_position, len(skills), str(skill_threshold).strip(),
        minimum_years, maximum_years,
    )

    return {
        "job_position": job_position,
        "jd_text": jd_text,
        "key_skills": skills,
        "skill_threshold": str(skill_threshold).strip(),
        "skill_columns": skill_columns,
        "minimum_experience_years": minimum_years,
        "maximum_experience_years": maximum_years,
        "minimum_experience": config["minimum_experience"],
        "maximum_experience": config["maximum_experience"],
        "minimum_matched_skills": minimum_count,
        "minimum_skill_match_ratio": minimum_ratio,
    }


# ============================================================
# 9. LOAD ALL PROJECT SETTINGS (NOTEBOOKS)
# ============================================================

def load_project_settings():
    """Load the API key, directories and current job inputs."""

    create_directories()

    api_key = load_api_key()
    job_inputs = load_job_inputs()

    return {
        "base_dir": BASE_DIR,
        "api_key": api_key,
        "job_inputs": job_inputs,
        "extraction_model": EXTRACTION_MODEL,
        "scoring_model": SCORING_MODEL,
        "temperature": LLM_TEMPERATURE,
        "request_delay_seconds": REQUEST_DELAY_SECONDS,
    }
