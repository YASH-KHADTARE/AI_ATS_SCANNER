
# ============================================================
# FILE: src/helpers/resume_analyzer.py
# PURPOSE: Extract structured information from resume Markdown
#          using Gemini (pipeline stage 4;
#          01_data_extraction.ipynb, cells 3-6)
#
# For every Markdown resume the LLM returns:
#   - candidate_name, email, phone   (optional strings)
#   - experience_years, experience_months (integers)
#   - skill_<name>: 0/1 for each key skill passed by HR
#
# Results are appended to the analysis CSV one row at a time, so
# the CSV doubles as a checkpoint: a re-run skips completed resumes.
# ============================================================

import json
import logging
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    MARKDOWN_RESUMES_DIR,
    EXTRACTION_MODEL,
    LLM_TEMPERATURE,
    REQUEST_DELAY_SECONDS,
    RESUME_ANALYSIS_PATH,
    RESUME_ERROR_LOG,
    load_api_key,
)
from src.helpers.logging_config import get_error_logger


logger = logging.getLogger(__name__)

CSV_ENCODING = "utf-8-sig"


# ============================================================
# 1. OUTPUT SCHEMA
# ============================================================

# Basic candidate details. Optional: blank in the CSV means the
# resume did not contain the information.
CONTACT_KEYS = [
    "candidate_name",
    "email",
    "phone",
]

# Always required integers.
EXPERIENCE_KEYS = [
    "experience_years",
    "experience_months",
]

# Regex fallbacks used when the LLM returns null for email/phone.
EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
PHONE_PATTERN = re.compile(
    r"(?<![\w.])(?:\+?\d[\d\s().-]{7,}\d)(?![\w.])"
)


def get_required_keys(skill_columns):
    """JSON fields the LLM must return as integers."""
    return EXPERIENCE_KEYS + list(skill_columns.values())


def get_output_columns(skill_columns):
    """Columns saved in the resume analysis CSV."""
    return ["file_name"] + CONTACT_KEYS + get_required_keys(skill_columns)


def get_contact_dtypes():
    """pandas dtype map keeping contact columns as text."""
    return {key: str for key in CONTACT_KEYS}


# ============================================================
# 2. EXTRACTION PROMPT
# ============================================================

RESUME_ANALYSIS_PROMPT = """
You are an expert resume parsing and professional experience calculation
system. Extract accurate, structured information from the resume.

RESUME:
{resume_text}

CURRENT DATE:
{current_date}

INSTRUCTIONS:

============================================================
0. CANDIDATE DETAILS
============================================================

Extract the candidate's basic contact details exactly as written
in the resume:
- candidate_name: the candidate's full name, usually at the top.
- email: the candidate's email address.
- phone: the candidate's contact number, including country code
  if present.

Do not guess or invent any of these. If a value is not present in
the resume, return null for that field. Do not return the names,
emails or phone numbers of references, companies or recruiters.

============================================================
1. PROFESSIONAL WORK EXPERIENCE
============================================================

Calculate total professional work experience using the following
strict, sequential procedure.

IMPORTANT:
- Do not guess or invent dates.
- Do not double-count overlapping employment periods.
- Use the CURRENT DATE provided above for all calculations.
- Follow the steps in order.

STEP 1: IDENTIFY EXPLICITLY MENTIONED TOTAL EXPERIENCE

First, search the entire resume for statements that explicitly mention
total professional experience.

Examples:
- "3 years of experience"
- "2 years and 6 months of experience"
- "Total experience: 4.5 years"
- "Professional experience: 36 months"

Record the explicitly mentioned total experience internally.
Do not assume that this value is correct without checking the
employment date ranges in Step 2.

If the resume does not explicitly mention total experience,
record that no explicit total was found.

Do not confuse:
- Years of experience required for a job
- Experience mentioned in a job description
- Academic experience
- Internship duration
- Project duration
with the candidate's actual total professional experience.

STEP 2: IDENTIFY ALL PROFESSIONAL EMPLOYMENT DATE RANGES

Read the resume carefully and identify every employment period
listed under sections such as:
- Professional Experience
- Work Experience
- Employment History
- Experience
- Career History

For each employment entry, extract:
- Start month and year, if available
- End month and year, if available
- Whether the role is marked Present, Current, or Ongoing

Examples:
- January 2022 - March 2023
- Jun 2023 - Dec 2024
- February 2025 - Present

Include professional full-time and part-time employment if listed.
Include freelance work only when explicitly presented as professional
employment.

Do not count education, academic projects, or internships as
professional work experience unless they are explicitly identified
as professional employment.

Do not use a stated total-experience figure as a substitute for
identifying employment date ranges.

DATE HANDLING RULES:

A. If the end date is "Present", "Current", or "Ongoing", use the
   CURRENT DATE provided above as the end date.

B. If both start and end month/year are available, use those dates.

C. If only a year is available, do not invent a month. Use January
   as the start month only when the start year alone is given, and
   December as the end month only when the end year alone is given.
   Treat these as explicit estimation conventions and apply them
   consistently.

D. If the start date is missing or unusable, do not invent it.
   Exclude that period from the date-based calculation.

E. If an employment entry contains conflicting or impossible dates,
   do not invent a correction. Exclude the invalid period.

STEP 3: CALCULATE TOTAL EXPERIENCE PRECISELY

Calculate the duration of every valid employment date range.

Use calendar months as the calculation unit.

For each date range:
- Convert the start and end dates to year-month values.
- Calculate the number of months between them:
  (end_year - start_year) * 12 + (end_month - start_month).
- If the end date is Present, use the current year and month from
  the CURRENT DATE provided above.
- Do not add an extra month merely because both endpoint months
  appear in the resume.

Example:
January 2022 - January 2023 = 12 months.
June 2023 - December 2024 = 18 months.

Then:
- Combine all valid employment periods.
- Identify overlapping periods.
- Count overlapping calendar months only once.
- Sum the non-overlapping employment months to obtain total
  professional experience in months.

If the candidate changed jobs without a gap, count the continuous
employment period normally. If there is a gap between jobs, do not
count the gap.

STEP 4: CROSS-CHECK THE EXPLICIT TOTAL

Compare the calculated date-based total from Step 3 with the
explicitly stated total from Step 1, if available.

The final result MUST be based on the date ranges identified and
calculated in Steps 2 and 3, not on the stated total.

If the stated total differs from the date-based calculation,
use the date-based calculation.

Do not add the explicitly stated total to the date-based total.
Do not count the same experience twice.

STEP 5: CONVERT TOTAL MONTHS TO YEARS AND REMAINING MONTHS

After calculating the final total number of months:

experience_years = total_months // 12
experience_months = total_months % 12

Examples:
- 6 total months = 0 years, 6 months
- 12 total months = 1 year, 0 months
- 18 total months = 1 year, 6 months
- 29 total months = 2 years, 5 months
- 36 total months = 3 years, 0 months

If no valid professional employment date ranges can be identified,
return:
- experience_years = 0
- experience_months = 0

Do not use an explicitly stated total as the final result when
there are no usable employment date ranges.

============================================================
2. SKILLS
============================================================

For each required skill below, return a binary indicator:
1 if the skill is explicitly mentioned anywhere in the resume
(including a common abbreviation or alternate name, such as
"ML" for Machine Learning) or clearly demonstrated by the
candidate's skills, projects or work, otherwise 0.

REQUIRED SKILLS (JSON field: skill name):
{skill_instructions}

Do not infer skills solely from unrelated job titles.

============================================================
3. OUTPUT REQUIREMENTS
============================================================

Return only valid JSON matching the specified output schema.

Required fields:
- candidate_name: string, or null if not present
- email: string, or null if not present
- phone: string, or null if not present
- experience_years: non-negative integer
- experience_months: integer from 0 to 11
{skill_fields}

Do not include additional fields or explanations.

Before returning the JSON, internally verify:
1. All eligible employment periods were identified.
2. Present roles were calculated through the supplied current date.
3. Overlapping periods were counted only once.
4. The total months were converted correctly into years and months.
5. The stated total experience was not added again to the date-based
   calculation.

{format_instructions}
"""


# ============================================================
# 3. INITIALIZE GEMINI, JSON PARSER AND CHAIN
# ============================================================

def create_resume_analyzer(skill_columns):
    """Create the Gemini extraction chain for the required skills."""

    load_api_key()

    try:
        llm = ChatGoogleGenerativeAI(
            model=EXTRACTION_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize Gemini model {EXTRACTION_MODEL}: {exc}"
        ) from exc

    parser = JsonOutputParser()

    # Insert the key skills into the prompt
    skill_instructions = "\n".join(
        f"- {column}: {skill}"
        for skill, column in skill_columns.items()
    )

    skill_fields = "\n".join(
        f"- {column}: integer, either 0 or 1"
        for column in skill_columns.values()
    )

    prompt = PromptTemplate(
        template=RESUME_ANALYSIS_PROMPT,
        input_variables=["resume_text", "current_date"],
        partial_variables={
            "skill_instructions": skill_instructions,
            "skill_fields": skill_fields,
            "format_instructions": parser.get_format_instructions(),
        },
    )

    logger.info(
        "Resume analyzer ready: model=%s, %d skills.",
        EXTRACTION_MODEL, len(skill_columns),
    )

    # Build the LangChain pipeline
    return prompt | llm | parser


# ============================================================
# 4. VALIDATE THE RETURNED JSON FIELDS
# ============================================================

def clean_contact_value(value):
    """Return a stripped string, or None for missing/blank values."""

    if not isinstance(value, str):
        return None

    value = " ".join(value.split())

    return value or None


def extract_contact_fallback(resume_text):
    """
    Recover email and phone with regex when the LLM returns null.

    Returns:
        dict: {"email": str | None, "phone": str | None}
    """

    email_match = EMAIL_PATTERN.search(resume_text)

    phone = None

    for match in PHONE_PATTERN.finditer(resume_text):
        digits = re.sub(r"\D", "", match.group())

        # Real phone numbers have 10-15 digits; skip dates, IDs, etc.
        if 10 <= len(digits) <= 15:
            phone = " ".join(match.group().split())
            break

    return {
        "email": email_match.group() if email_match else None,
        "phone": phone,
    }


def validate_analysis_response(response, skill_columns, resume_text=""):
    """
    Validate and normalize the LLM response before saving it.

    Contact details are optional and never fail the resume.
    Experience and skill fields must be integers in range.
    """

    if not isinstance(response, dict):
        raise ValueError("LLM response is not a JSON object.")

    # Step 1: normalize contact details.
    for key in CONTACT_KEYS:
        response[key] = clean_contact_value(response.get(key))

    # Step 2: regex fallback for missing email/phone.
    if resume_text and (
        response["email"] is None or response["phone"] is None
    ):
        fallback = extract_contact_fallback(resume_text)

        for key in ["email", "phone"]:
            if response[key] is None and fallback[key]:
                response[key] = fallback[key]
                logger.info("%s recovered with regex fallback.", key)

    # Step 3: required integer fields.
    required_keys = get_required_keys(skill_columns)

    missing_keys = set(required_keys) - set(response.keys())

    if missing_keys:
        raise ValueError(
            f"Missing JSON fields: {sorted(missing_keys)}"
        )

    # Reject booleans and strings, not just non-numbers.
    for key in required_keys:
        value = response[key]

        if type(value) is not int:
            raise ValueError(
                f"{key} must be an integer; received {value!r}"
            )

    if (
        response["experience_years"] < 0
        or not 0 <= response["experience_months"] <= 11
    ):
        raise ValueError("Invalid experience years/months.")

    for key in skill_columns.values():
        if response[key] not in (0, 1):
            raise ValueError(
                f"{key} must be either 0 or 1."
            )

    return response


# ============================================================
# 5. LOAD ALREADY COMPLETED RESUMES
# ============================================================

def load_completed_files(output_csv, output_columns):
    """
    The CSV itself acts as the checkpoint. On restart, previously
    successful resumes are skipped.

    If the key skills changed since the CSV was created, the old
    results no longer match, so all resumes are analyzed again.

    Returns:
        set[str]: Markdown file names already analyzed.
    """

    output_csv = Path(output_csv)

    if output_csv.exists():
        try:
            existing_df = pd.read_csv(
                output_csv,
                encoding=CSV_ENCODING,
                dtype=get_contact_dtypes(),
            )

        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()

        if list(existing_df.columns) == output_columns:
            # Only treat valid, successful records as completed.
            completed = set(
                existing_df.loc[
                    existing_df["file_name"].notna(), "file_name"
                ].astype(str)
            )

            logger.info(
                "Checkpoint found: %d resumes already analyzed.",
                len(completed),
            )

            return completed

        logger.warning(
            "Key skills changed since %s was created. "
            "Re-analyzing all resumes.",
            output_csv.name,
        )

    # Create the CSV with headers before starting processing.
    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(columns=output_columns).to_csv(
            output_csv,
            index=False,
            encoding=CSV_ENCODING,
        )

    except OSError as exc:
        raise OSError(
            f"Could not create analysis CSV {output_csv}: {exc}"
        ) from exc

    return set()


# ============================================================
# 6. PROCESS RESUMES ONE BY ONE
# ============================================================

def analyze_resumes(
    skill_columns,
    resume_folder=MARKDOWN_RESUMES_DIR,
    output_csv=RESUME_ANALYSIS_PATH,
    error_log=RESUME_ERROR_LOG,
):
    """
    Extract contact details, experience and skills from every
    pending Markdown resume.

    Args:
        skill_columns: Skill name -> CSV column (from key_skills).
        resume_folder: Folder containing the Markdown resumes.
        output_csv:    Analysis CSV (also the checkpoint).
        error_log:     File that receives per-resume failures.

    Each successful row is appended to the CSV immediately.

    Returns:
        pandas.DataFrame: The full resume analysis CSV.
    """

    resume_folder = Path(resume_folder)
    output_csv = Path(output_csv)

    error_logger = get_error_logger(__name__, error_log)

    # Step 1: validate the input folder.
    if not resume_folder.exists():
        raise FileNotFoundError(
            f"Resume folder does not exist: {resume_folder}"
        )

    # Step 2: work out which resumes still need analysis.
    output_columns = get_output_columns(skill_columns)

    completed_files = load_completed_files(output_csv, output_columns)

    all_md_files = sorted(resume_folder.rglob("*.md"))

    pending_files = [
        file_path
        for file_path in all_md_files
        if file_path.name not in completed_files
    ]

    logger.info("Total Markdown files found : %d", len(all_md_files))
    logger.info("Already completed          : %d", len(completed_files))
    logger.info("Pending resumes            : %d", len(pending_files))
    logger.info("Output CSV                 : %s", output_csv)

    chain = create_resume_analyzer(skill_columns) if pending_files else None

    # Step 3: analyze each pending resume.
    for index, file_path in enumerate(pending_files, start=1):

        logger.info(
            "[%d/%d] Processing: %s", index, len(pending_files), file_path.name
        )

        try:
            # A. Read the extracted Markdown text
            resume_text = file_path.read_text(encoding="utf-8").strip()

            if not resume_text:
                raise ValueError("Resume Markdown file is empty.")

            # B. Invoke Gemini and parse the JSON response
            response = chain.invoke({
                "resume_text": resume_text,
                "current_date": date.today().strftime("%Y-%m-%d"),
            })

            # C. Validate the returned JSON fields
            response = validate_analysis_response(
                response, skill_columns, resume_text
            )

            # D. Prepare one CSV row
            row = {"file_name": file_path.name}

            for key in CONTACT_KEYS + get_required_keys(skill_columns):
                row[key] = response[key]

            # E. Append immediately after successful execution
            pd.DataFrame([row], columns=output_columns).to_csv(
                output_csv,
                mode="a",
                header=False,
                index=False,
                encoding=CSV_ENCODING,
            )

            # Update the in-memory checkpoint after the CSV write.
            completed_files.add(file_path.name)

            logger.info("SUCCESS: %s", json.dumps(row, ensure_ascii=False))

        except Exception as exc:
            # Log the error but continue with the remaining resumes.
            error_logger.exception(
                "Failed to process %s", file_path.name
            )

            logger.warning(
                "FAILED: %s | %s: %s (can be retried on the next run)",
                file_path.name, type(exc).__name__, exc,
            )

        # F. Optional pause between API requests
        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    # Step 4: summary.
    try:
        final_df = pd.read_csv(
            output_csv,
            encoding=CSV_ENCODING,
            dtype=get_contact_dtypes(),
        )

    except pd.errors.EmptyDataError:
        final_df = pd.DataFrame(columns=output_columns)

    logger.info("=" * 60)
    logger.info("RESUME EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info("Markdown files found : %d", len(all_md_files))
    logger.info("Successfully saved   : %d", len(completed_files))
    logger.info(
        "Still pending/failed : %d",
        len(all_md_files) - len(completed_files),
    )
    logger.info("CSV saved at: %s", output_csv)

    return final_df
