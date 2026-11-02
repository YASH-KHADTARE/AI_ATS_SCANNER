# AI ATS Scanner

An AI-powered Applicant Tracking System backend. HR uploads a ZIP of PDF
resumes together with the job details; the service extracts each resume,
pulls out structured information with Gemini, filters candidates against
the HR criteria, scores the eligible ones against the Job Description and
returns a ranked list. Everything for a job is stored in its own folder
and can be retrieved later through the API.

Repository: https://github.com/YASH-KHADTARE/AI_ATS_SCANNER

---

## Quick Start (clone and run)

Prerequisites: [Git](https://git-scm.com/downloads), **Python 3.11** and a
[Google Gemini API key](https://aistudio.google.com/app/apikey).

**1. Clone the repository**

```bash
git clone https://github.com/YASH-KHADTARE/AI_ATS_SCANNER.git
cd AI_ATS_SCANNER
```

The code lives on the `feature/ats-api` branch, which is the default, so
the clone already has it checked out. To be explicit:

```bash
git clone -b feature/ats-api https://github.com/YASH-KHADTARE/AI_ATS_SCANNER.git
```

**2. Create and activate a virtual environment**

```powershell
# Windows (PowerShell)
python -m venv ai_ats_scanner
.\ai_ats_scanner\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3.11 -m venv ai_ats_scanner
source ai_ats_scanner/bin/activate
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

Docling downloads its PDF models on first use, so the first run of a job
takes a few extra minutes and needs internet access.

**4. Add your Gemini API key**

Copy the template and fill in the key. `.env` is git-ignored, so it never
gets committed.

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Then open `.env` and set:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

**5. Start the API**

```bash
uvicorn src.main.main:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoints from the browser, or
follow the Postman steps in §7. The first request to `POST /jobs/screen`
creates the `jobs/` folder automatically.

**Troubleshooting**

| Symptom | Fix |
|---|---|
| `GOOGLE_API_KEY is missing` on startup | Step 4: `.env` must sit in the project root next to `readme.md`. |
| `Activate.ps1 cannot be loaded` (PowerShell) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again. |
| `python` is not 3.11 | Install 3.11 and create the venv with `py -3.11 -m venv ai_ats_scanner` (Windows) or `python3.11 -m venv ...`. |

---

## 1. Tech Stack

| Area | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| PDF to text | Docling (PDF -> Markdown, tables preserved) |
| LLM | Google Gemini via LangChain (`langchain-google-genai`) |
| Data | Pandas, CSV + Excel (openpyxl) |
| Config | python-dotenv (`.env` holds `GOOGLE_API_KEY`) |

---

## 2. Project Structure

```
AI_ATS_SCANNER/
├── .env                        GOOGLE_API_KEY=... (git-ignored)
├── .env.example                template for .env
├── requirements.txt            pinned Python dependencies
├── readme.md
├── project_architecture.txt
│
├── config/
│   ├── __init__.py             re-exports config.config
│   └── config.py               paths, model settings, validation,
│                               per-job paths, form-input parsing
│
├── src/
│   ├── main/
│   │   └── main.py             FastAPI app (adds the routers)
│   ├── api/                    one file per route
│   │   ├── screen_job.py       POST /jobs/screen
│   │   ├── list_jobs.py        GET  /jobs
│   │   ├── job_results.py      GET  /jobs/{job_id}/results
│   │   ├── download_results.py GET  /jobs/{job_id}/results/download
│   │   └── common.py           helpers shared by the routes
│   └── helpers/                pipeline stages
│       ├── pipeline.py         orchestrates the stages for one job
│       ├── file_handler.py     ZIP validation and safe extraction
│       ├── candidate_identifier.py  stable candidate IDs (manifest)
│       ├── docling_extractor.py     PDF -> Markdown
│       ├── resume_analyzer.py       Gemini: contact, experience, skills
│       ├── experience_filter.py     experience range filter
│       ├── skills_filter.py         skill threshold filter
│       ├── candidate_scoring.py     Gemini scoring + ranking
│       ├── csv_handler.py           atomic CSV helpers
│       └── logging_config.py        app and per-job error logging
│
├── jobs/                       one folder per executed job (see §5)
│
├── data/                       inputs/outputs used by the notebooks
│   ├── input_data/             Resumes.zip, JD.txt, skills.json, config.json
│   ├── output_data/
│   └── logs/app.log            API application log
│
└── notebook/                   exploratory notebooks (stages 1-6)
```

The `data/input_data` files and `config.load_job_inputs()` are kept for
the notebooks only; the API receives all inputs through the request form.

---

## 3. Setup

Follow the **Quick Start** above for a fresh clone. In short: create a
Python 3.11 virtual environment, run `pip install -r requirements.txt`,
copy `.env.example` to `.env` and set `GOOGLE_API_KEY`.

Model names, temperature, request delay and scoring weights live in
`config/config.py` (`EXTRACTION_MODEL`, `SCORING_MODEL`, `*_WEIGHT`).

---

## 4. Run the API

From the project root:

```powershell
uvicorn src.main.main:app --reload
```

- Interactive docs: http://127.0.0.1:8000/docs
- Health check:     `GET /` -> `{"status": "ok"}`

Startup fails immediately if `GOOGLE_API_KEY` is missing from `.env`.

---

## 5. Pipeline

`POST /jobs/screen` runs these stages synchronously and blocks until the
ranked files are written (expect roughly 15-25 s per resume on CPU).

| # | Stage | Helper | Output |
|---|---|---|---|
| 1 | Validate + unzip (PDF only, other files skipped) | `file_handler` | `resumes/` |
| 2 | Assign stable candidate IDs | `candidate_identifier` | `candidate_manifest.csv` |
| 3 | Convert PDF -> Markdown | `docling_extractor` | `markdown/` |
| 4 | Gemini extraction: name, email, phone, experience, skill flags | `resume_analyzer` | `results/1_resume_analysis.csv` |
| 5 | Python filters: experience range + skill threshold | `experience_filter`, `skills_filter` | `results/2_shortlisted_candidates.csv` |
| 6 | Gemini scoring (quality 20 %, skills/experience 40 %, JD match 40 %), Python final score + sort | `candidate_scoring` | `results/3_ranked_candidates.csv` |
| 7 | Finalize: Excel export, copy shortlisted PDFs | `pipeline` | `results/3_ranked_candidates.xlsx`, `shortlisted_resumes/` |

Only the skills passed in `key_skills` go into the Gemini prompt and
become `skill_<name>` columns. Stages 4 and 6 checkpoint after every
resume, so a failed resume is logged and the others continue.

### Job workspace

```
jobs/<JOB_ID>/
├── input/
│   ├── resumes.zip             uploaded archive
│   └── job.json                all form inputs as received
├── resumes/                    extracted PDFs
├── markdown/                   Docling output
├── results/
│   ├── 1_resume_analysis.csv
│   ├── 2_shortlisted_candidates.csv
│   ├── 3_ranked_candidates.csv
│   └── 3_ranked_candidates.xlsx
├── shortlisted_resumes/        PDFs of candidates who passed the filters
├── logs/
│   ├── resume_analysis_errors.log
│   └── candidate_scoring_errors.log
├── candidate_manifest.csv      candidate_id <-> source PDF (sha256)
└── status.json                 stage, counts, skipped files, errors
```

Job IDs are generated by the server: `JOB_<yyyymmdd>_<hhmmss>_<hex>`.

---

## 6. API Reference

### `POST /jobs/screen` — upload and run

`multipart/form-data`

| Field | Type | Example | Notes |
|---|---|---|---|
| `resumes` | file | `resumes.zip` | PDF resumes; nested folders allowed, non-PDFs skipped |
| `job_position` | text | `AI/ML Engineer` | |
| `job_description` | text | full JD text | used by the scoring prompt |
| `key_skills` | text | `Python, SQL, Machine Learning, FastAPI` | comma-separated; these become the skill columns |
| `skill_threshold` | text | `2/3` | fraction (`2/3`), ratio (`0.5`) or count (`2`) |
| `min_experience_years` | int | `2` | |
| `min_experience_months` | int | `4` | 0-11 |
| `max_experience_years` | int | `5` | |
| `max_experience_months` | int | `0` | 0-11; min must not exceed max |

Response `200`:

```json
{
  "job_id": "JOB_20261102_173850_7D07",
  "job_folder": "F:\\AI_ATS_SCANNER\\jobs\\JOB_20261102_173850_7D07",
  "inputs": { "job_position": "...", "key_skills": ["..."], "...": "..." },
  "summary": {
    "status": "completed", "stage": "Completed",
    "resumes_total": 2, "skipped_files": [], "markdown_failed": [],
    "analyzed": 2, "analysis_failed": 0,
    "shortlisted": 2, "ranked": 2, "scoring_failed": 0,
    "shortlisted_resumes": 2
  },
  "candidates": [
    {
      "rank": 1,
      "candidate_id": "CAND_000002",
      "file_name": "Harshit_Jhalani_Resume.md",
      "source_file": "Harshit_Jhalani_Resume.pdf",
      "candidate_name": "Harshit Jhalani",
      "email": "harshit@example.com",
      "phone": "+91 9079059068",
      "experience_years": 1, "experience_months": 7,
      "total_experience_years": 1.58,
      "skills_matched": 4,
      "skills": { "SQL": 1, "ML": 1, "Python": 1, "DL": 0, "Stats": 1 },
      "resume_quality_score": 85,
      "skills_projects_experience_score": 74,
      "jd_matching_score": 76,
      "final_score": 77.0
    }
  ]
}
```

Errors: `400` invalid form value / not a ZIP / corrupt ZIP / no PDFs
inside; `500` pipeline failure (the `job_id` is included so the logs in
`jobs/<JOB_ID>/logs/` can be inspected).

### `GET /jobs` — list executed jobs

```json
[
  { "job_id": "JOB_20261102_173850_7D07", "job_position": "Data Scientist",
    "status": "completed", "created_at": "2026-11-02T17:38:50",
    "updated_at": "2026-11-02T17:39:37" }
]
```

### `GET /jobs/{job_id}/results` — ranked candidates

Same `candidates` array as above plus `job_id`, `job_position`, `status`.
`404` unknown job, `409` if the job is not `completed`.

### `GET /jobs/{job_id}/results/download` — ranked Excel

Returns `<JOB_ID>_ranked_candidates.xlsx` (sheet "Ranked Candidates").
Same error codes as `/results`.

---

## 7. Testing with Postman

1. `POST http://127.0.0.1:8000/jobs/screen`, Body -> `form-data`, add the
   fields from §6 and set `resumes` to type **File**.
2. Copy `job_id` from the response.
3. `GET /jobs`, `GET /jobs/{job_id}/results`,
   `GET /jobs/{job_id}/results/download` (use *Send and Download*).

---

## 8. Notebooks

`notebook/01_data_extraction.ipynb`, `02_filtering.ipynb`,
`03_scoring.ipynb` contain the original exploratory versions of stages
1-6 and read their inputs from `data/input_data/`. The helpers in
`src/helpers/` are the production versions of the same logic.
