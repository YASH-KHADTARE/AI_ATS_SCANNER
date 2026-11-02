
# ============================================================
# FILE: src/helpers/experience_filter.py
# PURPOSE: Filter candidates by professional experience
#          (pipeline stage 5; 02_filtering.ipynb, step 6)
#
# Adds total_experience_years to the DataFrame and returns a
# boolean mask of candidates inside the HR-defined range.
# ============================================================

import logging

import pandas as pd


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["experience_years", "experience_months"]


def filter_by_experience(
    df,
    minimum_experience_years,
    maximum_experience_years,
):
    """
    Add total_experience_years and return the experience filter.

    total_experience_years = experience_years + experience_months / 12

    Bounds are inclusive and expressed in decimal years.

    Returns:
        pandas.Series: Boolean mask of experience-eligible rows.
    """

    # Step 1: validate inputs.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Experience filter requires columns: {missing}"
        )

    if minimum_experience_years > maximum_experience_years:
        raise ValueError(
            "Minimum experience cannot exceed maximum experience."
        )

    # Step 2: total experience in decimal years.
    years = pd.to_numeric(df["experience_years"], errors="coerce")
    months = pd.to_numeric(df["experience_months"], errors="coerce")

    df["total_experience_years"] = years + months / 12

    # Step 3: inclusive range check (NaN never passes).
    experience_filter = (
        (df["total_experience_years"] >= minimum_experience_years)
        & (df["total_experience_years"] <= maximum_experience_years)
    ).fillna(False)

    logger.info(
        "Experience filter (%.2f-%.2f years): %d eligible, %d excluded.",
        minimum_experience_years,
        maximum_experience_years,
        int(experience_filter.sum()),
        int((~experience_filter).sum()),
    )

    return experience_filter
