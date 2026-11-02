
# ============================================================
# FILE: src/helpers/skills_filter.py
# PURPOSE: Filter candidates by key-skill matches
#          (pipeline stage 5; 02_filtering.ipynb, step 7)
#
# Adds skills_matched (count of skill columns equal to 1) and
# returns a boolean mask of candidates meeting the HR threshold,
# expressed as a minimum count, a minimum ratio, or both.
# ============================================================

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def filter_by_skills(
    df,
    skill_columns,
    minimum_matched_skills=None,
    minimum_skill_match_ratio=None,
):
    """
    Add skills_matched and return the skills filter.

    skills_matched counts the skill columns equal to 1.
    If both thresholds are supplied, both must pass.

    Returns:
        pandas.Series: Boolean mask of skills-eligible rows.
    """

    # Step 1: validate inputs.
    if (
        minimum_matched_skills is None
        and minimum_skill_match_ratio is None
    ):
        raise ValueError(
            "Configure at least one skill-filter threshold."
        )

    if not skill_columns:
        raise ValueError("At least one skill column is required.")

    missing = [col for col in skill_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Skills filter requires columns: {missing}"
        )

    # Step 2: count matched skills per candidate.
    df["skills_matched"] = (
        df[skill_columns]
        .apply(pd.to_numeric, errors="coerce")
        .eq(1)
        .sum(axis=1)
    )

    # Step 3: apply the configured threshold(s).
    skills_filter = pd.Series(True, index=df.index)

    if minimum_matched_skills is not None:
        skills_filter &= df["skills_matched"] >= minimum_matched_skills

    if minimum_skill_match_ratio is not None:
        skills_filter &= (
            df["skills_matched"] / len(skill_columns)
            >= minimum_skill_match_ratio
        )

    logger.info(
        "Skills filter (min count=%s, min ratio=%s of %d skills): "
        "%d eligible, %d excluded.",
        minimum_matched_skills,
        minimum_skill_match_ratio,
        len(skill_columns),
        int(skills_filter.sum()),
        int((~skills_filter).sum()),
    )

    return skills_filter
