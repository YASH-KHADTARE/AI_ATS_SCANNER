
# ============================================================
# FILE: config/__init__.py
# PURPOSE: Expose config/config.py as the `config` package so
#          `from config import ...` keeps working across the project
# ============================================================

from config.config import *  # noqa: F401,F403
