
# ============================================================
# FILE: src/helpers/logging_config.py
# PURPOSE: Configure application-wide and per-job logging
#
#   setup_logging()    console + file logging on the root logger;
#                      called once at application startup
#   get_error_logger() logger that additionally writes ERROR records
#                      to a per-job error log file
# ============================================================

import logging
from pathlib import Path

from config import PIPELINE_LOG


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Matches the error-log format used in the notebooks.
ERROR_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


# ============================================================
# 1. APPLICATION LOGGING
# ============================================================

def setup_logging(
    log_level=logging.INFO,
    log_file=PIPELINE_LOG,
):
    """
    Configure console and file logging on the root logger.

    Call this once at application startup, before running the
    pipeline. Calling it again replaces the existing handlers.

    Returns:
        Path: The log file in use.
    """

    log_file = Path(log_file)

    # Step 1: make sure the log folder exists.
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create log directory {log_file.parent}: {exc}"
        ) from exc

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Step 2: remove handlers from a previous call to avoid
    # duplicate output.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Step 3: console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Step 4: file handler.
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")

    except OSError as exc:
        raise OSError(
            f"Could not open log file {log_file}: {exc}"
        ) from exc

    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info("Logging initialized.")
    logging.info("Log file: %s", log_file)

    return log_file


# ============================================================
# 2. PER-JOB ERROR LOGGING
# ============================================================

def get_error_logger(name, error_log):
    """
    Return a logger that also writes ERROR records to error_log.

    The logger is keyed by the error-log path, so each job gets its
    own logger and errors never leak into another job's log file.
    Records still propagate to `name` and the root logger, so they
    also appear on the console and in the application log.
    """

    error_log = Path(error_log)

    try:
        error_log.parent.mkdir(parents=True, exist_ok=True)

    except OSError as exc:
        raise OSError(
            f"Could not create error log directory "
            f"{error_log.parent}: {exc}"
        ) from exc

    # Dots in the key would create spurious logger hierarchy levels.
    log_key = str(error_log.resolve()).replace(".", "_")

    logger = logging.getLogger(f"{name}.{log_key}")

    # Avoid duplicate handlers if called again in the same process.
    for handler in logger.handlers[:]:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == error_log.resolve()
        ):
            return logger

    try:
        file_handler = logging.FileHandler(error_log, encoding="utf-8")

    except OSError as exc:
        raise OSError(
            f"Could not open error log {error_log}: {exc}"
        ) from exc

    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter(ERROR_LOG_FORMAT))

    logger.addHandler(file_handler)

    return logger
