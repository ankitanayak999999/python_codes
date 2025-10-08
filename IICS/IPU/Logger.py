import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(
    name: str = "app",
    log_file: str = "app.log",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
    """
    Generic reusable logger.
    - Logs to both console and rotating file
    - No hardcoding: caller passes name, path, and level
    - Safe for repeated calls (no duplicate handlers)
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers in repeated imports/runs
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Ensure folder exists for the log file
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file
    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
