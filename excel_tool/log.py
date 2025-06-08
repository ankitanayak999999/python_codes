# log_setup.py

import sys
import logging

class DualLogger:
    """Option 2: Logs to both file and console using print()."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logging(option=1, log_file="output.log"):
    """
    option 1: Redirect all print() to file only
    option 2: Print to both console and log file
    option 3: Use logging module (with levels and formatting)
    """

    if option == 1:
        sys.stdout = open(log_file, "w", encoding="utf-8")
        print(f"[Option 1] All print() output is redirected to: {log_file}")

    elif option == 2:
        sys.stdout = DualLogger(log_file)
        print(f"[Option 2] Logging to both console and file: {log_file}")

    elif option == 3:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logging.info(f"[Option 3] Logging started. Log file: {log_file}")
    else:
        raise ValueError("Invalid logging option. Choose 1, 2, or 3.")
