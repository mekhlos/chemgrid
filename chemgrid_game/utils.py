import logging
import sys

from datetime import datetime


def get_datetime_str() -> str:
    str_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return str_time


def setup_logger(name, level="INFO"):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(level)

    return logger
