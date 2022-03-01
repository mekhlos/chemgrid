import logging
import sys


def setup_logger(name, level="INFO"):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(level)

    return logger
