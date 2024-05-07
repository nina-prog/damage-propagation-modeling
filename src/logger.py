""" This module contains the logger setup for the project. """
import logging
import sys
import colorlog


formatter = colorlog.ColoredFormatter(
    "%(asctime)s [%(blue)s%(name)s:%(lineno)s%(reset)s] [%(log_color)s%(levelname)s%(reset)s] >>>> %(message)s",
    log_colors={ # 'DEBUG': cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
stream_handler = colorlog.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)


def setup_logger(name, level=logging.DEBUG):
    """Set up a logger with the given name and level.

    :param name: The name of the logger.
    :param level: The level of the logger.
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    return logger
