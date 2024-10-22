import logging

from .constants import LOGGER_KEY


def initialize_logger():
    logger = logging.getLogger(LOGGER_KEY)

    if logger.handlers:  # logger is already setup, don't setup again
        return
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # in the formatter, use the variable "user_ip"
    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    # Setting up the file handler
    file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.addHandler(handler)
