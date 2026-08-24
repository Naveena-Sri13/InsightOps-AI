"""Centralized logger for InsightOps AI project services.

This module configures a single, shared logger instance using only
Python's built-in logging module, with a stream handler suitable for
console output. Other modules may continue to use their own
``logging.getLogger(__name__)`` loggers independently of this module.
"""

import logging

_LOGGER_NAME = "insightops_ai"
_LOG_LEVEL = logging.INFO
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

logger: logging.Logger = logging.getLogger(_LOGGER_NAME)
logger.setLevel(_LOG_LEVEL)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(_LOG_LEVEL)
    _handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(_handler)
    logger.propagate = False