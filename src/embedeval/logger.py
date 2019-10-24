"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import logging


#: Holds the name of the root logger
LOGGER_NAME = "embedeval"

logger = logging.getLogger(LOGGER_NAME)


def get_component_logger(name):
    """Return a component specific logger instance"""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
