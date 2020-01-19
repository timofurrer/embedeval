"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

__license__ = "MIT"
__version__ = "1.0.1"


# Expose useful objects on package level
from embedeval.task import Task  # noqa
from embedeval.taskreport import TaskReport  # noqa
from embedeval.errors import EmbedevalError  # noqa
