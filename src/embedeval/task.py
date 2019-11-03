"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import typing
from abc import ABC, abstractmethod

from embedeval.taskregistry import registry as task_registry
from embedeval.embedding import WordEmbedding


class Task(ABC):
    """Interface for the Task API"""
    def __init_subclass__(cls, **kwargs):
        """Registry subclasses to the Task Registry for later discovery"""
        super().__init_subclass__(**kwargs)
        task_registry.register(cls)

    @abstractmethod
    def evaluate(self, embedding: WordEmbedding) -> typing.Optional[str]:
        """Evaluate this Task"""
        ...
