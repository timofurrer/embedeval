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
    """Base Class for the Task API

    Subclass this Task to automatically register
    an evaluation Task to the Task Registry.

    The Task Evaluation Algorithm must be implemented
    in the ``evaluate()`` method.
    """
    def __init_subclass__(cls, **kwargs):
        """Registry subclasses to the Task Registry for later discovery"""
        super().__init_subclass__(**kwargs)
        task_registry.register(cls)

    @abstractmethod
    def evaluate(self, embedding: WordEmbedding) -> typing.Optional[str]:
        """Evaluate this Task on the given Word Embedding

        The evaluation algorithm should always produce some kind of
        comparable statistics or measures which can be
        provided to the user to verify the quality of the
        given Word Embedding.

        This measure must be returned as a string from this method.

        It should contain everything needed by the user to verify the Embedding.
        """
        ...
