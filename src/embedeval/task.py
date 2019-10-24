"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import typing
from abc import ABC, abstractmethod

from embedeval.embedding import WordEmbedding


class Task(ABC):
    """Interface for the Task API"""
    def __init__(self, embedding: WordEmbedding) -> None:
        self.embedding = embedding

    @abstractmethod
    def evaluate(self) -> typing.Optional[str]:
        """Evaluate this Task"""
        ...
