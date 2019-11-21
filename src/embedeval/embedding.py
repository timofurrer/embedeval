"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from pathlib import Path

import numpy as np


class WordEmbedding(ABC):
    """Representation of a loaded immutable Word Embedding

    This interface should be implemented to represent
    concrete parsed Word Embeddings of a particular
    type.

    A Word Embedding always consists of a one-dimensional
    vector of words and a n-dimensional vector representing
    the position in the vector space for each word.
    """

    @property
    @abstractmethod
    def path(self) -> Path:
        """Get the path to the Word Embedding file"""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the Embedding.

        The first value in the tuple is the amount of words
        and the second value the vector size of each word.
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_words(self) -> List[str]:
        """Get a list of all words in the Word Embedding"""
        ...  # pragma: no cover

    @abstractmethod
    def get_word_vector(self, word: str) -> np.array:
        """Get the word vector for the given word from Word Embedding"""
        ...  # pragma: no cover
