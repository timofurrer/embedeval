"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

from typing import List
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class WordEmbedding:
    """Immutable representation of a loaded Word Embedding

    A Word Embedding always consists of a one-dimensional
    vector of words and a n-dimensional vector representing
    the position in the vector space for each word.
    """
    path: Path
    words: List[str]
    word_vectors: np.array
