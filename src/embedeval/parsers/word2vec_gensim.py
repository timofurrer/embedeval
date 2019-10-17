"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

from typing import List
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from embedeval.embedding import WordEmbedding


class KeyedVectorsWordEmbedding(WordEmbedding):
    """Represents a Gensim KeyedVectors specific Word Embedding"""
    def __init__(self, path, keyed_vectors):
        self._path = path
        self.keyed_vectors = keyed_vectors

    @property
    def path(self) -> Path:
        return self._path

    def get_words(self) -> List[str]:
        return list(self.keyed_vectors.vocab.keys())

    def get_word_vector(self, word: str) -> np.array:
        return self.keyed_vectors.word_vec(word)


def load_embedding(path: Path, binary=False) -> KeyedVectorsWordEmbedding:
    """Load the given Word2Vec Word Embedding using gensim

    The ``gensim.load_word2vec_format`` function is used to parse
    the word2vec Embdding file.
    The ``gensim.models.keyedvectors.KeyedVectors`` is wrapped in the
    embedeval specific ``WordEmbedding`` object.
    """
    keyed_vectors = KeyedVectors.load_word2vec_format(path, binary=binary)

    return KeyedVectorsWordEmbedding(path, keyed_vectors)
