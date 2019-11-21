"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

from typing import List, Tuple
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from embedeval.embedding import WordEmbedding
from embedeval.errors import EmbedevalError


class KeyedVectorsWordEmbedding(WordEmbedding):
    """Represents a word2vec KeyedVectors specific Word Embedding

    The word2vec file will be parsed by ``gensim``.

    The gensim ``KeyedVectors`` instance is made available
    in the ``self.keyed_vectors`` attribute.
    """

    def __init__(self, path, keyed_vectors):
        self._path = path
        #: Holds the gensim KeyedVectors instance
        self.keyed_vectors = keyed_vectors

    @property
    def path(self) -> Path:
        return self._path  # pragma: no cover

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.keyed_vectors.vectors), self.keyed_vectors.vector_size)

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
    try:
        keyed_vectors = KeyedVectors.load_word2vec_format(
            path, binary=binary, unicode_errors="ignore"
        )
    except Exception as exc:
        raise EmbedevalError(
            f"Failed to parse Embedding with gensim KeyedVectors: {exc}"
        )

    return KeyedVectorsWordEmbedding(path, keyed_vectors)
