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

from embedeval.errors import EmbedevalError
from embedeval.embedding import WordEmbedding


class SimpleWordEmbedding(WordEmbedding):
    """Represents a word2vec specific Word Embedding

    This Word Embedding should only be used for small datasets
    as it's purely implemented in Python and therefore somewhat slow.
    """

    def __init__(self, path, word_vectors):
        self._path = path
        self.word_vectors = word_vectors

    @property
    def path(self) -> Path:
        return self._path

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.word_vectors), self.word_vectors.values()[0].size)

    def get_words(self) -> List[str]:
        return list(self.word_vectors.keys())

    def get_word_vector(self, word: str) -> np.array:
        return self.word_vectors[word]


def load_embedding(path: Path) -> SimpleWordEmbedding:
    """Load the given Word2Vec Word Embedding

    The format for the Embedding expects the n x m matrix size
    in the first row of the text file.

    The current implementation fails, if that's not the case.
    """
    with open(path, "r", encoding="utf-8") as word2vec_file:
        header_line = word2vec_file.readline()
        try:
            word_size, word_vector_size = [int(x) for x in header_line.split()]
        except ValueError as exc:
            if "not enough" in str(exc):
                raise EmbedevalError(
                    "The given Embedding file doesn't contain the N x M "
                    "Embedding size in the header line"
                )
            elif "too many" in str(exc):
                raise EmbedevalError(
                    "The given Embedding file has too many values in the header line"
                )
            elif "invalid literal" in str(exc):
                raise EmbedevalError(
                    "The header line must contain two integers "
                    f"for the size but does: '{header_line}'"
                )
            else:
                raise EmbedevalError(
                    "Unable to extract N x M Embedding size form the header line"
                ) from exc

        word_vectors = {}

        for word_number, line in enumerate(word2vec_file):
            word, *raw_word_vector = line.split()

            word_vector = [np.float32(x) for x in raw_word_vector]

            if len(word_vector) != word_vector_size:
                raise EmbedevalError(
                    f"Promised word vector size {word_vector_size} from header "
                    f"wasn't matched on line {word_number + 2} with a size of {len(word_vector)}"
                )

            word_vectors[word] = word_vector

        if len(word_vectors) < word_size:
            raise EmbedevalError(
                f"Promised word size {word_size} from header "
                f"wasn't matched with a size of {len(word_vectors)}"
            )

        return SimpleWordEmbedding(path, word_vectors)
