"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

from pathlib import Path

import numpy as np

from embedeval.errors import EmbedevalError
from embedeval.embedding import WordEmbedding


def load_word2vec_text_embedding(path: Path) -> WordEmbedding:
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

        words = []
        word_vectors = np.zeros(shape=(word_size, word_vector_size), dtype=np.float32)

        for word_number, line in enumerate(word2vec_file):
            word, *raw_word_vector = line.split()

            word_vector = [
                np.float32(x)
                for x
                in raw_word_vector
            ]

            if len(word_vector) != word_vector_size:
                raise EmbedevalError(
                    f"Promised word vector size {word_vector_size} from header "
                    f"wasn't matched on line {word_number + 2} with a size of {len(word_vector)}"
                )

            words.append(word)
            word_vectors[word_number] = word_vector
            print("DONE: ", word_number)

        if len(words) < word_size:
            raise EmbedevalError(
                f"Promised word size {word_size} from header "
                f"wasn't matched with a size of {len(words)}"
            )

        return WordEmbedding(path, words, word_vectors)
