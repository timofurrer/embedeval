"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

from pathlib import Path

from gensim.models import KeyedVectors

from embedeval.embedding import WordEmbedding


def load_word2vec_embedding(path: Path, binary=False) -> WordEmbedding:
    """Load the given Word2Vec Word Embedding using gensim

    The ``gensim.load_word2vec_format`` function is used to parse
    the word2vec Embdding file.
    The ``gensim.models.keyedvectors.KeyedVectors`` is wrapped in the
    embedeval specific ``WordEmbedding`` object.
    """
    import time

    s = time.time()
    word2vec = KeyedVectors.load_word2vec_format(path, binary=binary)
    print(f"KeyedVectors.load_word2vec_format: {time.time() - s}s")

    # copy data from gensim KeyedVectors to WordEmbedding
    s = time.time()
    words = []
    word_vectors = []
    for word in word2vec.vocab.keys():
        words.append(word)
        word_vectors.append(word2vec.word_vec(word))

    print(f"copying shit: {time.time() - s}s")

    return WordEmbedding(path, words, word_vectors)
