"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import textwrap
import uuid

import numpy as np
import pytest

from embedeval.parsers.word2vec_gensim import load_embedding as gensim_load_embedding
from embedeval.parsers.word2vec_simple import load_embedding as simple_load_embedding


def create_tmp_word_embedding(path, embedding_content):
    """Create a temporary Word Embedding file"""
    # FIXME(TF): maybe refactor interface so that file system can be avoided in unit tests.
    created_file = path / str(uuid.uuid4())
    with open(created_file, "w", encoding="utf-8") as embedding_file:
        embedding_file.write(textwrap.dedent(embedding_content).strip())
    return created_file


@pytest.mark.parametrize(
    "load_embedding_func",
    [
        pytest.param(simple_load_embedding, id="simple parser"),
        pytest.param(gensim_load_embedding, id="gensim parser"),
    ],
)
def test_should_parse_word2vec_with_single_entry(load_embedding_func, tmp_path):
    """Loading a Word2Vec Embedding should pass for single word"""
    # GIVEN
    word2vec_path = create_tmp_word_embedding(
        tmp_path,
        """
            1 2
            word 1.0 2.0
        """,
    )

    # WHEN
    embedding = load_embedding_func(word2vec_path)

    # THEN
    assert embedding.get_words() == ["word"]
    assert np.array_equal(embedding.get_word_vector("word"), np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "load_embedding_func",
    [
        pytest.param(simple_load_embedding, id="simple parser"),
        pytest.param(gensim_load_embedding, id="gensim parser"),
    ],
)
def test_should_parse_word2vec_with_multiple_entires(load_embedding_func, tmp_path):
    """Loading a Word2Vec Embedding should pass for multiple word entries"""
    # GIVEN
    word2vec_path = create_tmp_word_embedding(
        tmp_path,
        """
            4 2
            word1 1.0 2.0
            word2 3.0 4.0
            word3 5.0 6.0
            word4 7.0 8.0
        """,
    )

    # WHEN
    embedding = load_embedding_func(word2vec_path)

    # THEN
    assert embedding.get_words() == ["word1", "word2", "word3", "word4"]
    assert np.array_equal(embedding.get_word_vector("word1"), np.array([1.0, 2.0]))
    assert np.array_equal(embedding.get_word_vector("word2"), np.array([3.0, 4.0]))
    assert np.array_equal(embedding.get_word_vector("word3"), np.array([5.0, 6.0]))
    assert np.array_equal(embedding.get_word_vector("word4"), np.array([7.0, 8.0]))
