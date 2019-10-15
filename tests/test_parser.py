"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import textwrap
import uuid

import pytest
import numpy as np

from embedeval.errors import EmbedevalError
from embedeval.parsers.word2vec_simple import load_word2vec_text_embedding


def create_tmp_word_embedding(path, embedding_content):
    """Create a temporary Word Embedding file"""
    # FIXME(TF): maybe refactor interface so that file system can be avoided in unit tests.
    created_file = path / str(uuid.uuid4())
    with open(created_file, "w", encoding="utf-8") as embedding_file:
        embedding_file.write(textwrap.dedent(embedding_content.strip()))
    return created_file


@pytest.mark.parametrize("embedding, expected_error_msg", [
    pytest.param(
        "",
        "The given Embedding file doesn't contain the N x M Embedding size in the header line",
        id="Empty Embedding file"
    ),
    pytest.param(
        "1",
        "The given Embedding file doesn't contain the N x M Embedding size in the header line",
        id="Only one size value given"
    ),
    pytest.param(
        "1 2 3",
        "The given Embedding file has too many values in the header line",
        id="No header line given or too many values in header line"
    )
])
def test_should_fail_if_missing_header_in_word2vec(embedding, expected_error_msg, tmp_path):
    """Loading a Word2Vec Embedding should fail if the header line is missing"""
    # GIVEN
    word2vec_path = create_tmp_word_embedding(tmp_path, embedding)

    # THEN
    with pytest.raises(EmbedevalError, match=expected_error_msg):
        # WHEN
        load_word2vec_text_embedding(word2vec_path)


def test_should_fail_if_not_integers_in_header_in_word2vec(tmp_path):
    """Loading a Word2Vec Embedding should fail if the header line doesn't contain two integers"""
    # GIVEN
    word2vec_path = create_tmp_word_embedding(tmp_path, "a b")

    # THEN
    expected_error_msg = "The header line must contain two integers for the size but does: 'a b'"
    with pytest.raises(EmbedevalError, match=expected_error_msg):
        # WHEN
        load_word2vec_text_embedding(word2vec_path)


def test_should_parse_word2vec_with_single_entry(tmp_path):
    """Loading a Word2Vec Embedding should pass for single word"""
    # GIVEN
    word2vec_path = create_tmp_word_embedding(
        tmp_path,
        """
            1 2
            word 1.0 2.0
        """
    )

    # WHEN
    embedding = load_word2vec_text_embedding(word2vec_path)

    # THEN
    assert embedding.words == ["word"]
    assert np.array_equal(embedding.word_vectors, np.array([[1.0, 2.0]]))


def test_should_parse_word2vec_with_multiple_entires(tmp_path):
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
        """
    )

    # WHEN
    embedding = load_word2vec_text_embedding(word2vec_path)

    # THEN
    assert embedding.words == ["word1", "word2", "word3", "word4"]
    assert np.array_equal(embedding.word_vectors, np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ]))


def test_should_fail_to_parse_word2vec_if_word_vector_size_mismatch(tmp_path):
    """Loading a Word2Vec Embedding should fail if the word vector sizes mismatch"""
    # GIVEN
    word2vec_path = create_tmp_word_embedding(
        tmp_path,
        """
            1 2
            word 1.0
        """
    )

    # THEN
    expected_error_msg = (
        "Promised word vector size 2 from header "
        "wasn't matched on line 2 with a size of 1"
    )
    with pytest.raises(EmbedevalError, match=expected_error_msg):
        # WHEN
        load_word2vec_text_embedding(word2vec_path)


def test_should_fail_to_parse_word2vec_if_word_size_too_high(tmp_path):
    """Loading a Word2Vec Embedding should fail if the word size is too high"""
    # GIVEN
    word2vec_path = create_tmp_word_embedding(
        tmp_path,
        """
            1 2
        """
    )

    # THEN
    expected_error_msg = (
        "Promised word size 1 from header wasn't matched with a size of 0"
    )
    with pytest.raises(EmbedevalError, match=expected_error_msg):
        # WHEN
        load_word2vec_text_embedding(word2vec_path)


def test_should_fail_to_parse_word2vec_if_word_size_too_low(tmp_path):
    """Loading a Word2Vec Embedding should fail if the word size is too low"""
    # GIVEN
    word2vec_path = create_tmp_word_embedding(
        tmp_path,
        """
            2 2
            word 1.0 2.0
        """
    )

    # THEN
    expected_error_msg = (
        "Promised word size 2 from header wasn't matched with a size of 1"
    )
    with pytest.raises(EmbedevalError, match=expected_error_msg):
        # WHEN
        load_word2vec_text_embedding(word2vec_path)
