"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by David Staub <david_staub@hotmail.com>
:license: MIT, see LICENSE for more details.
"""

from embedeval.tasks.en_got_word_analogy import WordAnalogyTask
from unittest.mock import MagicMock


def test_word_analogy_should_fail_on_wrong_return_string():
    # GIVEN
    task = WordAnalogyTask()
    embedding_mock = MagicMock(name="embedding")
    embedding_mock.keyed_vectors.most_similar_cosmul.return_value = {
        "some": 0.89,
        "thing": 0.45,
        "else": 0.43,
    }

    # WHEN
    report = task.evaluate(embedding_mock)

    # THEN
    assert not report.outcome


def test_word_analogy_should_pass_when_lannister_is_returned():
    # GIVEN
    task = WordAnalogyTask()
    embedding_mock = MagicMock(name="embedding")
    embedding_mock.keyed_vectors.most_similar_cosmul.return_value = {
        "some": 0.89,
        "Lannister": 0.45,
        "else": 0.43,
    }

    # WHEN
    report = task.evaluate(embedding_mock)

    # THEN
    assert report.outcome
