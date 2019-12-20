"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by David Staub <david_staub@hotmail.com>
:license: MIT, see LICENSE for more details.
"""

from embedeval.tasks.en_got_odd_one_out import OddOneOutTask
from unittest.mock import MagicMock


def test_odd_one_out_should_fail_on_wrong_return_string():
    # GIVEN
    task = OddOneOutTask()
    embedding_mock = MagicMock(name="embedding")
    embedding_mock.keyed_vectors.doesnt_match.return_value = "Wrong"

    # WHEN
    report = task.evaluate(embedding_mock)

    # THEN
    assert not report.outcome


def test_odd_one_out_should_pass_when_riverrun_is_returned():
    # GIVEN
    task = OddOneOutTask()
    embedding_mock = MagicMock(name="embedding")
    embedding_mock.keyed_vectors.doesnt_match.return_value = "Riverrun"

    # WHEN
    report = task.evaluate(embedding_mock)

    # THEN
    assert report.outcome
