"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by David Staub <david_staub@hotmail.com>
:license: MIT, see LICENSE for more details.
"""

from embedeval.tasks.offense_detection import OffenseDetectionTask
from unittest.mock import MagicMock


def test_offense_detection_should_pass():
    # GIVEN
    task = OffenseDetection()
    embedding_mock = MagicMock(name="embedding")
    embedding_mock.get_word_vector.side_effect = KeyError("Mock")

    # WHEN
    report = task.evaluate(embedding_mock)

    # THEN
    assert report.outcome
