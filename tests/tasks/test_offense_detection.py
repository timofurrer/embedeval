"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by David Staub <david_staub@hotmail.com>
:license: MIT, see LICENSE for more details.
"""

import pytest

from embedeval.tasks.de_offense_detection import OffenseDetectionTask


@pytest.fixture(name="offense_detection_task_setup")
def create_offense_detection_task(mocker):
    """This fixture creates an OffenseDetectionTask

    The OffenseDetectionTask has some mocked behavior
    to allow unit testing.
    """
    mocker.patch("embedeval.tasks.de_offense_detection.OffenseDetectionTask._load_dataset")
    mocker.patch(
        "embedeval.tasks.de_offense_detection.OffenseDetectionTask._calculate_sentence_length"
    )  # noqa
    mocker.patch("pandas.concat")
    create_cnn_model_mock = mocker.patch(
        "embedeval.tasks.de_offense_detection.OffenseDetectionTask._create_cnn_model"
    )
    model_mock = mocker.MagicMock(name="CNN model")
    create_cnn_model_mock.return_value = model_mock

    yield OffenseDetectionTask(), model_mock.evaluate


def test_offense_detection_should_pass_with_correct_acc_and_f1_score(
    offense_detection_task_setup, mocker
):
    # given
    offense_detection_task, model_evaluate_mock = offense_detection_task_setup

    mocked_model_loss, mocked_model_acc, mocked_model_f1_score = 0.5, 0.7, 0.75
    model_evaluate_mock.return_value = (
        mocked_model_loss,
        mocked_model_acc,
        mocked_model_f1_score,
    )

    embedding_mock = mocker.MagicMock(name="word embedding")

    # when
    report = offense_detection_task.evaluate(embedding_mock)

    # then
    assert report.outcome


@pytest.mark.parametrize(
    "model_acc, model_f1_score",
    [
        pytest.param(0.1, 1.0, id="Too less accuracy"),
        pytest.param(1.0, 0.1, id="Too less f1 score"),
    ],
)
def test_offense_detection_should_fail_with_weak_performance(
    model_acc, model_f1_score, offense_detection_task_setup, mocker
):
    # given
    offense_detection_task, model_evaluate_mock = offense_detection_task_setup

    model_evaluate_mock.return_value = (0, model_acc, model_f1_score)

    embedding_mock = mocker.MagicMock(name="word embedding")

    # when
    report = offense_detection_task.evaluate(embedding_mock)

    # then
    assert report.outcome is False
