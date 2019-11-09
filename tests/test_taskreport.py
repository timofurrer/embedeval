"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import pytest

import colorful as cf

from embedeval.taskreport import TaskReport


@pytest.fixture(autouse=True)
def disable_coloring():
    cf.disable()


def test_taskreport_should_default_initialize_as_failed():
    # WHEN
    report = TaskReport("task")

    # THEN
    assert report.outcome is False


def test_taskreport_should_be_able_to_access_all_members():
    # GIVEN
    report = TaskReport("name", outcome=False, title="title", body="body")

    # THEN
    assert report.name == "name"
    assert report.outcome is False
    assert report.title == "title"
    assert report.body == "body"


def test_taskreport_should_be_formatted_in_str():
    # GIVEN
    report = TaskReport("name", outcome=False, title="title", body="body")

    # WHEN
    str_report = str(report)

    # THEN
    assert str_report == (
        """
The Task name failed:
    title
Details:
    body
    """.strip()
    )
