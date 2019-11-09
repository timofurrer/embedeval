"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import logging
from pathlib import Path

from click.testing import CliRunner
import pytest

import embedeval
from embedeval import __version__
from embedeval.cli import cli, logger


@pytest.fixture(name="existing_embed_file")
def create_embed_file(tmpdir):
    embed_filepath = tmpdir / "embed.vec"
    embed_filepath.write("0 0")
    yield str(embed_filepath)


def test_cli_should_output_version_to_stdout_when_asked():
    # GIVEN
    runner = CliRunner()

    # WHEN
    result = runner.invoke(cli, ["--version"])

    # THEN
    assert result.exit_code == 0
    assert result.output == f"embedeval, version {__version__}\n"


def test_cli_should_enable_debug_logs_in_debug_mode():
    # GIVEN
    runner = CliRunner()

    # WHEN
    runner.invoke(cli, ["--debug"])

    # THEN
    assert logger.level == logging.DEBUG


def test_cli_should_load_tasks_from_default_location(existing_embed_file, mocker):
    # GIVEN
    runner = CliRunner()
    load_tasks_mock = mocker.patch("embedeval.cli.load_tasks")
    expected_default_path = Path(embedeval.__file__).absolute().parent / "tasks"

    # WHEN
    runner.invoke(cli, [existing_embed_file, "--task", "foo"])

    # THEN
    load_tasks_mock.assert_called_once_with([expected_default_path])


def test_cli_should_load_additional_tasks_dir_prior_to_default(
    existing_embed_file, tmpdir, mocker
):
    # GIVEN
    runner = CliRunner()
    load_tasks_mock = mocker.patch("embedeval.cli.load_tasks")
    given_additional_tasks_dir = Path(str(tmpdir))

    # WHEN
    runner.invoke(
        cli,
        [
            existing_embed_file,
            "--tasks-path",
            str(given_additional_tasks_dir),
            "--task",
            "foo",
        ],
    )

    # THEN
    load_tasks_mock.assert_called_once_with([given_additional_tasks_dir, mocker.ANY])


def test_cli_should_create_given_single_task(existing_embed_file, mocker):
    # GIVEN
    runner = CliRunner()
    mocker.patch("embedeval.cli.load_tasks")
    taskregistry_create_task_mock = mocker.patch(
        "embedeval.cli.task_registry.create_task"
    )

    # WHEN
    runner.invoke(cli, [existing_embed_file, "--task", "foo"])

    # THEN
    taskregistry_create_task_mock.assert_called_once_with("foo")


def test_cli_should_create_all_given_tasks(existing_embed_file, mocker):
    # GIVEN
    runner = CliRunner()
    mocker.patch("embedeval.cli.load_tasks")
    taskregistry_create_task_mock = mocker.patch(
        "embedeval.cli.task_registry.create_task"
    )

    # WHEN
    runner.invoke(
        cli, [existing_embed_file, "--task", "foo", "--task", "bar", "--task", "meh"]
    )

    # THEN
    taskregistry_create_task_mock.assert_has_calls(
        [mocker.call("foo"), mocker.call("bar"), mocker.call("meh")]
    )
