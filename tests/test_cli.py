"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

from click.testing import CliRunner

from embedeval import __version__
from embedeval.cli import cli


def test_cli_should_output_version_to_stdout_when_asked():
    # GIVEN
    runner = CliRunner()

    # WHEN
    result = runner.invoke(cli, ["--version"])

    # THEN
    assert result.exit_code == 0
    assert result.output == f"embedeval, version {__version__}\n"
