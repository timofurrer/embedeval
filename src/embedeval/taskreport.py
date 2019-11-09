"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import textwrap
from dataclasses import dataclass

import colorful as cf


@dataclass(frozen=True)
class TaskReport:
    """Represents an evaluation report of a Task"""

    #: Holds the name of the Task
    name: str
    #: Holds the outcome of the Task
    outcome: bool = False
    #: Holds the title of the Task Report
    title: str = ""
    #: Holds the body of the Task Report
    body: str = ""

    def __str__(self):
        """Format the Report for the console output"""

        formatted_outcome = (
            cf.forestGreen("passed") if self.outcome else cf.firebrick("failed")
        )

        formatted_title = "\n".join(" " * 16 + l for l in self.title.split("\n"))
        formatted_body = "\n".join(" " * 16 + l for l in self.body.split("\n"))

        output = f"""
            The Task {cf.bold(self.name)} {formatted_outcome}:
{formatted_title}
            {cf.bold("Details")}:
{formatted_body}
        """

        return textwrap.dedent(output).strip()
