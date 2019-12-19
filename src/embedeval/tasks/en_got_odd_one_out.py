"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by David Staub <david_staub@hotmail.com)
:license: MIT, see LICENSE for more details.
"""

from embedeval.task import Task, TaskReport
from embedeval.logger import get_component_logger

logger = get_component_logger("odd_one_out_task")


class OddOneOutTask(Task):  # type: ignore
    """Represents an Odd One Out Task"""

    NAME = "en-got-odd-one-out"

    def evaluate(self, embedding) -> TaskReport:
        # define the inputs for the Task
        words = "Eddard Catelyn Rob Riverrun Sansa"

        # define the goal for the odd one out function
        goal = "Riverrun"

        # define the title for the report
        report_title = (
            f"Which of the following words is the odd one out?\n" f"    {words}"
        )

        # evaluate odd one out
        odd_one_out = embedding.keyed_vectors.doesnt_match(words.split())

        # evaluate
        if goal != odd_one_out:
            logger.error(
                "Goal %s was not found to be the odd one out, " "instead it was %s",
                goal,
                odd_one_out,
            )
            return TaskReport(
                self.NAME,
                outcome=False,
                title=report_title,
                body=(
                    f"The goal '{goal}' was not found, "
                    f"instead '{odd_one_out}' was found"
                ),
            )

        logger.debug("Found goal %s as the odd one out", odd_one_out)

        return TaskReport(
            self.NAME,
            outcome=True,
            title=report_title,
            body=f"The goal '{goal}' was found",
        )
