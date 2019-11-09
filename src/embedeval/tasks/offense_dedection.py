"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by David Staub <david_staub@hotmail.com)
:license: MIT, see LICENSE for more details.
"""

from embedeval.task import Task, TaskReport
from embedeval.logger import get_component_logger

logger = get_component_logger("offense_dedection")


class OffenseDedectionTask(Task):  # type: ignore
    """Represents an Offense dedection Task"""

    NAME = "offense-dedection"

    def evaluate(self, embedding) -> TaskReport:
        # define the minimum score to pass the Task
        goal_f2_score = 0.75
        goal_accuracy = 0.75

        # define the title for the report
        report_title = (
            f"Dedect if the defined Tweets are offensive or not."
        )

        # create model

        # train model

        # evaluate model
        actual_accuracy = 0.71934
        actual_f2_score = 0.77823

        if goal_accuracy > actual_accuracy or goal_f2_score > actual_f2_score:
            logger.error(
                "The prediction of the model was not good enough. "
                "The Goal of %.3f accuracy and %.3f F2 score was not reached, "
                "the actual accuracy was %.3f and F2 was %.3f",
                goal_accuracy,
                goal_f2_score,
                actual_accuracy,
                actual_f2_score
            )
            return TaskReport(
                self.NAME,
                outcome=False,
                title=report_title,
                body=(
                    f"""The prediction of the model was not good enough.
The goal of {goal_accuracy:.2} accuracy and {goal_f2_score:.2} F2 score was not reached.
The actual accuracy was {actual_accuracy:.2} and F2 score was {actual_f2_score:.2}."""
                )
            )

        logger.debug(
            "The accuracy of the prediction was %.3f and the F2 score was %.3f",
            actual_accuracy,
            actual_f2_score
        )

        return TaskReport(
            self.NAME,
            outcome=True,
            title=report_title,
            body=(
                f"""The prediction of the model was accurate {actual_accuracy:.2%} of the time.
The goal of minimum {goal_accuracy:.2} accuracy and {goal_f2_score:.2} F2 score was reached.
The actual accuracy was {actual_accuracy:.2} and F2 score was {actual_f2_score:.2}."""
            )
        )
