"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import colorful as cf

from embedeval.task import Task, TaskReport
from embedeval.logger import get_component_logger

logger = get_component_logger("word_analogy_task")


class WordAnalogyTask(Task):  # type: ignore
    """Represents a Word Analogy Task"""

    NAME = "en-got-word-analogy"

    def evaluate(self, embedding) -> TaskReport:
        # define the inputs for the Task
        positives = ["Stark", "Jaime"]
        negatives = ["Eddard"]

        # define the goal for the most similar word analogy
        goal = "Lannister"

        # define the title for the report
        report_title = (
            f"{positives[0]} is related to {negatives[0]}, "
            f"as ??? is related to {positives[-1]}"
        )

        # evaluate most similar word analogies
        most_similar_analogy = dict(
            embedding.keyed_vectors.most_similar_cosmul(  # type: ignore
                positive=positives, negative=negatives
            )
        )

        # evaluate actual similarity against the set goal
        if goal not in most_similar_analogy:

            result_list = "\n".join(
                f"    {k}: {v:.2}" for k, v in most_similar_analogy.items()
            )

            logger.error("Goal %s not found in most similar word analogies", goal)
            return TaskReport(
                self.NAME,
                outcome=False,
                title=report_title,
                body=f"""The goal of '{goal}' was not found.
The following dictionary was returned:
{result_list}""",
            )

        logger.debug(
            "Found goal %s with a similarity of %f", goal, most_similar_analogy[goal]
        )

        return TaskReport(
            self.NAME,
            outcome=True,
            title=report_title,
            body=(
                f"The Goal of '{goal}' was found with a "
                f"{cf.bold}similarity of {most_similar_analogy[goal]:.2}{cf.reset}."
            ),
        )
