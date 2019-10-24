"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import typing

from embedeval.task import Task
from embedeval.logger import get_component_logger

logger = get_component_logger("word_analogy_task")


class WordAnalogyTask(Task):
    """Represents a Word Analogy Task"""

    def evaluate(self) -> typing.Optional[str]:
        # define the inputs for the Task
        positives = ["Stark", "Jaime"]
        negatives = ["Eddard"]

        # define the goal for the most similar word analogy
        goal = "Lannister"

        # evaluate most similar word analogies
        most_similar_analogy = dict(
            self.embedding.keyed_vectors.most_similar_cosmul(  # type: ignore
                positive=positives,
                negative=negatives
            )
        )

        # evaluate actual similarity against the set goal
        if goal not in most_similar_analogy:
            logger.error("Goal %s not found in most similar word analogies", goal)
            return None

        logger.debug("Found goal %s with a similarity of %f", goal, most_similar_analogy[goal])

        return f"""
            The Task:
                {positives[0]} is related to {negatives[0]}, as ??? is related to {positives[-1]}
            was successful.
            The Goal of "{goal}" was found with a similarity of {most_similar_analogy[goal]}.
        """
