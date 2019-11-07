"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by David Staub <david_staub@hotmail.com)
:license: MIT, see LICENSE for more details.
"""

import typing

import colorful as cf

from embedeval.task import Task
from embedeval.logger import get_component_logger

logger = get_component_logger("odd_one_out_task")


class OddOneOutTask(Task):  # type: ignore
    """Represents an Odd One Out Task"""

    NAME = "odd-one-out"

    def evaluate(self, embedding) -> typing.Optional[str]:
        # define the inputs for the Task
        words = "Eddard Catelyn Rob Riverrun Sansa"

        # define the goal for the odd one out function
        goal = "Lannister"

        # evaluate odd one out
        odd_one_out = embedding.keyed_vectors.doesnt_match(words.split())

        # evaluate
        if goal != odd_one_out:
            logger.error("Goal %s was not found to be the odd one out, "
                         "instead it was %s", goal, odd_one_out)
            return None

        logger.debug("Found goal %s as the odd one out", odd_one_out)

        return f"""
            {cf.bold}The Task{cf.reset}:
                Which of the following words is the odd one out?
                {words}
            was {cf.underlined}successful{cf.reset}.
            The Goal "{goal}" was found{cf.reset}.
        """  # noqa
