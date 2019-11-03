"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import textwrap
import logging

import click

from embedeval.logger import logger
from embedeval.parsers.word2vec_gensim import load_embedding
from embedeval.tasks.word_analogy import WordAnalogyTask

logging.basicConfig(
    level=logging.CRITICAL, format="%(asctime)s - %(name)s [%(levelname)s]: %(message)s"
)


def enable_debug_mode(ctx, param, enabled):
    """Enable the logging module to log in debug mode

    The logging is enabled if the --debug option
    is given.
    """
    if enabled:
        logger.setLevel(logging.DEBUG)
        logger.debug("Enabled debug mode")
    else:
        logger.setLevel(logging.ERROR)


@click.command(name="embedeval")
@click.version_option()
@click.help_option("--help", "-h")
@click.option(
    "--debug",
    "-d",
    "is_debug_mode",
    is_flag=True,
    help="Enable debug mode",
    callback=enable_debug_mode,
)
@click.argument("embedding_path", type=click.Path(exists=True, dir_okay=False))
def cli(is_debug_mode, embedding_path):
    """embedeval - NLP Embeddings Evaluation Tool

    Evaluate and generate Reports for your
    NLP Word Embeddings.
    """
    logger.debug("Loading embedding %s", embedding_path)
    embedding = load_embedding(embedding_path, binary=True)
    logger.debug("Loaded embedding")

    task = WordAnalogyTask(embedding)
    result = task.evaluate()

    formatted_result = textwrap.dedent(result).strip()
    print(formatted_result)
