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

logging.basicConfig(level=logging.DEBUG)


@click.command(name="embedeval")
@click.version_option()
@click.help_option("--help", "-h")
@click.argument("embedding_path", type=click.Path(exists=True, dir_okay=False))
def cli(embedding_path):
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
