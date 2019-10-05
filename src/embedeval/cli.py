"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import click


@click.command(name="embedeval")
@click.version_option()
@click.help_option("--help", "-h")
def cli():
    """embedeval - NLP Embeddings Evaluation Tool

    Evaluate and generate Reports for your
    NLP Word Embeddings.
    """
    print("NLP Embeddings Evaluation Tool")
