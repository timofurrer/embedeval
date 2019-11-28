"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import sys
import logging
import warnings
from pathlib import Path

import click
import colorful as cf

from embedeval.logger import logger
from embedeval.parsers.word2vec_gensim import load_embedding
from embedeval.taskregistry import registry as task_registry, load_tasks
from embedeval.errors import EmbedevalError

logging.basicConfig(
    level=logging.CRITICAL, format="%(asctime)s - %(name)s [%(levelname)s]: %(message)s"
)

# suppress all warnings when running the application
warnings.simplefilter("ignore")

#: Holds the path to the tasks directory deployed with embedeval
__TASKS_DIR__ = Path(__file__).parent / "tasks"


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


def create_tasks(ctx, param, task_names):
    """Create the given Tasks"""
    tasks_paths = [__TASKS_DIR__]
    if ctx.params["tasks_path"] is not None:
        logger.debug(
            "Using additional path to load tasks: %s", ctx.params["tasks_path"]
        )
        tasks_paths.insert(0, Path(ctx.params["tasks_path"]))

    # load all tasks deployed with embedeval
    load_tasks(tasks_paths)

    # create Tasks with the given Embedding
    try:
        tasks = [task_registry.create_task(n) for n in task_names]
    except EmbedevalError as exc:
        print(f"{cf.bold_firebrick('Error:')} {cf.firebrick(exc)}", file=sys.stderr)
        raise click.Abort()
    return tasks


@click.command(name="embedeval")
@click.version_option()
@click.help_option("--help", "-h")
@click.option(
    "--debug",
    "-d",
    "is_debug_mode",
    is_flag=True,
    is_eager=True,
    callback=enable_debug_mode,
    help="Enable debug mode",
)
@click.option(
    "--tasks-path",
    "-p",
    is_eager=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Additional Path where Tasks are loaded from",
)
@click.option(
    "--task",
    "-t",
    "tasks",
    multiple=True,
    callback=create_tasks,
    help="The Task to evaluate on the given Embedding (can be specified multiple times)",
)
@click.argument(
    "embedding_path", is_eager=True, type=click.Path(exists=True, dir_okay=False)
)
def cli(is_debug_mode, embedding_path, tasks_path, tasks):
    """embedeval - NLP Embeddings Evaluation Tool

    Evaluate and generate Reports for your
    NLP Word Embeddings.

    The Word Embeddings need to be provided as word2vec keyed vectors in a file.
    The file can either be in a binary format (if the file has the .bin) extension
    or in plain text (if the file has the .vec) extension.
    """
    # load the Word Embedding
    print(cf.italic(f"Loading embedding {embedding_path} ..."), flush=True, end=" ")
    is_binary_format = embedding_path.suffix == ".bin"
    try:
        embedding = load_embedding(embedding_path, binary=is_binary_format)
    except EmbedevalError as exc:
        print(cf.bold_firebrick("[FAILED]"), flush=True, end="\n\n")
        print(f"{cf.bold_firebrick('Error:')} {cf.firebrick(exc)}", file=sys.stderr)
        raise click.Abort()
    else:
        print(cf.bold("[OK]"), flush=True, end="\n\n")

    # evaluate all tasks
    logger.debug("Evaluating %d Tasks ...", len(tasks))
    for task_nbr, task in enumerate(tasks, start=1):
        logger.debug("Evaluating Task %s ...", task.NAME)
        report = task.evaluate(embedding)
        print(report, end="\n\n", flush=True)
        logger.debug("Evaluated %d of %d Tasks", task_nbr, len(tasks))
