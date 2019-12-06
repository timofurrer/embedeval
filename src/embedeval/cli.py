"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import inspect
import logging
import re
import sys
import warnings
from pathlib import Path

import click
import colorful as cf
from click_default_group import DefaultGroup

from embedeval.errors import EmbedevalError
from embedeval.logger import logger
from embedeval.parsers.word2vec_gensim import load_embedding
from embedeval.taskregistry import load_tasks
from embedeval.taskregistry import registry as task_registry

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


@click.group(
    cls=DefaultGroup, name="embedeval", default="eval", default_if_no_args=False
)
@click.version_option()
@click.help_option("--help", "-h")
def cli():
    """embedeval - NLP Embeddings Evaluation Tool

    Evaluate and generate reports for your NLP Word Embeddings.

    Use the ``eval`` command to evaluate a Word Embedding.
    It's also the default command so use it like the following:

    $ embedeval -t <your-task> embedding.vec
    """
    pass


@cli.command(name="eval")
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
    "path_to_embedding",
    is_eager=True,
    type=click.Path(exists=True, dir_okay=False),
    callback=lambda _, __, p: Path(p),
)
def eval_cli_command(is_debug_mode, path_to_embedding, tasks_path, tasks):
    """Evaluate and generate reports for a NLP Word Embedding (default command)

    The Word Embeddings need to be provided as word2vec keyed vectors in a file.
    The file can either be in a binary format (if the file has the .bin) extension
    or in plain text (if the file has the .vec) extension.
    """
    # load the Word Embedding
    print(cf.italic(f"Loading embedding {path_to_embedding} ..."), flush=True, end=" ")
    is_binary_format = path_to_embedding.suffix == ".bin"
    try:
        embedding = load_embedding(path_to_embedding, binary=is_binary_format)
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
        try:
            report = task.evaluate(embedding)
            print(report, end="\n\n", flush=True)
        except Exception as exc:
            print(
                cf.firebrick(f"Failed to evaluate task: {exc}"), end="\n\n", flush=True
            )
        logger.debug("Evaluated %d of %d Tasks", task_nbr, len(tasks))


@cli.command("tasks")
@click.option(
    "--tasks-path",
    "-p",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Additional Path where Tasks are loaded from",
)
def tasks_cli_command(tasks_path):
    """Show all available tasks"""
    # load all available tasks
    tasks_paths = [__TASKS_DIR__]
    if tasks_path:
        tasks_paths += tasks_path
    load_tasks(tasks_paths)

    # nicely print all available loaded tasks
    print("The following tasks are available for evaluation:")
    for task_name in sorted(task_registry.tasks):
        print(f"    * {cf.bold(task_name)}")


@cli.command("create-task")
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
    "--target-task-path",
    "-t",
    default=Path().absolute(),
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    callback=lambda _, __, p: Path(p),
    help="Target directory for the new Task (default: $PWD)",
)
@click.option(
    "--tasks-path",
    "-p",
    is_eager=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Additional Path where Tasks are loaded from. Those can be used to base on.",
)
@click.option(
    "--based-on",
    "-b",
    help="The name from a built-in Task of which the new Task should be based on",
)
@click.argument("new-task-name")
def create_task_cli_command(
    is_debug_mode, target_task_path, tasks_path, based_on, new_task_name
):
    """Create a new Task for Word Embedding evaluations

    \b
    embedeval ships with a skeleton Task which is used by default.
    Using the ``--based-on`` option changes that skeleton Task to the given Task.

    \b
    Examples:

    \b
        # create a new Task named "rating-analysis" based on the default skeleton Task
        $ embedeval create-task rating-analysis

    \b
        # create a new Task named "rating-analysis" in the directory "tasks" reletive to the execution
        $ embedeval create-task rating-analysis --target-task-path tasks/

    \b
        # create a new Task named "rating-analysis" based on the built-in "offense-detection" Task
        $ embedeval create-task rating-analysis --based-on offense-detection

    \b
        # create a new Task named "rating-analysis-v2"
        # based on a custom "rating-analysis" Task in the "tasks" directory
        $ embedeval create-task rating-analysis-v2 --based-on rating-analysis --tasks-path tasks/
    """  # noqa
    # generate the Python module name for the new Task
    task_module_name = re.sub(r"[^A-Za-z0-9_]", lambda c: "_", new_task_name)
    target_taskfile_path = target_task_path / (task_module_name + ".py")

    if based_on is None:  # based on default Skeleton Task
        base_task_path = __TASKS_DIR__ / "skeleton.py.in"
        with base_task_path.open("r") as skeleton_file:
            target_task_contents = skeleton_file.read().format(
                task_name=new_task_name, task_module_name=task_module_name,
            )
    else:
        # load all available tasks
        tasks_paths = [__TASKS_DIR__]
        if tasks_path:
            tasks_paths += tasks_path
        load_tasks(tasks_paths)

        base_task_cls = next(
            (x for x in task_registry.tasks.values() if x.NAME == based_on), None
        )

        if base_task_cls is None:
            print(f"Error: Task {based_on} is not known to embedeval", file=sys.stderr)
            raise click.Abort()

        base_task_path = Path(inspect.getsourcefile(base_task_cls))
        with base_task_path.open("r") as based_on_file:
            target_task_contents = based_on_file.read()

    with target_taskfile_path.open("w") as target_taskfile:
        target_taskfile.write(target_task_contents)

    print(f"Created the new Task '{new_task_name}' based on {base_task_path}")
    print(f"Make sure to edit and finalize the new Task at {target_taskfile_path}")
