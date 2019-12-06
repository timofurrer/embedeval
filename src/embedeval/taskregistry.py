"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

from embedeval.errors import EmbedevalError
from embedeval.logger import get_component_logger

logger = get_component_logger("task-registry")


def load_tasks(locations: List[Path]) -> List[Path]:
    """Load all Python modules in the given locations

    All given locations must already be expanded regarding
    * Environment Variables
    * User Home Directory
    """
    loaded_modules = []
    for location in locations:
        if not location.exists():
            raise EmbedevalError(
                f"Location '{location}' to load modules does not exist"
            )

        for module_path in location.glob("**/*.py"):
            load_module(module_path)
            loaded_modules.append(module_path)

    return loaded_modules


def load_module(path: Path) -> None:
    """Load the given module into the Python runtime"""
    module_name = path.stem
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise EmbedevalError(
                f"No loader for module {module_name} found"
            )  # pragma: no cover

        # NOTE(TF): add loaded module to ``sys.modules`` so that ``inspect`` will work correctly.
        sys.modules[module_name] = module

        spec.loader.exec_module(module)  # type: ignore
    except Exception as exc:
        raise EmbedevalError(
            "Unable to import module '{0}' from '{1}': {2}".format(
                module_name, path, exc
            )
        )


class TaskRegistry:
    """Registry for all available Tasks"""

    def __init__(self) -> None:
        self.tasks: Dict[str, Any] = {}

    def register(self, task_cls) -> None:
        """Register the given Task in the Registry"""
        try:
            task_name = task_cls.NAME
        except AttributeError:
            raise EmbedevalError(
                f"The Task class must have a `NAME` attribute "
                f"which is used for discovery"
            )

        if task_name in self.tasks:
            logger.debug(
                "Not registering Task %s, "
                "because one with the same name has already been registered",
                task_name,
            )
        else:
            self.tasks[task_name] = task_cls
            logger.debug("Registered Task %s", task_name)

    def create_task(self, name: str):
        """Create a new Task of the given type"""
        task_cls = self.get_task_cls(name)
        task = task_cls()
        logger.debug("Created Task %s", name)
        return task

    def get_task_cls(self, name: str):
        """Get a registered Task class for the given Task name"""
        try:
            return self.tasks[name]
        except KeyError:
            raise EmbedevalError(
                f"No Task with name '{name}' registered. "
                f"Choose one of: {', '.join(self.tasks.keys())}"
            )


#: Holds a global instance of the Task Registry.
#  It can be used to register or retrieve Tasks.
registry = TaskRegistry()
