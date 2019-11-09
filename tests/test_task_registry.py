"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import pytest

from embedeval.errors import EmbedevalError
from embedeval.taskregistry import TaskRegistry, load_module, load_tasks


def test_should_fail_to_register_task_without_name():
    # GIVEN
    registry = TaskRegistry()

    class FaultyTask:
        ...

    # THEN
    with pytest.raises(EmbedevalError, match="Task class must have a `NAME` attribute"):
        # WHEN
        registry.register(FaultyTask)


def test_should_successfully_register_task_with_a_name():
    # GIVEN
    registry = TaskRegistry()

    class Task:
        NAME = "some-task"

    # WHEN
    registry.register(Task)

    # THEN
    assert registry.get_task_cls("some-task") is Task


def test_should_not_register_task_with_same_name_twice():
    # GIVEN
    registry = TaskRegistry()

    class Task:
        NAME = "some-task"

    class LateTask:
        NAME = "some-task"

    registry.register(Task)

    # WHEN
    registry.register(LateTask)

    # THEN
    assert registry.get_task_cls("some-task") is Task


def test_should_fail_to_get_not_existent_task():
    # GIVEN
    registry = TaskRegistry()

    class Task:
        NAME = "some-task"

    registry.register(Task)

    # THEN
    with pytest.raises(
        EmbedevalError, match="No Task with name 'other-task' registered"
    ):
        # WHEN
        registry.get_task_cls("other-task")


def test_should_successfully_create_registered_task():
    # GIVEN
    registry = TaskRegistry()

    class Task:
        NAME = "some-task"

    registry.register(Task)

    # WHEN
    task = registry.create_task("some-task")

    # THEN
    assert isinstance(task, Task)


def test_should_fail_to_load_task_for_not_existent_location(mocker):
    # GIVEN
    non_existant_location = mocker.MagicMock()
    non_existant_location.exists.return_value = False
    locations = [mocker.MagicMock(), non_existant_location]

    # THEN
    with pytest.raises(EmbedevalError):
        # WHEN
        load_tasks(locations)


def test_should_load_python_module_in_location(mocker):
    # GIVEN
    load_module_mock = mocker.patch("embedeval.taskregistry.load_module")
    module_mock = mocker.MagicMock(name="Python Module")
    location = mocker.MagicMock()
    location.glob.return_value = [module_mock]

    # WHEN
    loaded_locations = load_tasks([location])

    # THEN
    load_module_mock.assert_called_once_with(module_mock)
    assert loaded_locations == [module_mock]


def test_should_fail_task_loading_when_import_fails(mocker):
    # GIVEN
    spec_mock = mocker.patch("importlib.util.spec_from_file_location")
    spec_mock.return_value.loader.exec_module.side_effect = Exception("buuhu!")
    mocker.patch("importlib.util.module_from_spec")
    location = mocker.MagicMock("Python Module Path")
    location.stem = "foo"

    # THEN
    with pytest.raises(
        EmbedevalError, match="Unable to import module 'foo' .*: buuhu!"
    ):
        # WHEN
        load_module(location)
