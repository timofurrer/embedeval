Usage
=====

**embedeval** is deployed as Command Line Interface Tool for Linux, Mac OS X and Windows.
It can be downloaded from PyPI using pip:

.. code:: bash

    pip install embedeval


Task Management
---------------

An important part of *embedeval* is the management of Tasks.
The CLI supports two commands to either list the available Tasks
or create new Tasks.

.. _list_tasks:

List Tasks
~~~~~~~~~~

To list all available built-in Tasks use the following command:

.. code:: bash

    embedeval tasks


It'll list the names of all available Tasks. If there is
an additional directory of Tasks where *embedeval* should
look at use the ``-p`` or ``--tasks-path`` option:

.. code:: bash

    # also look in the relative tasks/ directory for Tasks
    embedeval tasks -p tasks/

    # also look in the absolute /etc/tasks directory for Tasks
    embedeval tasks --tasks-path /etc/tasks

Create Tasks
~~~~~~~~~~~~

The second CLI command can be used to create new Tasks.

This command is described in detail in the :ref:`A new Task from the CLI <new_task_cli>` section.


Task Evaluation
---------------

The most important part of **embedeval**, however, is the evaluation of
Word Embeddings using Tasks.

The following few section go into detail on how to execute an evaluation
and how to interpret its reports.

To execute an evaluation the ``embedeval`` default command can be used
or it can be specified explicitly using the ``embedeval eval`` command:

.. code:: bash

    # run an evaluation on embeddding.vec
    # with built-in task word-analogy
    embedeval embedding.vec -t word-analogy

    # same as above, but use the ``eval`` command explicitly
    embedeval eval embedding.vec -t word-analogy

The ``-t`` or ``--task`` CLI option can be used one or multiple times
to specify which Tasks should be run. The available Tasks
can be listed with the :ref:`tasks command <list_tasks>`.

In case the Tasks to run are not built-in Tasks the
``-p`` or ``--tasks-path`` CLI option can be used to
specify the directory where the Tasks are located:

.. code:: bash

    # run an evaluation on embeddding.vec
    # with task word-analogy from the tasks/ directory
    embedeval embedding.vec -t word-analogy -p tasks/
