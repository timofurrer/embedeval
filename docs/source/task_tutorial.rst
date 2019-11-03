Task Implementation Tutorial
============================

Tasks are the heart pieces of embedeval.
They are used to evaluate Word Embeddings on their quality
with respect to certain measures given certain baselines.

The Task API can be found in detail in the :ref:`Task API <task_api>`
section of this documentation.

The following sections will guide through the steps which need
to be done to implement a Task and how to do an evaluation with them.

Step 1: Task Anatomy
--------------------

First we'll see how a Task is represented in code.

A Task is a subclass of the following Python Interface:

.. literalinclude:: ../../src/embedeval/task.py
   :pyobject: Task

Don't mind the ``__init__subclass__`` magic method as it's only used internally
to register the Tasks in the Task Registry.

Important is ``evaluate()`` method. that's the method used to implement
the evaluation algorithm of a Task.

Step 2: Define a Task
---------------------

The first thing to think about is the name of the Task.
The name is used to reference the Task from the command line
to use it for the evaluation.

This name can be used as the class name suffixed with ``Task``
and must be used as the value for the ``NAME`` class attribute:

.. testcode::

   from embedeval.task import Task


   class WordSimilarityTask(Task):
       """Compare two words in regards of how similar they are"""

       NAME = "word-similarity"

       def evaluate(self, embedding):
           ...

This Task can be implemented in an appropriatly named Python module
within the embedeval source code at: ``src/embedeval/tasks/``.
A good name module for this Task would be ``src/embedeval/tasks/word_similarity.py``.


Step 3: Implement Task Algorithm
--------------------------------

The most important step is to implement the evaluation algorithm
for the new Task.

The ``evaluate()`` method is the place for that.
It is given a ``WordEmbedding`` instance and should return an
evauation report as a human readable string:


.. testcode::

   import colorful as cf

   from embedeval.task import Task


   class WordSimilarityTask(Task):
       """Compare two words in regards of how similar they are"""

       NAME = "word-similarity"

       def evaluate(self, embedding):
           woman_vector = embedding.get_word_vector("woman")
           man_vector = embedding.get_word_vector("man")

           # TODO: ``cosine_similarity`` must be implemented
           similarity = cosine_similarity(woman_vector, man_vector)

           return f"""
               {cf.bold}The Task{cf.reset}:
                   How similar is the word woman to the word man?
               was {cf.underlined}successful{cf.reset}.
               The similarity was measured to be
                   {cf.bold}{similarity:.2}{cf.reset}
           """


Step 4: Run the Task
--------------------

Now that the evaluation algorithm is implemented we can run the Task
on a Word Embedding from the command line using ``embedeval``:

.. code-block:: bash

   embedeval embedding.vec -t word-similarity
