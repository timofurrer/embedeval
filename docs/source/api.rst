.. _api:

API
===

.. module:: embedeval

This part of the documentation lists the full API reference of all public
classes and functions.

Word Embedding Representation
-----------------------------

Embedeval uses a generic internal representation for Word Embeddings.
This representation *is not* dependent on the source format of the Embedding itself,
to give a :class:`Task` the chance to be *Embedding format independent*.
However, a Task may reference and therefore depend on the source format of the
Embedding.

.. autoclass:: embedeval.embedding.WordEmbedding
   :members:

.. _task_api:

Task API
--------

The Tasks are the heart piece of Embedeval.
The :class:`Task` must be subclassed by a concrete implementation of
an evaluation Task to implement the evaluation algorithm.

.. autoclass:: embedeval.task.Task
   :members:


Implemented Word Embedding Parsers
----------------------------------

Embedeval implements parsers for some well-known Word Embedding formats.

.. autoclass:: embedeval.parsers.word2vec_gensim.KeyedVectorsWordEmbedding
   :members:

.. autoclass:: embedeval.parsers.word2vec_simple.SimpleWordEmbedding
   :members:


Top-Level Package Exports
-------------------------

The ``embedeval`` Python package exports the Task API as top-level names:

.. testcode::

    from embedeval import (
        Task,
        TaskReport,
        EmbedevalError
    )
