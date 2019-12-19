# NLP Embeddings Evaluation Tool

[![PyPI License](https://img.shields.io/pypi/l/embedeval.svg)](https://github.com/timofurrer/embedeval/blob/master/LICENSE)
<br>
[![Actions Status](https://github.com/timofurrer/embedeval/workflows/CI/badge.svg)](https://github.com/timofurrer/embedeval/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
<br>
[![PyPI version](https://badge.fury.io/py/embedeval.svg)](https://badge.fury.io/py/embedeval)
[![PyPI](https://img.shields.io/pypi/pyversions/embedeval.svg)](https://pypi.python.org/pypi/embedeval)
[![PyPI](https://img.shields.io/pypi/wheel/embedeval.svg)](https://pypi.python.org/pypi/embedeval)

***

The *NLP Embeddings Evaluation Tool* is a command line tool to evaluate Natural Language Processing Embeddings
using custom intrinsic and extrinsic tasks.

# Installation

embedeval is available as `pip` package:

```bash
python -m pip install embedeval
```

NOTE: it might not be installable as of today using pip with PyPI.
However, installing from source will work. Use `.` instead of `embedeval` in the pip command.

# Getting started

Run the `word-analogy` Task on your Word Embedding:

```bash
embedeval embedding.vec -t word-analogy
```

Run the `word-analogy` and `word-similarity` Tasks on your Word Embedding:

```bash
embedeval embedding.vec -t word-analogy -t word-similarity
```

# Documentation

The whole documentation of embedeval is available on [Read The Docs](http://embedeval.readthedocs.org).

# Supported platforms

embedeval is supported on Windows, Mac and Linux

# Contribution

Yes, we are looking for some contributors and people who spread out a word about embedeval. Help us to improve these piece of software. You don't know what to do?
Just have a look at the Issues or create a new one.
Please have a look at the [Contributing Guidelines](https://github.com/timofurrer/embedeval/blob/master/.github/CONTRIBUTING.md), too.

# Project Information

embedeval is released under the MIT license, its documentation lives at [Read The Docs](http://embedeval.readthedocs.org),
the code on [GitHub](https://github.com/timofurrer/embedeval),
and the latest release on [PyPI](https://pypi.org/project/embedeval).
It’s rigorously tested on Python 3.5+.

If you'd like to contribute to embedeval you're most welcome and we've written a [little guide](https://github.com/timofurrer/embedeval/blob/master/.github/CONTRIBUTING.md) to get you started!

***

*<p align="center">This project is published under [MIT](LICENSE).<br>A [Timo Furrer](https://tuxtimo.me) project.<br>- :tada: -</p>*
