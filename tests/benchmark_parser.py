"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

from pathlib import Path

import pytest

from embedeval.parsers.word2vec_gensim import load_embedding

DOWNLOADED_TEST_DATA = Path(__file__).parent / "data" / "downloads"


@pytest.mark.parametrize(
    "word2vec_path",
    [
        pytest.param(
            DOWNLOADED_TEST_DATA / "cc.de.300.vec", id="cc.de.300.vec (2M / 300)"
        )
    ],
)
def test_parser_word2vec_text_benchmark(word2vec_path, benchmark):
    """Test benchmarks for loading word2vec text Embeddings"""
    benchmark.pedantic(load_embedding, args=(word2vec_path,), iterations=1, rounds=1)


@pytest.mark.parametrize(
    "word2vec_path",
    [
        pytest.param(
            DOWNLOADED_TEST_DATA / "cc.de.300.bin", id="cc.de.300.bin (2M / 300)"
        )
    ],
)
def test_parser_word2vec_bin_benchmark(word2vec_path, benchmark):
    """Test benchmarks for loading word2vec text Embeddings"""
    benchmark.pedantic(
        load_embedding,
        args=(word2vec_path,),
        kwargs={"binary": True},
        iterations=1,
        rounds=1,
    )
