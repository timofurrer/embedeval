"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by Timo Furrer <tuxtimo@gmail.com>
:license: MIT, see LICENSE for more details.
"""

import functools
import re
from pathlib import Path

from setuptools import find_packages, setup

#: Holds a list of packages to install with the binary distribution
PACKAGES = find_packages(where="src")
META_FILE = Path("src").absolute() / "embedeval" / "__init__.py"
KEYWORDS = [
    "machine-learning",
    "natural-language-processing",
    "nlp",
    "embeddings",
    "evaluation",
    "report",
]
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Other Audience",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Education :: Testing",
]

#: Holds the runtime requirements for the end user
INSTALL_REQUIRES = [
    "click>=7",
    "click-default-group",
    "colorful",
    "numpy",
    "pandas",
    "gensim",
    "keras",
    "tensorflow",
    "nltk",
    # somehow botocore which is a transitive dependency has some requirements not correctly pinned.
    "python-dateutil<2.8.1"
]
#: Holds runtime requirements and development requirements
EXTRAS_REQUIRES = {
    # extras for contributors
    "docs": ["sphinx"],
    "tests": ["coverage", "pytest", "pytest-mock", "pytest-benchmark"],
    "notebooks": ["jupyter", "matplotlib", "seaborn"],
}
EXTRAS_REQUIRES["dev"] = (
    EXTRAS_REQUIRES["tests"]
    + EXTRAS_REQUIRES["docs"]
    + EXTRAS_REQUIRES["notebooks"]
    + ["pre-commit"]
)

#: Holds the contents of the README file
with open("README.md", encoding="utf-8") as readme:
    __README_CONTENTS__ = readme.read()


@functools.lru_cache()
def read(metafile):
    """
    Return the contents of the given meta data file assuming UTF-8 encoding.
    """
    with open(str(metafile), encoding="utf-8") as f:
        return f.read()


def get_meta(meta, metafile):
    """
    Extract __*meta*__ from the given metafile.
    """
    contents = read(metafile)
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), contents, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


setup(
    name="embedeval",
    version=get_meta("version", META_FILE),
    license=get_meta("license", META_FILE),
    description="NLP Embeddings Evaluation Tool",
    long_description=__README_CONTENTS__,
    long_description_content_type="text/markdown",
    author="Timo Furrer, David Staub",
    author_email="tuxtimo@gmail.com, david_staub@hotmail.com",
    maintainer="Timo Furrer, David Staub",
    maintainer_email="tuxtimo@gmail.com, david_staub@hotmail.com",
    platforms=["Linux", "Windows", "MAC OS X"],
    url="https://github.com/timofurrer/hslu-wipro",
    download_url="https://github.com/timofurrer/hslu-wipro",
    bugtrack_url="https://github.com/timofurrer/hslu-wipro/issues",
    packages=PACKAGES,
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    entry_points={"console_scripts": ["embedeval = embedeval.cli:cli"]},
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
)
