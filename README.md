# Tixent: A Text Splitting Tool

**PyPI**
[![PyPI - Version](https://img.shields.io/pypi/v/tixent.svg)](https://pypi.org/project/tixent)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tixent.svg)](https://pypi.org/project/tixent)
[![License](https://img.shields.io/pypi/l/tixent.svg)](https://github.com/sincekmori/tixent/blob/main/LICENSE)

**CI/CD**
[![test](https://github.com/sincekmori/tixent/actions/workflows/test.yml/badge.svg)](https://github.com/sincekmori/tixent/actions/workflows/test.yml)
[![lint](https://github.com/sincekmori/tixent/actions/workflows/lint.yml/badge.svg)](https://github.com/sincekmori/tixent/actions/workflows/lint.yml)

**Build System**
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

**Code**
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

**Docstrings**
[![docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![numpy](https://img.shields.io/badge/%20style-numpy-459db9.svg)](https://numpydoc.readthedocs.io/en/latest/format.html)

---

## Installation

```console
pip install tixent
```

## Example

Suppose we have a function template that generates a string from a list of texts.
Additionally, suppose we have a large list of texts.
When you apply that list of texts to the function, it generates a long string.

Tixent can split the string generated by the template function so that the return value of _counter_ for each element is less than a certain number.

Here, _counter_ is a function that maps a string to an integer.
Examples of such functions are `len`, which measures the length of a string, or `tiktoken_counter("text-davinci-003")`, which measures the number of tokens in a string

```python
from typing import List

from tixent import split, tiktoken_counter


def summarization_template(texts: List[str]) -> str:
    text = " ".join(texts)
    t = "Summarize the following text.\n"
    t += f'Text: """{text}"""'
    return t


texts = [
    "Lorem ipsum dolor sit amet",
    "consectetur adipiscing elit",
    "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
    "Ut enim ad minim veniam",
    "quis nostrud exercitation ullamco laboris nisi",
    "ut aliquip ex ea commodo consequat",
    "Duis aute irure dolor in reprehenderit in voluptate velit",
    "esse cillum dolore eu fugiat nulla pariatur",
    "Excepteur sint occaecat cupidatat non proident",
    "sunt in culpa qui officia deserunt mollit anim id est laborum",
]
counter = tiktoken_counter("text-davinci-003")
max_count = 60

split_texts = split(texts, summarization_template, counter, max_count)
for text in split_texts:
    count = counter(text)
    assert count <= max_count
    print(f"count: {count}")
    print(text)
    print()
```

```console
count: 60
Summarize the following text.
Text: """Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam"""

count: 58
Summarize the following text.
Text: """quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat Duis aute irure dolor in reprehenderit in voluptate velit"""

count: 43
Summarize the following text.
Text: """esse cillum dolore eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident"""

count: 31
Summarize the following text.
Text: """sunt in culpa qui officia deserunt mollit anim id est laborum"""
```
