from typing import List

import pytest
from tiktoken.model import MODEL_TO_ENCODING

from tixent._core import _check_text_length, split, tiktoken_counter
from tixent.exceptions import TextTooLongError


class TestTiktokenCounter:
    valid_model_names = list(MODEL_TO_ENCODING.keys())
    valid_encoding_names = list(set(MODEL_TO_ENCODING.values()))

    def test_valid_model_name(self) -> None:
        name = self.valid_model_names[0]
        counter = tiktoken_counter(name)
        assert counter("hello") > 0

    def test_valid_encoding_name(self) -> None:
        name = self.valid_encoding_names[0]
        counter = tiktoken_counter(name)
        assert counter("hello") > 0

    def test_invalid_name(self) -> None:
        name = "foo"
        assert name not in self.valid_model_names
        assert name not in self.valid_encoding_names
        with pytest.raises(ValueError, match=name):
            tiktoken_counter(name)


class TestCheckTextLength:
    @staticmethod
    def concat_template(texts: List[str]) -> str:
        return "".join(texts)

    @staticmethod
    def length_counter(text: str) -> int:
        return len(text)

    def test_normal(self) -> None:
        texts = ["a" * 10]
        max_count = 11
        _check_text_length(texts, self.concat_template, self.length_counter, max_count)

    def test_too_long(self) -> None:
        texts = ["a" * 10]
        max_count = 9
        with pytest.raises(TextTooLongError):
            _check_text_length(
                texts, self.concat_template, self.length_counter, max_count
            )


class TestSplit:
    @staticmethod
    def summarization_template(texts: List[str]) -> str:
        text = " ".join(texts)
        t = "Summarize the following text.\n\n"
        t += f'Text: """{text}"""'
        return t

    def test_can_split_texts(self) -> None:
        texts = [
            "Lorem ipsum dolor sit amet",
            "consectetur adipiscing elit",
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
            "Ut enim ad minim veniam",
            "quis nostrud exercitation ullamco laboris nisi",
            "ut aliquip ex ea commodo consequat.",
            "Duis aute irure dolor in reprehenderit in voluptate velit",
            "esse cillum dolore eu fugiat nulla pariatur.",
            "Excepteur sint occaecat cupidatat non proident",
            "sunt in culpa qui officia deserunt mollit anim id est laborum.",
        ]
        counter = tiktoken_counter("p50k_base")
        max_count = 60

        split_texts = split(texts, self.summarization_template, counter, max_count)
        lower_limit_of_count = counter(self.summarization_template([]))
        for text in split_texts:
            assert lower_limit_of_count < counter(text) <= max_count
            assert text.startswith("Summarize the following text.")

        assert texts[0] in split_texts[0]
        assert texts[-1] in split_texts[-1]

    def test_empty_texts(self) -> None:
        counter = tiktoken_counter("p50k_base")
        assert split([], self.summarization_template, counter, 10) == []
