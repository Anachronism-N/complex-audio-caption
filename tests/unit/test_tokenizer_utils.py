"""Dependency-free tests for timestamp vocabulary registration."""

from __future__ import annotations

import pytest

from sceneledger.models.tokenizer_utils import (
    compute_timestamp_token_ids,
    ensure_atomic_timestamp_tokens,
    input_embedding_module_name,
)


class FakeTokenizer:
    def __init__(self):
        self.special: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.special:
            return [self.special[text]]
        return [ord(char) for char in text]

    def add_special_tokens(
        self, payload: dict, *, replace_additional_special_tokens: bool = True
    ) -> int:
        assert replace_additional_special_tokens is False
        before = len(self.special)
        for token in payload["additional_special_tokens"]:
            self.special.setdefault(token, 1000 + len(self.special))
        return len(self.special) - before

    def __len__(self) -> int:
        return 1000 + len(self.special)


class FakeModel:
    resized_to: int | None = None

    def resize_token_embeddings(self, size: int) -> None:
        self.resized_to = size


class FakeEmbeddingModel(FakeModel):
    def __init__(self):
        self.embedding = object()

    def get_input_embeddings(self):
        return self.embedding

    def named_modules(self):
        return [("", self), ("language_model.embed_tokens", self.embedding)]


def test_timestamp_registration_is_fail_closed_then_atomic():
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="not atomic"):
        compute_timestamp_token_ids(tokenizer)
    model = FakeModel()
    ids, added = ensure_atomic_timestamp_tokens(
        tokenizer, model, register_missing=True
    )
    assert added == 301
    assert len(ids) == 301
    assert model.resized_to == len(tokenizer)


def test_input_embedding_module_suffix_is_discovered():
    assert input_embedding_module_name(FakeEmbeddingModel()) == "embed_tokens"
