"""Tokenizer/model preparation for decisecond target tokens."""

from __future__ import annotations

from sceneledger.models.target_formatter import T_TOKEN_COUNT, time_to_token


def timestamp_tokens() -> list[str]:
    return [time_to_token(index * 0.1) for index in range(T_TOKEN_COUNT)]


def compute_timestamp_token_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for token in timestamp_tokens():
        sub_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(sub_ids) != 1:
            raise ValueError(
                f"timestamp token {token!r} is not atomic (encoded as {sub_ids}); "
                "register special tokens and resize model embeddings first"
            )
        ids.add(int(sub_ids[0]))
    if len(ids) != T_TOKEN_COUNT:
        raise ValueError(
            f"timestamp vocabulary collision: expected {T_TOKEN_COUNT} unique IDs, got {len(ids)}"
        )
    return ids


def ensure_atomic_timestamp_tokens(
    tokenizer,
    model=None,
    *,
    register_missing: bool = False,
) -> tuple[set[int], int]:
    """Validate or register the 301 timestamp tokens.

    Returns ``(token_ids, added_count)``.  Registration is opt-in because
    resizing a pretrained LM without saving the embedding modules would make
    LoRA checkpoints unusable.
    """
    try:
        return compute_timestamp_token_ids(tokenizer), 0
    except ValueError:
        if not register_missing:
            raise
    added = int(
        tokenizer.add_special_tokens(
            {"additional_special_tokens": timestamp_tokens()},
            replace_additional_special_tokens=False,
        )
    )
    if model is None:
        raise ValueError("a model is required when registering timestamp tokens")
    if not hasattr(model, "resize_token_embeddings"):
        raise TypeError("model does not implement resize_token_embeddings")
    model.resize_token_embeddings(len(tokenizer))
    return compute_timestamp_token_ids(tokenizer), added


def input_embedding_module_name(model) -> str:
    """Return the PEFT module suffix for selective new-token training."""
    getter = getattr(model, "get_input_embeddings", None)
    if getter is None or getter() is None:
        raise RuntimeError("model does not expose get_input_embeddings()")
    target = getter()
    for name, module in model.named_modules():
        if module is target:
            suffix = name.rsplit(".", 1)[-1]
            if suffix:
                return suffix
    raise RuntimeError("could not locate input embedding module for PEFT token training")


__all__ = [
    "compute_timestamp_token_ids",
    "input_embedding_module_name",
    "ensure_atomic_timestamp_tokens",
    "timestamp_tokens",
]
