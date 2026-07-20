# tests/unit/test_text_pipeline.py
# E-CIP v3.0 — Unit tests for models/sentiment/finetune.py:head_tail_tokenize
#
# Uses a minimal fake tokenizer (not a real HuggingFace download) — the
# function under test only needs a callable that returns {"input_ids": [...]}
# and cls/sep/pad token id attributes, so a real DistilBERT tokenizer would
# just add network I/O and a large download without testing anything the
# fake doesn't already exercise.

from __future__ import annotations

from typing import Any

from models.sentiment.finetune import HEAD_TOKENS, MAX_TOKENS, TAIL_TOKENS, head_tail_tokenize


class FakeTokenizer:
    """Whitespace-splits text into pseudo-token ids; one id per word."""

    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        truncation: bool = False,
        padding: str | None = None,
        return_tensors: Any = None,
    ) -> dict[str, Any]:
        words = text.split()
        input_ids = [hash(w) % 30000 + 1000 for w in words]  # avoid 0/101/102 collisions
        if truncation and max_length is not None:
            input_ids = input_ids[: max_length - 2]
        if add_special_tokens:
            input_ids = [self.cls_token_id, *input_ids, self.sep_token_id]
        attention_mask = [1] * len(input_ids)
        if padding == "max_length" and max_length is not None:
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestHeadTailTokenize:
    def test_short_text_no_truncation(self) -> None:
        tokenizer = FakeTokenizer()
        result = head_tail_tokenize("This product is great", tokenizer)
        assert result["truncation_applied"] is False
        assert len(result["input_ids"]) == MAX_TOKENS

    def test_long_text_triggers_head_tail_truncation(self) -> None:
        tokenizer = FakeTokenizer()
        # Force well past MAX_TOKENS - 2 so head+tail path is exercised.
        long_text = " ".join(f"word{i}" for i in range(MAX_TOKENS * 2))
        result = head_tail_tokenize(long_text, tokenizer)
        assert result["truncation_applied"] is True
        assert len(result["input_ids"]) == MAX_TOKENS

    def test_head_and_tail_both_preserved(self) -> None:
        """
        Fix #12: simple tail-truncation loses the review's conclusion —
        this verifies the FIRST and LAST distinctive words both survive
        truncation, not just one end.
        """
        tokenizer = FakeTokenizer()
        words = [f"uniqueword{i}" for i in range(MAX_TOKENS * 2)]
        long_text = " ".join(words)
        result = head_tail_tokenize(long_text, tokenizer)

        first_word_id = tokenizer(words[0], add_special_tokens=False)["input_ids"][0]
        last_word_id = tokenizer(words[-1], add_special_tokens=False)["input_ids"][0]

        input_ids = result["input_ids"]
        assert first_word_id in input_ids, "head tokens must survive truncation"
        assert last_word_id in input_ids, "tail tokens must survive truncation (Fix #12)"

    def test_truncated_length_respects_head_tail_budget(self) -> None:
        tokenizer = FakeTokenizer()
        long_text = " ".join(f"word{i}" for i in range(MAX_TOKENS * 2))
        result = head_tail_tokenize(long_text, tokenizer)
        # cls + head + tail + sep, then padded/truncated to MAX_TOKENS
        assert len(result["input_ids"]) == MAX_TOKENS
        assert HEAD_TOKENS + TAIL_TOKENS + 2 <= MAX_TOKENS + 2
