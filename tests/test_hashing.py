import datetime as dt
import hashlib
import json
import unicodedata

import pytest

from review_radar.hashing import cache_key, content_hash, normalize_text, text_hash


class TestNormalize:
    def test_collapses_whitespace_and_casefolds(self) -> None:
        assert normalize_text("  Hello \n\t World !  ") == "hello world !"

    def test_nfc(self) -> None:
        nfd = unicodedata.normalize("NFD", "café")
        assert normalize_text(nfd) == "café"

    def test_casefold_beyond_lower(self) -> None:
        assert normalize_text("Straße") == "strasse"

    def test_empty(self) -> None:
        assert normalize_text("  \n ") == ""


class TestTextHash:
    def test_known_value(self) -> None:
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert text_hash("  Hello\n\tWORLD ") == expected

    def test_stable_across_variants(self) -> None:
        base = text_hash("café au lait")
        assert text_hash(unicodedata.normalize("NFD", "CAFÉ  au\nlait")) == base

    def test_different_text_differs(self) -> None:
        assert text_hash("a") != text_hash("b")

    def test_non_ascii(self) -> None:
        h = text_hash("Ужасная доставка 😡")
        assert len(h) == 64
        assert h == hashlib.sha256("ужасная доставка 😡".encode()).hexdigest()


GOLDEN_TEXT_HASH = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
GOLDEN_CACHE_DIGEST = "a937b8a6cd8cd4ea92b698ccd820be517912bb33306043dc9d3dab1ad999d680"


def key(text: str = "hi", provider: str = "p", model: str = "m", version: str = "v1") -> str:
    return cache_key(text, provider=provider, model=model, prompt_version=version)


class TestGolden:
    def test_text_hash_literal(self) -> None:
        assert text_hash("Hello  World") == GOLDEN_TEXT_HASH

    def test_cache_key_literal(self) -> None:
        got = cache_key("Hello  World", provider="anthropic", model="m1", prompt_version="v1")
        assert got == GOLDEN_CACHE_DIGEST


class TestCacheKey:
    def test_known_value(self) -> None:
        payload = json.dumps([text_hash("hi"), "p", "m", "v1"])
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        assert key() == expected

    def test_deterministic_and_normalized(self) -> None:
        assert key(" HI ") == key("hi")

    def test_provider_change(self) -> None:
        assert key(provider="p1") != key(provider="p2")

    def test_model_change(self) -> None:
        assert key(model="m1") != key(model="m2")

    def test_prompt_version_change(self) -> None:
        assert key(version="v1") != key(version="v2")

    def test_text_change(self) -> None:
        assert key("hi") != key("ho")

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            cache_key("hi", "p", "m", "v1")  # type: ignore[misc]

    def test_separator_ambiguity(self) -> None:
        assert key(provider="a", model="bc") != key(provider="ab", model="c")
        assert key(model="a", version="bc") != key(model="ab", version="c")
        assert key(model="a|b", version="c") != key(model="a", version="b|c")
        assert key(model="a\x00b", version="c") != key(model="a", version="b\x00c")

    @pytest.mark.parametrize(
        ("field", "name"),
        [("provider", "provider"), ("model", "model"), ("version", "prompt_version")],
    )
    @pytest.mark.parametrize("bad", ["", " ", "\t\n  "])
    def test_blank_rejected(self, field: str, name: str, bad: str) -> None:
        with pytest.raises(ValueError, match=f"{name} must not be empty"):
            key(**{field: bad})


class TestContentHash:
    DAY = dt.date(2026, 11, 10)

    def test_matches_documented_encoding(self) -> None:
        expected = hashlib.sha256(
            json.dumps(["2026-11-10", text_hash("Hello"), "web"]).encode("utf-8")
        ).hexdigest()

        assert content_hash(self.DAY, "Hello", "web") == expected

    def test_text_is_normalized(self) -> None:
        assert content_hash(self.DAY, " HELLO  world ", None) == content_hash(
            self.DAY, "hello world", None
        )

    def test_none_source_is_distinct_from_string(self) -> None:
        assert content_hash(self.DAY, "a", None) != content_hash(self.DAY, "a", "None")

    def test_parts_cannot_collide(self) -> None:
        assert content_hash(self.DAY, "a|b", "c") != content_hash(self.DAY, "a", "b|c")
