"""
test_preprocessor.py — Unit tests for the TextPreprocessor.

Run:
    pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from ml_pipeline.data.preprocessor import TextPreprocessor


class TestTextPreprocessor:

    def setup_method(self):
        self.prep = TextPreprocessor()

    def test_lowercase(self):
        assert self.prep.clean("HELLO WORLD") == self.prep.clean("hello world")

    def test_removes_html(self):
        result = self.prep.clean("<p>Breaking news</p>")
        assert "<" not in result and ">" not in result

    def test_removes_urls(self):
        result = self.prep.clean("Read more at https://example.com/story")
        assert "http" not in result

    def test_normalises_money(self):
        result = self.prep.clean("Company earns $5.4B")
        assert "MONEY" in result

    def test_normalises_percent(self):
        result = self.prep.clean("Prices rose 7%")
        assert "PERCENT" in result

    def test_empty_string(self):
        assert self.prep.clean("") == ""

    def test_none_like_non_string(self):
        assert self.prep.clean(None) == ""  # type: ignore

    def test_removes_short_tokens(self):
        result = self.prep.clean("a b c the quick fox")
        for token in result.split():
            assert len(token) > 1

    def test_punctuation_removed(self):
        result = self.prep.clean("Hello! World... Yes?")
        assert "!" not in result and "?" not in result

    def test_batch_preserves_order(self):
        texts = ["first article", "second article", "third article"]
        results = self.prep.clean_batch(texts)
        assert len(results) == 3
        assert "first" in results[0]
        assert "second" in results[1]
        assert "third" in results[2]

    def test_real_headline(self):
        headline = "Apple Inc. reports record $90B in Q2 2024 earnings — stock rises 4%"
        result = self.prep.clean(headline)
        assert isinstance(result, str)
        assert len(result) > 0
        # Stock-related words should survive
        assert any(w in result for w in ["appl", "stock", "record", "earn", "rise"])
