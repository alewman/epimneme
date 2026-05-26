"""Tests for engram.rerank — keyword reranking module."""

from epimneme.rerank import keyword_rerank, _tokenize, _extract_phrases, RerankResult


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Hello world, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Stop words removed
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_single_word(self):
        assert _tokenize("python") == ["python"]

    def test_only_stop_words(self):
        assert _tokenize("the is a an") == []


class TestExtractPhrases:
    def test_bigrams(self):
        tokens = _tokenize("memory management system")
        phrases = _extract_phrases(tokens)
        assert "memory management" in phrases
        assert "management system" in phrases

    def test_short_input(self):
        phrases = _extract_phrases(["hello"])
        assert phrases == []

    def test_empty(self):
        assert _extract_phrases([]) == []


class TestKeywordRerank:
    def test_basic_reranking(self):
        results = [
            ("id1", 0.8, "Python memory management techniques"),
            ("id2", 0.85, "JavaScript event loop overview"),
            ("id3", 0.7, "Python garbage collection and memory pools"),
        ]
        reranked = keyword_rerank("python memory", results)

        assert len(reranked) == 3
        assert all(isinstance(r, RerankResult) for r in reranked)
        # Python memory items should get boosted
        scores_by_id = {r.memory_id: r.final_score for r in reranked}
        assert scores_by_id["id1"] > scores_by_id["id2"]

    def test_empty_results(self):
        reranked = keyword_rerank("test query", [])
        assert reranked == []

    def test_preserves_all_items(self):
        results = [
            ("id1", 0.5, "content one"),
            ("id2", 0.6, "content two"),
            ("id3", 0.7, "content three"),
        ]
        reranked = keyword_rerank("unrelated query xyz", results)
        assert len(reranked) == 3
        ids = {r.memory_id for r in reranked}
        assert ids == {"id1", "id2", "id3"}

    def test_phrase_matching_boost(self):
        results = [
            ("id1", 0.5, "memory management is important"),
            ("id2", 0.5, "management of various memory types"),
        ]
        reranked = keyword_rerank("memory management", results)
        # id1 has exact phrase "memory management", should score higher
        assert reranked[0].memory_id == "id1"

    def test_custom_weights(self):
        results = [
            ("id1", 0.5, "python programming"),
            ("id2", 0.5, "java programming"),
        ]
        reranked = keyword_rerank("python", results, term_weight=0.5)
        assert reranked[0].memory_id == "id1"
        assert reranked[0].final_score > reranked[1].final_score
