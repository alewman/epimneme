"""Tests for engram.fusion — RRF fusion, proper noun extraction, preference terms, adaptive weights."""

import pytest
from epimneme.core.models import Memory, MemoryKind, MemoryResult
from epimneme.fusion import (
    rrf_fuse, extract_proper_nouns, extract_preference_terms,
    adaptive_keyword_weight,
)


def _mr(mid: str, score: float = 0.0, content: str = "") -> MemoryResult:
    """Helper to build a MemoryResult with a given id and score."""
    return MemoryResult(
        memory=Memory(id=mid, kind=MemoryKind.FACT, content=content),
        score=score,
    )


class TestRRFFuse:
    def test_single_list(self):
        results = [_mr("a", 0.9), _mr("b", 0.8), _mr("c", 0.7)]
        fused = rrf_fuse(results)
        assert len(fused) == 3
        # First item has highest RRF score
        assert fused["a"].score > fused["b"].score > fused["c"].score

    def test_two_lists_overlap(self):
        vec = [_mr("a", 0.9), _mr("b", 0.8), _mr("c", 0.7)]
        kw = [_mr("b", 0.3), _mr("d", 0.2), _mr("a", 0.1)]
        fused = rrf_fuse(vec, kw)
        assert len(fused) == 4  # a, b, c, d
        # b appears in both as #2 and #1 — should outscore a (#1 + #3)
        # b: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        # a: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
        assert fused["b"].score > fused["a"].score

    def test_disjoint_lists(self):
        vec = [_mr("a"), _mr("b")]
        kw = [_mr("c"), _mr("d")]
        fused = rrf_fuse(vec, kw)
        assert len(fused) == 4
        # a and c tied at rank 1 in their respective lists
        assert abs(fused["a"].score - fused["c"].score) < 1e-9

    def test_empty_lists(self):
        fused = rrf_fuse([], [])
        assert fused == {}

    def test_one_empty_one_populated(self):
        vec = [_mr("a"), _mr("b")]
        fused = rrf_fuse(vec, [])
        assert len(fused) == 2

    def test_item_in_both_gets_higher_score(self):
        """Memory appearing in both lists beats one appearing in only one."""
        vec = [_mr("a", content="shared"), _mr("only_vec")]
        kw = [_mr("a", content="shared"), _mr("only_kw")]
        fused = rrf_fuse(vec, kw)
        # 'a' appears in both at rank 1 → double RRF contribution
        assert fused["a"].score > fused["only_vec"].score
        assert fused["a"].score > fused["only_kw"].score

    def test_rrf_scores_are_rank_based(self):
        """Raw input scores should be ignored; only rank matters."""
        # Huge score difference but same rank → same RRF score
        list1 = [_mr("a", 0.99)]
        list2 = [_mr("b", 0.01)]
        fused = rrf_fuse(list1, list2)
        assert abs(fused["a"].score - fused["b"].score) < 1e-9

    def test_weighted_rrf(self):
        """Higher weight gives a list more influence."""
        vec = [_mr("a"), _mr("b")]
        kw = [_mr("c"), _mr("d")]
        fused = rrf_fuse(vec, kw, weights=[1.0, 0.5])
        # 'a' from vec (weight 1.0) should outscore 'c' from kw (weight 0.5)
        assert fused["a"].score > fused["c"].score
        # Ratio should be 2:1
        assert abs(fused["a"].score / fused["c"].score - 2.0) < 1e-9


class TestExtractProperNouns:
    def test_basic_names(self):
        nouns = extract_proper_nouns("Did John talk about Python with Rachel?")
        assert "John" in nouns
        assert "Python" in nouns
        assert "Rachel" in nouns

    def test_filters_common_words(self):
        nouns = extract_proper_nouns("What did The man say?")
        assert "What" not in nouns
        assert "The" not in nouns

    def test_empty_string(self):
        assert extract_proper_nouns("") == []

    def test_no_proper_nouns(self):
        assert extract_proper_nouns("all lowercase words here") == []

    def test_deduplication(self):
        nouns = extract_proper_nouns("John met John at John's house")
        assert nouns.count("John") == 1

    def test_preserves_order(self):
        nouns = extract_proper_nouns("Alice met Bob then Carol")
        assert nouns == ["Alice", "Bob", "Carol"]


class TestExtractPreferenceTerms:
    def test_prefer_statement(self):
        terms = extract_preference_terms("I prefer the Sony A7R IV for landscape photography.")
        term_set = {t.lower() for t in terms}
        assert "sony" in term_set
        assert "a7r" in term_set
        assert "landscape" in term_set
        assert "photography" in term_set

    def test_like_statement(self):
        terms = extract_preference_terms("I really like turbinado sugar in my cookies.")
        term_set = {t.lower() for t in terms}
        assert "turbinado" in term_set
        assert "sugar" in term_set
        assert "cookies" in term_set

    def test_no_preference_language(self):
        terms = extract_preference_terms("The weather today is sunny and warm.")
        assert terms == []

    def test_bought_purchased(self):
        terms = extract_preference_terms("I bought a portable power bank last week.")
        term_set = {t.lower() for t in terms}
        assert "portable" in term_set
        assert "power" in term_set
        assert "bank" in term_set

    def test_multiple_sentences(self):
        text = "The cat sat on the mat. I enjoy hiking in the mountains. Rain is wet."
        terms = extract_preference_terms(text)
        term_set = {t.lower() for t in terms}
        assert "hiking" in term_set
        assert "mountains" in term_set
        # Non-preference sentences shouldn't contribute
        assert "cat" not in term_set
        assert "rain" not in term_set

    def test_deduplication(self):
        terms = extract_preference_terms("I like Sony. I prefer Sony cameras.")
        lower_terms = [t.lower() for t in terms]
        assert lower_terms.count("sony") == 1

    def test_upgrade_looking(self):
        terms = extract_preference_terms(
            "I'm looking to upgrade my camera flash. Can you recommend a Godox V1?"
        )
        term_set = {t.lower() for t in terms}
        assert "camera" in term_set
        assert "flash" in term_set


class TestAdaptiveKeywordWeight:
    BASE = 0.75

    def test_vague_any_tips_no_nouns(self):
        w = adaptive_keyword_weight("Any tips?", self.BASE)
        assert w == pytest.approx(self.BASE * 0.3)

    def test_vague_any_tips_with_nouns_keeps_weight(self):
        # "portrait photography" are specific nouns — keywords can match
        w = adaptive_keyword_weight("Any tips for portrait photography?", self.BASE)
        assert w == self.BASE

    def test_vague_any_advice_with_nouns_keeps_weight(self):
        w = adaptive_keyword_weight("Any advice on making cocktails at home?", self.BASE)
        assert w == self.BASE

    def test_vague_any_suggestions_no_nouns(self):
        w = adaptive_keyword_weight("Any suggestions?", self.BASE)
        assert w == pytest.approx(self.BASE * 0.3)

    def test_vague_what_recommend_no_nouns(self):
        w = adaptive_keyword_weight("What do you recommend?", self.BASE)
        assert w == pytest.approx(self.BASE * 0.3)

    def test_factual_when_did(self):
        w = adaptive_keyword_weight("When did we configure the PostgreSQL port?", self.BASE)
        assert w == self.BASE

    def test_factual_what_is_the(self):
        w = adaptive_keyword_weight("What is the database connection string?", self.BASE)
        assert w == self.BASE

    def test_factual_which_port(self):
        w = adaptive_keyword_weight("Which port is PostgreSQL running on?", self.BASE)
        assert w == self.BASE

    def test_short_preference_query_no_nouns(self):
        # Short query with preference indicator and few nouns → half weight
        w = adaptive_keyword_weight("I need a good one", self.BASE)
        assert w == pytest.approx(self.BASE * 0.5)

    def test_short_preference_with_nouns_keeps_weight(self):
        # Has specific nouns → keep full weight
        w = adaptive_keyword_weight("I need a good camera tripod", self.BASE)
        assert w == self.BASE

    def test_normal_query_untouched(self):
        w = adaptive_keyword_weight("How is the deployment process configured?", self.BASE)
        assert w == self.BASE

    def test_long_preference_not_reduced(self):
        # Long query with preference indicator — specific enough, keep weight
        long_q = "I recently bought a Sony A7R IV and I'm looking for compatible lenses that work well for portrait photography in low light"
        w = adaptive_keyword_weight(long_q, self.BASE)
        # >12 words, so _PREF_INDICATORS short-query path doesn't fire
        assert w == self.BASE

    def test_factual_overrides_vague(self):
        # "What is the" pattern is factual even with vague words
        w = adaptive_keyword_weight("What is the best tips database?", self.BASE)
        assert w == self.BASE

    def test_vague_with_specific_nouns_keeps_weight(self):
        # "chocolate chip cookies" are specific nouns worth matching
        w = adaptive_keyword_weight(
            "My chocolate chip cookies need something extra. Any advice?", self.BASE
        )
        assert w == self.BASE
