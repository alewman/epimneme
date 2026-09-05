"""Tests for engram.fusion — RRF fusion, proper noun extraction, preference terms, adaptive weights."""

from datetime import date, datetime, timezone

import pytest
from epimneme.core.models import Memory, MemoryKind, MemoryResult
from epimneme.fusion import (
    rrf_fuse, extract_proper_nouns, extract_preference_terms,
    adaptive_keyword_weight, apply_temporal_boost, _extract_memory_date,
    temporal_partition,
)


def _mr(mid: str, score: float = 0.0, content: str = "", tags: list[str] | None = None,
        created_at: datetime | None = None) -> MemoryResult:
    """Helper to build a MemoryResult with a given id and score."""
    kwargs = {"id": mid, "kind": MemoryKind.FACT, "content": content, "tags": tags or []}
    if created_at is not None:
        kwargs["created_at"] = created_at
    return MemoryResult(
        memory=Memory(**kwargs),
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


class TestExtractMemoryDate:
    """Regression tests for the LME slash-date parsing bug.

    Real LongMemEval haystacks encode dates as `[Date: 2023/05/20 (Sat) 09:05]`
    (slash-delimited, with a weekday/time suffix) — not ISO `YYYY-MM-DD`. The
    original regexes only matched ISO, so every benchmark memory fell through
    to `created_at` (ingestion wall-clock time), silently disabling the
    temporal boost on the exact data it was built for.
    """

    def test_slash_date_in_content(self):
        mr = _mr("a", content="[Date: 2023/05/20 (Sat) 09:05]\n[USER]: hi")
        assert _extract_memory_date(mr) == date(2023, 5, 20)

    def test_iso_date_in_content_still_works(self):
        mr = _mr("a", content="[Date: 2023-05-20]\n[USER]: hi")
        assert _extract_memory_date(mr) == date(2023, 5, 20)

    def test_slash_date_tag(self):
        mr = _mr("a", content="no date here", tags=["benchmark", "lme", "2023/05/20 (Sat) 09:05"])
        assert _extract_memory_date(mr) == date(2023, 5, 20)

    def test_iso_date_tag_still_works(self):
        mr = _mr("a", content="no date here", tags=["2023-05-20"])
        assert _extract_memory_date(mr) == date(2023, 5, 20)

    def test_falls_back_to_created_at(self):
        ca = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mr = _mr("a", content="no date here", created_at=ca)
        assert _extract_memory_date(mr) == date(2024, 1, 1)

    def test_no_date_anywhere_returns_none(self):
        mr = _mr("a", content="x")
        # created_at always has a default_factory value in practice, but guard
        # the None branch explicitly since the function checks `is not None`.
        mr.memory.created_at = None
        assert _extract_memory_date(mr) is None


class TestApplyTemporalBoost:
    """Confirms the boost actually discriminates once dates parse correctly."""

    def test_boosts_memory_near_target_date(self):
        # Reference date is the latest logical date in the result set (May 30).
        # Query asks about "10 days ago" → target = May 20.
        near = _mr("near", score=0.5, content="[Date: 2023/05/20 (Sat) 09:05]\n[USER]: bought sneakers")
        far = _mr("far", score=0.5, content="[Date: 2023/05/30 (Tue) 09:05]\n[USER]: unrelated")
        fused = {"near": near, "far": far}

        apply_temporal_boost(fused, "What did I buy 10 days ago?", reference_date=date(2023, 5, 30))

        assert fused["near"].score > fused["far"].score

    def test_noop_without_relative_date_expression(self):
        mr = _mr("a", score=0.5, content="[Date: 2023/05/20 (Sat) 09:05]\n[USER]: hi")
        fused = {"a": mr}
        apply_temporal_boost(fused, "What is my favorite color?", reference_date=date(2023, 5, 30))
        assert fused["a"].score == 0.5


class TestTemporalPartition:
    """Structural reorder: in-window candidates first, order preserved within each group."""

    def test_promotes_in_window_ahead_of_out_of_window(self):
        far_but_ranked_first = _mr("far", score=0.9, content="[Date: 2023/06/15 (Thu) 09:05]\nunrelated")
        near_but_ranked_second = _mr("near", score=0.5, content="[Date: 2023/05/20 (Sat) 09:05]\nsneakers")
        results = [far_but_ranked_first, near_but_ranked_second]

        partitioned = temporal_partition(results, target_date=date(2023, 5, 20), sigma_days=3.5)

        assert [r.memory.id for r in partitioned] == ["near", "far"]

    def test_preserves_relative_order_within_each_partition(self):
        a = _mr("a", content="[Date: 2023/05/19 (Fri) 09:05]\nx")  # in-window
        b = _mr("b", content="[Date: 2023/05/21 (Sun) 09:05]\ny")  # in-window
        c = _mr("c", content="[Date: 2023/07/01 (Sat) 09:05]\nz")  # out-of-window
        d = _mr("d", content="[Date: 2023/08/01 (Tue) 09:05]\nw")  # out-of-window
        partitioned = temporal_partition([a, c, b, d], target_date=date(2023, 5, 20), sigma_days=3.5)
        assert [r.memory.id for r in partitioned] == ["a", "b", "c", "d"]

    def test_undated_results_treated_as_out_of_window(self):
        dated = _mr("dated", content="[Date: 2023/05/20 (Sat) 09:05]\nx")
        undated = _mr("undated", content="no date here")
        partitioned = temporal_partition([undated, dated], target_date=date(2023, 5, 20), sigma_days=3.5)
        assert [r.memory.id for r in partitioned] == ["dated", "undated"]

    def test_keeps_all_candidates(self):
        results = [_mr(str(i), content=f"no date {i}") for i in range(5)]
        partitioned = temporal_partition(results, target_date=date(2023, 5, 20), sigma_days=3.5)
        assert len(partitioned) == 5
