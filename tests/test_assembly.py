"""Tests for epimneme.assembly — deterministic post-retrieval context formatting."""

from datetime import date, datetime, timezone

import pytest
from epimneme.assembly import (
    Excerpt,
    excerpt_date,
    annotate_temporal,
    chronological_order,
    prune_superseded,
    select_k,
    budget_by_chars,
    group_by_session,
    expand_parents,
    assemble_context,
)


def _ex(text, score=0.5, **metadata):
    return Excerpt(text=text, score=score, metadata=metadata)


class TestExcerptDate:
    def test_slash_header(self):
        ex = _ex("[Date: 2023/05/20 (Sat) 09:05]\n[USER]: hi")
        assert excerpt_date(ex) == date(2023, 5, 20)

    def test_iso_header(self):
        ex = _ex("[Date: 2023-05-20]\n[USER]: hi")
        assert excerpt_date(ex) == date(2023, 5, 20)

    def test_tag_fallback(self):
        ex = _ex("no header here", tags=["2023/05/20 (Sat) 09:05"])
        assert excerpt_date(ex) == date(2023, 5, 20)

    def test_created_at_fallback(self):
        ex = _ex("no header here", created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert excerpt_date(ex) == date(2024, 1, 1)

    def test_no_date_anywhere(self):
        ex = _ex("no header here")
        assert excerpt_date(ex) is None


class TestAnnotateTemporal:
    def test_resolves_target_date_preamble(self):
        excerpts = [_ex("[Date: 2023/05/20 (Sat) 09:05]\n[USER]: bought sneakers")]
        annotated, preamble = annotate_temporal(
            excerpts, "What did I buy 10 days ago?", reference_date=date(2023, 5, 30)
        )
        assert preamble == "The question refers to approximately 2023/05/20."

    def test_no_preamble_without_relative_date(self):
        excerpts = [_ex("[Date: 2023/05/20 (Sat) 09:05]\n[USER]: hi")]
        _, preamble = annotate_temporal(excerpts, "What is my favorite color?", reference_date=date(2023, 5, 30))
        assert preamble is None

    def test_delta_annotation_before(self):
        excerpts = [_ex("[Date: 2023/05/20 (Sat) 09:05]\n[USER]: bought sneakers")]
        annotated, _ = annotate_temporal(excerpts, "10 days ago", reference_date=date(2023, 5, 30))
        assert "10 days before the question" in annotated[0].text

    def test_delta_annotation_same_day(self):
        excerpts = [_ex("[Date: 2023/05/30 (Tue) 09:05]\n[USER]: hi")]
        annotated, _ = annotate_temporal(excerpts, "anything", reference_date=date(2023, 5, 30))
        assert "on the day of the question" in annotated[0].text

    def test_undated_excerpt_untouched(self):
        excerpts = [_ex("no date here")]
        annotated, _ = annotate_temporal(excerpts, "10 days ago", reference_date=date(2023, 5, 30))
        assert annotated[0].text == "no date here"

    def test_no_reference_date_available_is_noop(self):
        excerpts = [_ex("no date anywhere")]
        annotated, preamble = annotate_temporal(excerpts, "10 days ago")
        assert preamble is None
        assert annotated[0].text == "no date anywhere"


class TestChronologicalOrder:
    def test_sorts_by_date_ascending(self):
        a = _ex("[Date: 2023/05/25 (Thu) 01:00]\nlater")
        b = _ex("[Date: 2023/05/20 (Sat) 01:00]\nearlier")
        ordered = chronological_order([a, b])
        assert ordered == [b, a]

    def test_stable_tiebreak_on_rank_for_missing_dates(self):
        a = _ex("no date a")
        b = _ex("no date b")
        ordered = chronological_order([a, b])
        assert ordered == [a, b]

    def test_undated_sorts_after_dated(self):
        dated = _ex("[Date: 2023/05/20 (Sat) 01:00]\ndated")
        undated = _ex("no date")
        ordered = chronological_order([undated, dated])
        assert ordered == [dated, undated]


class TestPruneSuperseded:
    def test_explicit_supersedes_drops_old(self):
        old = _ex("[Date: 2023/05/01]\nold fact", memory_id="old")
        new = _ex("[Date: 2023/05/20]\nnew fact", memory_id="new", supersedes="old")
        result = prune_superseded([old, new])
        ids = [ex.metadata.get("memory_id") for ex in result]
        assert "old" not in ids
        assert "new" in ids

    def test_no_link_keeps_both(self):
        a = _ex("[Date: 2023/05/01]\nfact a", memory_id="a")
        b = _ex("[Date: 2023/05/20]\nunrelated fact b", memory_id="b")
        result = prune_superseded([a, b])
        assert len(result) == 2

    def test_near_duplicate_different_dates_flags_older(self):
        older = _ex(
            "[Date: 2023/05/01 (Mon) 01:00]\n[USER]: my favorite color is blue and I like it a lot",
            memory_id="older",
        )
        newer = _ex(
            "[Date: 2023/05/20 (Sat) 01:00]\n[USER]: my favorite color is blue and I like it a lot",
            memory_id="newer",
        )
        result = prune_superseded([older, newer])
        by_id = {ex.metadata.get("memory_id"): ex for ex in result}
        assert "SUPERSEDED" in by_id["older"].text
        assert "SUPERSEDED" not in by_id["newer"].text
        # Both are kept — this is a heuristic flag, not a drop.
        assert len(result) == 2

    def test_divergent_entities_not_flagged(self):
        a = _ex("[Date: 2023/05/01 (Mon) 01:00]\n[USER]: I like the color red", memory_id="a")
        b = _ex("[Date: 2023/05/20 (Sat) 01:00]\n[USER]: I like the color blue", memory_id="b")
        result = prune_superseded([a, b])
        assert all("SUPERSEDED" not in ex.text for ex in result)


class TestSelectK:
    def test_counting_query_gets_wide_k(self):
        excerpts = [_ex(f"item {i}") for i in range(30)]
        selected = select_k(excerpts, "How many times did I mention my car?")
        assert len(selected) == 20

    def test_short_query_gets_narrow_k(self):
        excerpts = [_ex(f"item {i}") for i in range(30)]
        selected = select_k(excerpts, "My car color?")
        assert len(selected) == 5

    def test_default_query_gets_standard_k(self):
        excerpts = [_ex(f"item {i}") for i in range(30)]
        selected = select_k(excerpts, "What did I say about my car during our long conversation last week?")
        assert len(selected) == 10

    def test_respects_fewer_available_than_k(self):
        excerpts = [_ex("only one")]
        selected = select_k(excerpts, "short q")
        assert len(selected) == 1


class TestBudgetByChars:
    def test_keeps_all_under_budget(self):
        excerpts = [_ex("short")] * 3
        kept, truncated = budget_by_chars(excerpts, budget_chars=1000)
        assert len(kept) == 3
        assert not truncated

    def test_truncates_when_over_budget(self):
        excerpts = [_ex("x" * 100) for _ in range(5)]
        kept, truncated = budget_by_chars(excerpts, budget_chars=250)
        assert len(kept) < 5
        assert truncated

    def test_always_keeps_first_even_if_oversized(self):
        excerpts = [_ex("x" * 5000)]
        kept, truncated = budget_by_chars(excerpts, budget_chars=100)
        assert len(kept) == 1
        assert not truncated  # nothing was dropped — the only item was kept


class TestGroupBySession:
    def test_merges_same_session(self):
        a = _ex("[Date: 2023/05/20 (Sat) 01:00]\n[USER]: q1\n[ASSISTANT]: a1", session_id="s1", turn_index=0)
        b = _ex("[Date: 2023/05/20 (Sat) 02:00]\n[USER]: q2\n[ASSISTANT]: a2", session_id="s1", turn_index=1)
        grouped = group_by_session([a, b])
        assert len(grouped) == 1
        assert "q1" in grouped[0].text
        assert "q2" in grouped[0].text
        # Only one date header should remain
        assert grouped[0].text.count("[Date:") == 1

    def test_orders_by_turn_index_not_input_order(self):
        a = _ex("[Date: 2023/05/20]\nsecond turn", session_id="s1", turn_index=1)
        b = _ex("[Date: 2023/05/20]\nfirst turn", session_id="s1", turn_index=0)
        grouped = group_by_session([a, b])
        assert grouped[0].text.index("first turn") < grouped[0].text.index("second turn")

    def test_passthrough_without_session_id(self):
        a = _ex("standalone")
        grouped = group_by_session([a])
        assert grouped == [a]

    def test_different_sessions_not_merged(self):
        a = _ex("[Date: 2023/05/20]\nfrom s1", session_id="s1")
        b = _ex("[Date: 2023/05/21]\nfrom s2", session_id="s2")
        grouped = group_by_session([a, b])
        assert len(grouped) == 2


class TestExpandParents:
    def test_no_fetcher_is_noop(self):
        excerpts = [_ex("hit", session_id="s1", memory_id="a")]
        assert expand_parents(excerpts, None, "query") == excerpts

    def test_counting_query_skips_expansion(self):
        excerpts = [_ex("hit", session_id="s1", memory_id="a")]

        def fetcher(ex):
            return [_ex("neighbor", session_id="s1", memory_id="b")]

        result = expand_parents(excerpts, fetcher, "How many times did I mention this?")
        assert len(result) == 1

    def test_too_many_sessions_skips_expansion(self):
        excerpts = [
            _ex("hit", session_id=f"s{i}", memory_id=str(i)) for i in range(5)
        ]
        calls = []

        def fetcher(ex):
            calls.append(ex)
            return [_ex("neighbor", session_id=ex.metadata["session_id"], memory_id="new")]

        result = expand_parents(excerpts, fetcher, "narrow question")
        assert result == excerpts
        assert calls == []

    def test_splices_in_fetched_neighbors(self):
        hit = _ex("[Date: 2023/05/20]\nhit turn", session_id="s1", memory_id="a")

        def fetcher(ex):
            return [_ex("[Date: 2023/05/20]\nneighbor turn", session_id="s1", memory_id="b")]

        result = expand_parents([hit], fetcher, "narrow question")
        assert len(result) == 2
        ids = {ex.metadata.get("memory_id") for ex in result}
        assert ids == {"a", "b"}

    def test_does_not_duplicate_already_present_neighbor(self):
        hit = _ex("hit", session_id="s1", memory_id="a")
        already_present = _ex("already there", session_id="s1", memory_id="b")

        def fetcher(ex):
            return [_ex("dup of b", session_id="s1", memory_id="b")]

        result = expand_parents([hit, already_present], fetcher, "narrow question")
        ids = [ex.metadata.get("memory_id") for ex in result]
        assert ids.count("b") == 1

    def test_no_session_ids_present_is_noop(self):
        excerpts = [_ex("standalone", memory_id="a")]

        def fetcher(ex):
            return [_ex("should not appear")]

        result = expand_parents(excerpts, fetcher, "narrow question")
        assert result == excerpts


class TestAssembleContext:
    def test_parent_expansion_respects_budget(self):
        hit = _ex("[Date: 2023/05/20]\n" + "x" * 100, session_id="s1", memory_id="a", score=1.0)

        def fetcher(ex):
            return [
                _ex("[Date: 2023/05/20]\n" + "y" * 100, session_id="s1", memory_id=f"n{i}")
                for i in range(50)  # far more than any reasonable budget
            ]

        result = assemble_context(
            [hit], "narrow question", budget_chars=500, fetch_neighbors=fetcher
        )
        assert result.char_count <= 500
        assert result.truncated


    def test_end_to_end_smoke(self):
        excerpts = [
            _ex("[Date: 2023/05/20 (Sat) 09:05]\n[USER]: bought sneakers\n[ASSISTANT]: nice!", score=0.9),
            _ex("[Date: 2023/05/25 (Thu) 09:05]\n[USER]: unrelated chat", score=0.5),
        ]
        result = assemble_context(excerpts, "What did I buy 10 days ago?", reference_date=date(2023, 5, 30))
        assert "approximately 2023/05/20" in result.text
        assert result.excerpt_count == 2
        assert result.char_count == len(result.text)
        assert not result.truncated

    def test_respects_budget(self):
        excerpts = [_ex("[Date: 2023/05/20]\n" + "x" * 200, score=1.0 - i * 0.01) for i in range(20)]
        result = assemble_context(excerpts, "simple question", budget_chars=500)
        assert result.char_count <= 500
        assert result.truncated

    def test_empty_input(self):
        result = assemble_context([], "any question")
        assert result.excerpt_count == 0
        assert result.text == ""
        assert not result.truncated
