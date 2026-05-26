"""Tests for engram.dedup — SimHash near-duplicate detection + entity isolation."""


from epimneme.dedup import (
    compute_simhash,
    entities_diverge,
    extract_distinctive_nouns,
    hamming_distance,
    is_near_duplicate,
)


# ── compute_simhash ──────────────────────────────────────────────────────────


class TestComputeSimhash:
    def test_empty_string(self):
        assert compute_simhash("") == 0

    def test_whitespace_only(self):
        assert compute_simhash("   ") == 0

    def test_deterministic(self):
        """Same input should always produce the same hash."""
        h1 = compute_simhash("the quick brown fox jumps over the lazy dog")
        h2 = compute_simhash("the quick brown fox jumps over the lazy dog")
        assert h1 == h2

    def test_case_insensitive(self):
        """SimHash lowercases tokens, so case shouldn't matter."""
        h1 = compute_simhash("Hello World")
        h2 = compute_simhash("hello world")
        assert h1 == h2

    def test_returns_integer(self):
        h = compute_simhash("some text content")
        assert isinstance(h, int)

    def test_different_texts_differ(self):
        """Completely different texts should produce different hashes."""
        h1 = compute_simhash("the quick brown fox jumps over the lazy dog")
        h2 = compute_simhash("quantum physics molecular biology chemistry")
        assert h1 != h2

    def test_similar_texts_close(self):
        """Similar texts should have low Hamming distance."""
        h1 = compute_simhash("PostgreSQL runs on port 5432 by default")
        h2 = compute_simhash("PostgreSQL runs on port 5432 as default")
        dist = hamming_distance(h1, h2)
        assert dist < 10  # Should be close

    def test_single_token(self):
        h = compute_simhash("hello")
        assert h != 0
        assert isinstance(h, int)


# ── hamming_distance ─────────────────────────────────────────────────────────


class TestHammingDistance:
    def test_identical(self):
        assert hamming_distance(0, 0) == 0
        assert hamming_distance(42, 42) == 0
        assert hamming_distance(0xFFFF, 0xFFFF) == 0

    def test_one_bit_different(self):
        assert hamming_distance(0b1000, 0b0000) == 1
        assert hamming_distance(0b0001, 0b0000) == 1

    def test_all_bits_different_8bit(self):
        assert hamming_distance(0xFF, 0x00) == 8

    def test_known_value(self):
        # 0b1010 vs 0b0101 → differ in all 4 lower bits
        assert hamming_distance(0b1010, 0b0101) == 4

    def test_symmetric(self):
        a, b = 123456, 654321
        assert hamming_distance(a, b) == hamming_distance(b, a)

    def test_non_negative(self):
        assert hamming_distance(999, 0) >= 0


# ── is_near_duplicate ────────────────────────────────────────────────────────


class TestIsNearDuplicate:
    def test_identical_hashes(self):
        assert is_near_duplicate(42, 42) is True

    def test_within_threshold(self):
        # Differ by 2 bits (threshold default=3)
        assert is_near_duplicate(0b1100, 0b1111) is True  # distance 2

    def test_exactly_at_threshold(self):
        # Differ by exactly 3 bits (threshold=3)
        # 0b1110 ^ 0b1001 = 0b0111 → 3 bits differ
        assert is_near_duplicate(0b1110, 0b1001, threshold=3) is True

    def test_beyond_threshold(self):
        # Differ by 4 bits with threshold=3
        # 0b1111 ^ 0b0000 = 0b1111 → 4 bits
        assert is_near_duplicate(0b1111, 0b0000, threshold=3) is False

    def test_custom_threshold(self):
        # Differ by 5 bits
        h1, h2 = 0b11111, 0b00000  # 5 bits different
        assert is_near_duplicate(h1, h2, threshold=4) is False
        assert is_near_duplicate(h1, h2, threshold=5) is True

    def test_real_text_duplicates(self):
        """Near-identical texts should have low Hamming distance."""
        h1 = compute_simhash("The server runs on PostgreSQL 16 with pgvector")
        h2 = compute_simhash("The server runs on PostgreSQL 16 with pgvector extension")
        dist = hamming_distance(h1, h2)
        # Similar sentences — distance should be relatively small
        assert dist < 20  # generous threshold for test stability

    def test_real_text_different(self):
        """Completely different texts should NOT be duplicates."""
        h1 = compute_simhash("The server runs on PostgreSQL 16 with pgvector")
        h2 = compute_simhash("My favorite color is blue and I like pizza")
        assert is_near_duplicate(h1, h2, threshold=3) is False


# ── Entity isolation ─────────────────────────────────────────────────────────


class TestExtractDistinctiveNouns:
    def test_basic_extraction(self):
        nouns = extract_distinctive_nouns("I like the red Sony camera")
        assert "red" in nouns
        assert "sony" in nouns
        assert "camera" in nouns

    def test_filters_stopwords(self):
        nouns = extract_distinctive_nouns("I like the one with good quality")
        assert "like" not in nouns
        assert "the" not in nouns
        assert "good" not in nouns

    def test_empty_string(self):
        assert extract_distinctive_nouns("") == frozenset()

    def test_returns_frozenset(self):
        result = extract_distinctive_nouns("Sony camera equipment")
        assert isinstance(result, frozenset)


class TestEntitiesDiverge:
    def test_same_content(self):
        assert entities_diverge("I like the red Sony camera", "I prefer the red Sony camera") is False

    def test_different_entities(self):
        assert entities_diverge("I like the red one", "I like the blue one") is True

    def test_completely_different(self):
        assert entities_diverge(
            "I bought a Sony A7R IV camera",
            "I planted tomatoes in my garden"
        ) is True

    def test_similar_entities(self):
        # Share "port", "5432" — Jaccard 2/6 ≈ 0.33 ≥ 0.3 threshold
        assert entities_diverge(
            "The PostgreSQL database runs well on Linux",
            "PostgreSQL database performance is good on Linux servers"
        ) is False

    def test_same_topic_different_wording(self):
        # "database", "port", "5432" shared — should NOT diverge
        assert entities_diverge(
            "The database uses port 5432",
            "PostgreSQL listens on port 5432"
        ) is False

    def test_empty_content(self):
        # Can't determine entities — default to "same" (no diverge)
        assert entities_diverge("", "something here") is False
        assert entities_diverge("a b", "c d") is False  # all too short
