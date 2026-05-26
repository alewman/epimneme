"""Tests for engram.core.models — pure unit tests, no DB required."""

from epimneme.core.models import (
    ContextBundle,
    Entity,
    EntityKind,
    EntityResult,
    Memory,
    MemoryKind,
    MemoryResult,
    Project,
    Relationship,
    Session,
)


# ── MemoryKind / EntityKind ──────────────────────────────────────────────────

class TestEnums:
    def test_memory_kind_values(self):
        assert MemoryKind("fact") == MemoryKind.FACT
        assert MemoryKind("decision") == MemoryKind.DECISION
        assert MemoryKind("procedure") == MemoryKind.PROCEDURE
        assert MemoryKind("issue") == MemoryKind.ISSUE

    def test_entity_kind_values(self):
        assert EntityKind("file") == EntityKind.FILE
        assert EntityKind("module") == EntityKind.MODULE
        assert EntityKind("concept") == EntityKind.CONCEPT


# ── Memory model ─────────────────────────────────────────────────────────────

class TestMemory:
    def test_defaults(self):
        m = Memory(kind=MemoryKind.FACT, content="test")
        assert m.id  # auto-generated
        assert m.confidence == 1.0
        assert m.obsolete is False
        assert m.tags == []

    def test_custom_fields(self):
        m = Memory(
            kind=MemoryKind.DECISION,
            content="chose HNSW",
            subject="indexing",
            confidence=0.8,
            tags=["db", "perf"],
        )
        assert m.kind == MemoryKind.DECISION
        assert m.subject == "indexing"
        assert m.confidence == 0.8
        assert len(m.tags) == 2


# ── ContextBundle.to_prompt ──────────────────────────────────────────────────

class TestContextBundle:
    def _make_memory(self, content: str, kind: MemoryKind = MemoryKind.FACT, **kw) -> Memory:
        return Memory(kind=kind, content=content, **kw)

    def test_empty_bundle(self):
        bundle = ContextBundle()
        assert bundle.to_prompt() == ""

    def test_project_header(self):
        bundle = ContextBundle(
            project=Project(name="test-proj", description="A test project", path="/tmp")
        )
        prompt = bundle.to_prompt()
        assert "## Project: test-proj" in prompt
        assert "A test project" in prompt
        assert "Path: /tmp" in prompt

    def test_handoff_included(self):
        session = Session(project_id="p1", task="fix bug")
        session.summary = "Fixed the auth bug"
        session.handoff = "Need to add tests"
        bundle = ContextBundle(last_session=session)
        prompt = bundle.to_prompt()
        assert "Fixed the auth bug" in prompt
        assert "Need to add tests" in prompt

    def test_dedup_across_sections(self):
        m = self._make_memory("shared memory", kind=MemoryKind.DECISION)
        bundle = ContextBundle(
            recent_decisions=[m],
            relevant_memories=[MemoryResult(memory=m, score=0.9, source="semantic")],
        )
        prompt = bundle.to_prompt()
        # Should appear only once (in decisions, not again in relevant)
        assert prompt.count("shared memory") == 1

    def test_token_budget_limits_output(self):
        # Create many memories to exceed budget
        decisions = [self._make_memory(f"decision {i}" * 50, kind=MemoryKind.DECISION) for i in range(100)]
        bundle = ContextBundle(recent_decisions=decisions)
        prompt = bundle.to_prompt(max_tokens=100)  # ~400 chars
        assert len(prompt) <= 500  # Some overhead allowed

    def test_long_memory_truncated(self):
        m = self._make_memory("x" * 2000, kind=MemoryKind.FACT)
        bundle = ContextBundle(
            relevant_memories=[MemoryResult(memory=m, score=0.9, source="semantic")]
        )
        prompt = bundle.to_prompt()
        # The 2000-char content should be truncated to ~1500
        assert "…" in prompt

    def test_priority_ordering(self):
        """Decisions should appear before relevant memories."""
        d = self._make_memory("important decision", kind=MemoryKind.DECISION)
        r = self._make_memory("relevant context", kind=MemoryKind.FACT)
        bundle = ContextBundle(
            recent_decisions=[d],
            relevant_memories=[MemoryResult(memory=r, score=0.9, source="semantic")],
        )
        prompt = bundle.to_prompt()
        pos_decision = prompt.index("important decision")
        pos_relevant = prompt.index("relevant context")
        assert pos_decision < pos_relevant

    def test_all_sections_populated(self):
        bundle = ContextBundle(
            project=Project(name="p"),
            last_session=Session(project_id="p", task="t", summary="s"),
            recent_decisions=[self._make_memory("d", kind=MemoryKind.DECISION)],
            known_issues=[self._make_memory("i", kind=MemoryKind.ISSUE)],
            procedures=[self._make_memory("proc", kind=MemoryKind.PROCEDURE, subject="deploy")],
            preferences=[self._make_memory("pref", kind=MemoryKind.PREFERENCE)],
            relevant_memories=[MemoryResult(memory=self._make_memory("ctx"), score=0.8, source="semantic")],
            related_entities=[
                EntityResult(
                    entity=Entity(name="auth", kind=EntityKind.MODULE),
                    relationships=[Relationship(from_entity="auth", to_entity="db", relation="uses")],
                )
            ],
        )
        prompt = bundle.to_prompt(max_tokens=10000)
        for section in ["Key Decisions", "Known Issues", "Procedures", "User Preferences", "Relevant Context", "Key Entities"]:
            assert section in prompt, f"Missing section: {section}"


# ── Entity / Relationship ────────────────────────────────────────────────────

class TestEntity:
    def test_defaults(self):
        e = Entity(name="test.py", kind=EntityKind.FILE)
        assert e.id
        assert e.properties == {}
        assert e.project_id is None


class TestRelationship:
    def test_creation(self):
        r = Relationship(from_entity="a", to_entity="b", relation="uses")
        assert r.from_entity == "a"
        assert r.to_entity == "b"
        assert r.relation == "uses"
        assert r.properties == {}
