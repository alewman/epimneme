"""Tests for engram.skills — MCP resources, prompts, and instructions."""

from __future__ import annotations

import unittest

from epimneme.skills import INSTRUCTIONS, register_skills


class TestInstructions(unittest.TestCase):
    """Validate the instructions string sent to MCP clients on connect."""

    def test_instructions_is_nonempty_string(self):
        self.assertIsInstance(INSTRUCTIONS, str)
        self.assertGreater(len(INSTRUCTIONS), 100)

    def test_instructions_contains_all_tools(self):
        """Every MCP tool should be mentioned in the instructions."""
        tools = [
            "session_start",
            "session_end",
            "remember",
            "recall",
            "project_status",
            "entity_track",
            "entity_relate",
            "entity_explore",
        ]
        for tool in tools:
            with self.subTest(tool=tool):
                self.assertIn(tool, INSTRUCTIONS)

    def test_instructions_contains_memory_kinds(self):
        kinds = ["fact", "decision", "procedure", "pattern", "preference", "issue"]
        for kind in kinds:
            with self.subTest(kind=kind):
                self.assertIn(kind, INSTRUCTIONS)

    def test_instructions_mentions_resources(self):
        """Should tell agents they can browse resources."""
        self.assertIn("epimneme://", INSTRUCTIONS)


class TestResourceRegistration(unittest.TestCase):
    """Validate that resources register correctly on a FastMCP instance."""

    @classmethod
    def setUpClass(cls):
        from mcp.server.fastmcp import FastMCP

        cls.mcp = FastMCP("test-epimneme")
        register_skills(cls.mcp)

    def test_skills_overview_registered(self):
        """The master skills index should be a registered resource."""
        # list_resources is async, but we can check internal state
        resources = self.mcp._resource_manager._resources
        uris = [str(uri) for uri in resources.keys()]
        self.assertIn("epimneme://skills", uris)

    def test_all_recipes_registered(self):
        """All recipe resources should be registered."""
        resources = self.mcp._resource_manager._resources
        uris = [str(uri) for uri in resources.keys()]
        expected = [
            "epimneme://recipes/session-lifecycle",
            "epimneme://recipes/memory-kinds",
            "epimneme://recipes/knowledge-graph",
            "epimneme://recipes/cross-project",
            "epimneme://recipes/best-practices",
        ]
        for uri in expected:
            with self.subTest(uri=uri):
                self.assertIn(uri, uris)

    def test_resource_count(self):
        """Should have exactly 6 resources (1 overview + 5 recipes)."""
        resources = self.mcp._resource_manager._resources
        self.assertEqual(len(resources), 6)

    def test_resources_have_descriptions(self):
        """Every resource should have a description."""
        resources = self.mcp._resource_manager._resources
        for uri, resource in resources.items():
            with self.subTest(uri=str(uri)):
                self.assertTrue(
                    resource.description,
                    f"Resource {uri} missing description",
                )

    def test_resources_are_markdown(self):
        """Every resource should serve text/markdown."""
        resources = self.mcp._resource_manager._resources
        for uri, resource in resources.items():
            with self.subTest(uri=str(uri)):
                self.assertEqual(resource.mime_type, "text/markdown")


class TestPromptRegistration(unittest.TestCase):
    """Validate that prompts register correctly on a FastMCP instance."""

    @classmethod
    def setUpClass(cls):
        from mcp.server.fastmcp import FastMCP

        cls.mcp = FastMCP("test-epimneme")
        register_skills(cls.mcp)

    def test_all_prompts_registered(self):
        """All expected prompts should be registered."""
        prompts = self.mcp._prompt_manager._prompts
        expected = ["onboard-project", "knowledge-audit", "session-handoff"]
        for name in expected:
            with self.subTest(name=name):
                self.assertIn(name, prompts)

    def test_prompt_count(self):
        """Should have exactly 3 prompts."""
        prompts = self.mcp._prompt_manager._prompts
        self.assertEqual(len(prompts), 3)

    def test_prompts_have_descriptions(self):
        """Every prompt should have a description."""
        prompts = self.mcp._prompt_manager._prompts
        for name, prompt in prompts.items():
            with self.subTest(name=name):
                self.assertTrue(
                    prompt.description,
                    f"Prompt {name} missing description",
                )


class TestResourceContent(unittest.TestCase):
    """Validate the actual content of recipe resources."""

    def test_session_recipe_covers_lifecycle(self):
        from epimneme.skills import _RECIPE_SESSION

        for keyword in ["session_start", "remember", "recall", "session_end", "handoff"]:
            with self.subTest(keyword=keyword):
                self.assertIn(keyword, _RECIPE_SESSION)

    def test_memory_kinds_recipe_covers_all_kinds(self):
        from epimneme.skills import _RECIPE_MEMORY_KINDS

        for kind in ["fact", "decision", "procedure", "pattern", "preference", "issue"]:
            with self.subTest(kind=kind):
                self.assertIn(f"## `{kind}`", _RECIPE_MEMORY_KINDS)

    def test_knowledge_graph_recipe_covers_operations(self):
        from epimneme.skills import _RECIPE_KNOWLEDGE_GRAPH

        for op in ["entity_track", "entity_relate", "entity_explore"]:
            with self.subTest(op=op):
                self.assertIn(op, _RECIPE_KNOWLEDGE_GRAPH)

    def test_best_practices_recipe_has_sections(self):
        from epimneme.skills import _RECIPE_BEST_PRACTICES

        for section in ["What to Remember", "Writing Good Memories", "Session Handoff", "Recall Strategies"]:
            with self.subTest(section=section):
                self.assertIn(section, _RECIPE_BEST_PRACTICES)


class TestPromptContent(unittest.TestCase):
    """Validate that prompts produce correct parameterized output."""

    @classmethod
    def setUpClass(cls):
        from mcp.server.fastmcp import FastMCP

        cls.mcp = FastMCP("test-epimneme")
        register_skills(cls.mcp)

    def test_onboard_prompt_includes_project_name(self):
        prompt_fn = self.mcp._prompt_manager._prompts["onboard-project"].fn
        result = prompt_fn(project_name="my-app", description="A web application")
        self.assertIn("my-app", result)
        self.assertIn("A web application", result)
        self.assertIn("session_start", result)
        self.assertIn("entity_track", result)

    def test_knowledge_audit_includes_project_name(self):
        prompt_fn = self.mcp._prompt_manager._prompts["knowledge-audit"].fn
        result = prompt_fn(project_name="epimneme")
        self.assertIn("epimneme", result)
        self.assertIn("project_status", result)
        self.assertIn("recall", result)

    def test_session_handoff_includes_session_id(self):
        prompt_fn = self.mcp._prompt_manager._prompts["session-handoff"].fn
        result = prompt_fn(session_id="abc-123", project_name="test-proj")
        self.assertIn("abc-123", result)
        self.assertIn("test-proj", result)
        self.assertIn("session_end", result)


if __name__ == "__main__":
    unittest.main()
