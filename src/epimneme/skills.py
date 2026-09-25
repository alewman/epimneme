"""Self-documenting skills, recipes, and prompts for MCP-connected agents.

When an agent connects to engram via MCP, it receives:
  1. `instructions` — core usage summary (sent automatically on connect)
  2. Resources — browseable docs at epimneme://skills/* and epimneme://recipes/*
  3. Prompts — parameterized workflows agents can invoke

This replaces the need for per-workspace copilot-instructions.md files.
The skills travel WITH the service — any agent, any workspace, any IDE.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


# ── Instructions (system prompt sent to every client on connect) ─────────────

INSTRUCTIONS = """\
Engram — persistent memory for AI coding agents.

## Session Lifecycle (REQUIRED)
1. **Start**: Call `session_start(project=..., task=...)` FIRST in every conversation.
   Save the returned `session_id`.
2. **During**: Use `remember()` to store and `recall()` to retrieve knowledge.
3. **End**: Call `session_end(session_id=..., summary=..., handoff_notes=...)` before finishing.

## Tools Quick Reference
| Tool | Purpose |
|------|---------|
| `session_start` | Begin session, receive previous context + handoff notes |
| `session_end` | Close session with summary for next agent |
| `remember` | Store a memory (fact, decision, procedure, pattern, preference, issue) |
| `recall` | Semantic search across memories |
| `project_status` | Overview of a project's knowledge + stats |
| `entity_track` | Track a named entity in the knowledge graph |
| `entity_relate` | Create relationship between entities |
| `entity_explore` | Traverse the knowledge graph from an entity |

## Memory Kinds
| Kind | When to Use |
|------|-------------|
| `fact` | Discrete knowledge: CLI names, file locations, config details |
| `decision` | Why something was done a certain way — prevents re-litigating |
| `procedure` | How to do something: build commands, test commands, workflows |
| `pattern` | Recurring conventions or gotchas |
| `preference` | User's working style: commit habits, review preferences |
| `issue` | Known bugs, tech debt, limitations |

## Key Behaviors
- `session_start` returns context from previous sessions — READ IT before working.
- Pass `project` to scope memories. Use `"*"` or omit for cross-project knowledge.
- `remember()` accepts optional `session_id` and `tags` for better organization.
- `recall()` supports `query`, `kind`, `project`, and `tags` filters.
- The knowledge graph (`entity_track/relate/explore`) links concepts across memories.

## Reading Recalled Memory
`recall()` returns excerpts; how you prompt over them matters more than retrieval
quality. Two measured pitfalls: never place the final answer last under a token
cap (it gets truncated away), and never offer a blanket "say Unknown if not
found" for answers that must be computed (readers take it literally and decline).
See `epimneme://recipes/answering-from-recall`.

Browse `epimneme://skills/*` resources for detailed recipes.\
"""


# ── Resource Content ─────────────────────────────────────────────────────────

_RECIPE_SESSION = """\
# Session Lifecycle

Every conversation should follow this pattern:

## 1. Start Session
```
session_start(project="my-project", task="implement feature X")
```
Returns:
- Previous session summary and handoff notes
- Key decisions made in earlier sessions
- Known issues and procedures
- Relevant entity graph

**Read the response carefully** — it contains context that prevents you from
re-discovering things the previous agent already learned.

## 2. Work + Remember
As you discover important things during the session, store them:

```
remember(
    subject="database migration",
    content="Must run `alembic upgrade head` after pulling — schema changed in commit abc123",
    kind="procedure",
    project="my-project",
    session_id="...",         # from session_start
    tags="database,migration"
)
```

Use `recall()` when you need to know something:
```
recall(query="how to run database migrations", project="my-project")
```

## 3. End Session
```
session_end(
    session_id="...",
    summary="Added user authentication with JWT tokens. 45 tests passing.",
    handoff_notes="Next: add refresh token rotation. The auth middleware is in src/auth.py."
)
```

The `handoff_notes` field is critical — it tells the NEXT agent exactly where to pick up.

## Anti-Patterns
- Don't skip `session_start` — you'll miss important context
- Don't forget `session_end` — the next agent will start from scratch
- Don't store trivial things — focus on decisions, gotchas, and procedures
- Don't store code — store *knowledge about* code (locations, patterns, why)\
"""

_RECIPE_MEMORY_KINDS = """\
# Memory Kinds — When to Use Each

## `fact`
Discrete, stable knowledge. Things that are TRUE about the project.

Examples:
- "The CLI entry point is `romfarmer` defined in pyproject.toml"
- "PostgreSQL runs on port 5432 in the epimneme-db container"
- "The project uses Python 3.12 with FastAPI"

## `decision`
WHY something was done a certain way. The most valuable kind — prevents
re-litigating choices that were already made.

Examples:
- "Chose PostgreSQL over SQLite for concurrent multi-agent access"
- "Demo mode grants admin (not agent) because it's opt-in and behind OAuth in prod"
- "Using simhash for dedup instead of exact match — tolerates minor wording changes"

## `procedure`
HOW to do something. Step-by-step instructions that should be repeatable.

Examples:
- "Build engram: cd /docker/compose/dev && docker compose --env-file ... build engram"
- "Run tests: docker exec engram pip install pytest && docker exec engram python -m pytest tests/ -x -q"
- "Deploy: docker compose up -d, then verify with curl /health"

## `pattern`
Recurring conventions, naming rules, or gotchas that apply broadly.

Examples:
- "All test files use unittest.mock.patch for database isolation"
- "Docker container paths differ from host: /app/src/engram/ vs /docker/appdata/engram/src/engram/"
- "Always rm -rf __pycache__ before docker cp to avoid stale .pyc double-collection"

## `preference`
User's working style and preferences.

Examples:
- "Iterative verify-then-commit workflow: make changes → test → commit"
- "Prefers practical smoke tests alongside unit tests"
- "Don't create markdown documentation files unless explicitly asked"

## `issue`
Known bugs, tech debt, and limitations to be aware of.

Examples:
- "N+1 query in session_start — batched in commit 4236c7b but monitor for regression"
- "Dashboard onclick handlers need HTML entity escaping for JSON — see jsonAttr() helper"
- "Rate limiter is in-process only — no distributed support yet"\
"""

_RECIPE_KNOWLEDGE_GRAPH = """\
# Knowledge Graph — Entities and Relationships

The knowledge graph tracks named entities (files, modules, concepts, people)
and the relationships between them.

## Tracking Entities
```
entity_track(
    name="auth.py",
    type="file",
    project="epimneme",
    observations="Authentication middleware — handles Bearer, OAuth, demo mode"
)
```

Entity types are freeform but common ones include:
- `file`, `module`, `class`, `function` — code artifacts
- `concept`, `feature`, `component` — architectural pieces
- `person`, `service`, `database` — external things
- `project` — tracked automatically by session_start

## Creating Relationships
```
entity_relate(
    from_entity="server.py",
    to_entity="auth.py",
    relation="imports",
    project="epimneme"
)
```

Relationships are also freeform strings. Common patterns:
- `imports`, `depends_on`, `uses` — code dependencies
- `contains`, `part_of`, `belongs_to` — composition
- `tests`, `tested_by` — test relationships
- `configures`, `configured_by` — configuration
- `blocks`, `blocked_by` — issue tracking

## Exploring the Graph
```
entity_explore(entity="auth.py", depth=2, direction="both", project="epimneme")
```

Returns all connected entities up to the given depth, with their relationships.
Use this to understand how things connect before making changes.

## When to Track
- Track entities when you first encounter important files or concepts
- Create relationships when you discover dependencies or connections
- Don't over-track — focus on things that would help a future agent navigate\
"""

_RECIPE_CROSS_PROJECT = """\
# Cross-Project Knowledge

Engram supports multiple projects in the same instance. Memories are scoped
by project name.

## Project Scoping
- Pass `project="my-project"` to scope memories to a specific project
- Omit `project` or use `project="*"` for cross-project knowledge
- `recall()` without `project` searches across ALL projects

## Cross-Project Patterns
Store knowledge that applies everywhere without a project scope:
```
remember(
    subject="git workflow",
    content="Always verify with builds/tests before committing",
    kind="preference"
    # no project — applies globally
)
```

## Project Status
Get an overview of what's known about a project:
```
project_status(project="my-project")
```

Returns memory counts by kind, recent sessions, active entities, and known issues.\
"""

_RECIPE_ANSWERING = """\
# Answering Questions from Recalled Memory

`recall()` gives you excerpts. How you prompt your reader over them changes the
answer more than the retrieval does. Both rules below were measured on
LongMemEval-S (500 questions), and each was worth more than any retrieval change
we could find.

## 1. Never put the answer last under a token budget

A prompt of the shape *"list every matching item, then give the total on the last
line"* is a good way to make a model actually scan the context — and a good way
to lose the answer. Enumeration is long, generation caps are short, and the total
is the first thing a truncated response drops.

Measured: 44 of 133 counting answers were cut off mid-sentence before reaching
the total. Truncated answers were correct 39% of the time against 71% for
complete ones. Raising the cap was worth **+12 points** of end-to-end accuracy
(p=0.0001) — three quarters of the gain coming from exactly those truncated
questions.

- Give enumeration room, or ask for the answer **first** and the working after.
- The failure is silent. The response looks like a real answer, just a wrong one.
- Check for it directly: a response that ends without terminal punctuation is
  truncated, however plausible its beginning.

## 2. Do not offer "Unknown" when the answer must be computed

A blanket *"if the answer cannot be found in the excerpts, say Unknown"* is right
for lookups and wrong for anything derived. An elapsed time, a count, or an
ordering is never literally present in the text, so a careful reader takes the
instruction at its word and declines.

Measured: 32 of 49 temporal misses were refusals, and two thirds of them had the
dates needed sitting in the context. Narrowing the instruction to *"answer
Unknown only if the underlying events are absent"* cut refusals from 38 to 10 and
was worth **+7.5 points**.

- Scope the escape hatch to missing **evidence**, not to a missing **answer**.
- Say what to compute: *"find the events, then calculate the interval asked for"*.
- Ask for the unit — *"7 days"*, not *"7"*. Without it a reader answers with a
  bare number that is correct and unusable. This cost 53 of 133 answers in one run.
- For ordering questions, say to name the events. Otherwise a reader primed to
  compute will answer with positions: `2, 3, 1`.

## 3. Read the generations, not the score

Both bugs above are invisible in an aggregate metric and obvious in ten lines of
output. One of them first appeared as a *significant regression* that was really
a scorer rejecting `7` for `7 days`.
"""


_RECIPE_BEST_PRACTICES = """\
# Best Practices

## What to Remember
**High value** — always store:
- Decisions and their rationale
- Build/test/deploy procedures
- Known issues and workarounds
- Architecture insights that took effort to discover

**Medium value** — store if non-obvious:
- File locations and purposes
- Configuration details
- Naming conventions and patterns

**Low value** — usually skip:
- Code snippets (they go stale)
- Obvious things discoverable from the code
- Temporary debugging notes

## Writing Good Memories
- `subject` should be a short, searchable label: "auth.py demo mode", "build command"
- `content` should be self-contained — readable without other context
- `tags` help with filtering: "auth,security", "build,docker"
- Keep memories atomic — one fact/decision per memory, not a dump of everything

## Session Handoff
The `handoff_notes` in `session_end` is the most impactful field in the system.
Write it like a note to a colleague who's taking over your shift:
- What were you working on?
- What's done, what's not?
- What should they do next?
- Any gotchas or things to watch out for?

## Recall Strategies
- Start broad: `recall(query="authentication")` 
- Narrow with kind: `recall(query="authentication", kind="decision")`
- Narrow with project: `recall(query="authentication", project="epimneme")`
- Use tags: `recall(query="auth", tags="security")` if you tagged memories\
"""


# ── Resource & Prompt Registration ───────────────────────────────────────────


def register_skills(mcp_server: FastMCP) -> None:
    """Register all skill resources and prompts on the given FastMCP server."""

    # ── Resources ────────────────────────────────────────────────────────

    @mcp_server.resource(
        "epimneme://skills",
        name="skills-overview",
        title="Engram Skills Overview",
        description="Master index of all engram recipes and capabilities",
        mime_type="text/markdown",
    )
    def skills_overview() -> str:
        return (
            "# Engram Skills\n\n"
            "Persistent memory for AI coding agents. "
            "Browse these resources for detailed usage guidance.\n\n"
            "## Available Recipes\n\n"
            "| Resource | Description |\n"
            "|----------|-------------|\n"
            "| `epimneme://recipes/session-lifecycle` | Session start/work/end workflow |\n"
            "| `epimneme://recipes/memory-kinds` | When to use each memory kind |\n"
            "| `epimneme://recipes/knowledge-graph` | Entity tracking and relationships |\n"
            "| `epimneme://recipes/cross-project` | Multi-project knowledge patterns |\n"
            "| `epimneme://recipes/answering-from-recall` | Prompting a reader over recalled excerpts |\n"
            "| `epimneme://recipes/best-practices` | Tips for effective memory use |\n"
        )

    @mcp_server.resource(
        "epimneme://recipes/session-lifecycle",
        name="recipe-session-lifecycle",
        title="Session Lifecycle",
        description="How to start, work through, and end an engram session",
        mime_type="text/markdown",
    )
    def recipe_session() -> str:
        return _RECIPE_SESSION

    @mcp_server.resource(
        "epimneme://recipes/memory-kinds",
        name="recipe-memory-kinds",
        title="Memory Kinds Guide",
        description="When to use fact vs decision vs procedure vs pattern vs preference vs issue",
        mime_type="text/markdown",
    )
    def recipe_kinds() -> str:
        return _RECIPE_MEMORY_KINDS

    @mcp_server.resource(
        "epimneme://recipes/answering-from-recall",
        name="recipe-answering-from-recall",
        title="Answering Questions from Recalled Memory",
        description="How to prompt a reader over recalled excerpts — measured pitfalls "
                    "that cost more accuracy than any retrieval change",
        mime_type="text/markdown",
    )
    def recipe_answering() -> str:
        return _RECIPE_ANSWERING

    @mcp_server.resource(
        "epimneme://recipes/knowledge-graph",
        name="recipe-knowledge-graph",
        title="Knowledge Graph Guide",
        description="How to track entities and relationships",
        mime_type="text/markdown",
    )
    def recipe_graph() -> str:
        return _RECIPE_KNOWLEDGE_GRAPH

    @mcp_server.resource(
        "epimneme://recipes/cross-project",
        name="recipe-cross-project",
        title="Cross-Project Knowledge",
        description="Working with multiple projects in one engram instance",
        mime_type="text/markdown",
    )
    def recipe_cross_project() -> str:
        return _RECIPE_CROSS_PROJECT

    @mcp_server.resource(
        "epimneme://recipes/best-practices",
        name="recipe-best-practices",
        title="Best Practices",
        description="Tips for writing effective memories and handoffs",
        mime_type="text/markdown",
    )
    def recipe_best_practices() -> str:
        return _RECIPE_BEST_PRACTICES

    # ── Prompts ──────────────────────────────────────────────────────────

    @mcp_server.prompt(
        name="onboard-project",
        title="Onboard a New Project",
        description="Bootstrap engram tracking for a new project — creates initial memories and entities",
    )
    def onboard_project(project_name: str, description: str) -> str:
        return (
            f"Set up engram tracking for the project '{project_name}'.\n\n"
            f"## Steps\n"
            f"1. `session_start(project=\"{project_name}\", task=\"initial onboarding\")`\n"
            f"2. Store the project description:\n"
            f"   ```\n"
            f"   remember(\n"
            f"       subject=\"{project_name}\",\n"
            f"       content=\"{description}\",\n"
            f"       kind=\"fact\",\n"
            f"       project=\"{project_name}\"\n"
            f"   )\n"
            f"   ```\n"
            f"3. Track the project as an entity:\n"
            f"   ```\n"
            f"   entity_track(\n"
            f"       name=\"{project_name}\",\n"
            f"       type=\"project\",\n"
            f"       project=\"{project_name}\",\n"
            f"       observations=\"{description}\"\n"
            f"   )\n"
            f"   ```\n"
            f"4. Explore the codebase and `remember()` key architecture decisions,\n"
            f"   file locations, build procedures, and known issues.\n"
            f"5. Track important files/modules with `entity_track()` and connect\n"
            f"   them with `entity_relate()`.\n"
            f"6. `session_end()` with a summary of what was onboarded.\n"
        )

    @mcp_server.prompt(
        name="knowledge-audit",
        title="Audit Project Knowledge",
        description="Review what engram knows about a project — find gaps and stale info",
    )
    def knowledge_audit(project_name: str) -> str:
        return (
            f"Audit engram's knowledge about '{project_name}'.\n\n"
            f"## Steps\n"
            f"1. `session_start(project=\"{project_name}\", task=\"knowledge audit\")`\n"
            f"2. `project_status(project=\"{project_name}\")` — review memory counts and stats.\n"
            f"3. `recall(query=\"known issues\", project=\"{project_name}\", kind=\"issue\")` — check for stale issues.\n"
            f"4. `recall(query=\"procedures\", project=\"{project_name}\", kind=\"procedure\")` — verify procedures still work.\n"
            f"5. `recall(query=\"decisions\", project=\"{project_name}\", kind=\"decision\")` — check decisions are still valid.\n"
            f"6. `entity_explore(entity=\"{project_name}\", depth=2, project=\"{project_name}\")` — review the knowledge graph.\n"
            f"7. Store any corrections or updates via `remember()`.\n"
            f"8. `session_end()` with audit findings.\n\n"
            f"Focus on: outdated facts, resolved issues still marked as open, "
            f"missing procedures, and knowledge graph gaps.\n"
        )

    @mcp_server.prompt(
        name="session-handoff",
        title="Prepare Session Handoff",
        description="Checklist for ending a session with good handoff notes",
    )
    def session_handoff(session_id: str, project_name: str) -> str:
        return (
            f"Prepare to hand off session `{session_id}` for project '{project_name}'.\n\n"
            f"## Handoff Checklist\n"
            f"Before calling `session_end()`, make sure you've:\n\n"
            f"1. **Stored key decisions** — any choices you made during this session\n"
            f"   that a future agent should know about. Use `kind=\"decision\"`.\n"
            f"2. **Updated procedures** — if you discovered or changed how to\n"
            f"   build/test/deploy, store with `kind=\"procedure\"`.\n"
            f"3. **Logged issues** — any bugs, tech debt, or limitations found.\n"
            f"   Use `kind=\"issue\"`.\n"
            f"4. **Updated entities** — if you worked with files or modules,\n"
            f"   track them with `entity_track()` and `entity_relate()`.\n\n"
            f"## Writing the Handoff\n"
            f"```\n"
            f"session_end(\n"
            f"    session_id=\"{session_id}\",\n"
            f"    summary=\"<What you accomplished in 1-2 sentences>\",\n"
            f"    handoff_notes=\"<What the next agent should do, any gotchas, current state>\"\n"
            f")\n"
            f"```\n\n"
            f"Good handoff notes answer:\n"
            f"- What were you working on?\n"
            f"- What's done vs. what's remaining?\n"
            f"- What should the next agent do first?\n"
            f"- Any pitfalls or things to watch out for?\n"
        )
