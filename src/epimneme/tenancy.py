"""Tenancy primitives — owners, and the names an owner may not claim.

An *owner* is the tenant boundary. Every project belongs to exactly one owner,
and project names are unique only *within* an owner: two tenants can both have
a project called ``peoplesoft`` without ever seeing each other's memories.

Before this existed, ``projects.name`` was globally unique, so the first key to
claim a name owned it for the whole installation and any other key granted that
name read the same rows.

Two ids are derived rather than stored, so nothing has to look them up:

* ``DEFAULT_OWNER_ID`` is the owner every pre-tenancy row was migrated onto.
* ``global_project_id(owner)`` is the per-owner home for memories that used to
  carry ``project_id IS NULL``. Giving them a real project is what lets every
  query filter on ``project_id``, with no NULL branch left to fail open.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_OWNER_ID = "00000000-0000-0000-0000-000000000000"
DEFAULT_OWNER_NAME = "default"

# The per-owner project that holds what used to be project_id IS NULL.
GLOBAL_PROJECT_NAME = "__global__"

# Sentinel the graph tables used for unscoped entities before owners existed.
LEGACY_GLOBAL_PROJECT_ID = "__global__"

#: Names no key may claim. ``*`` is the wildcard *grant* in ``api_keys.projects``
#: and must never also name a row — a project literally called ``*`` existed in
#: production before this check, created by an agent whose grant list leaked
#: into a claim. ``__global__`` is reserved for :func:`global_project_id`.
RESERVED_PROJECT_NAMES = frozenset({"*", GLOBAL_PROJECT_NAME})


#: Sentinel for the admin/maintenance callers that genuinely want every tenant.
#: Distinct from ``None``, which means "whichever owner is being served".
ALL_OWNERS = "__all_owners__"


def global_project_id(owner_id: str) -> str:
    """Return the deterministic id of an owner's global project."""
    return f"global-{owner_id}"


#: Stand-in id for a project name that does not exist in the caller's owner.
#: A read scoped to an unknown name must return nothing — before owners, the
#: unresolved name fell through to ``project_id=None``, which meant *every*
#: project, so a typo widened the search instead of narrowing it.
NO_SUCH_PROJECT = "__no_such_project__"


class ReservedProjectName(ValueError):
    """Raised when a caller tries to claim or create a reserved project name.

    A subclass of ValueError so existing ``except ValueError`` paths still
    catch it, and a distinct type so the API can answer 400 rather than 500.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"'{name}' is a reserved project name and cannot be used")
        self.name = name


def is_global_project_id(project_id: Optional[str]) -> bool:
    """True if ``project_id`` is some owner's global project."""
    return bool(project_id) and project_id.startswith("global-")


def is_reserved_project_name(name: str) -> bool:
    """True if ``name`` may not be claimed as a project.

    Rejects the reserved words, the empty/whitespace name, and anything using
    the ``global-`` id prefix, so a claim can never collide with a derived id.
    """
    if not name or not name.strip():
        return True
    if name in RESERVED_PROJECT_NAMES:
        return True
    return name.startswith("global-")


# ── Ambient owner ────────────────────────────────────────────────────────────
# The owner is a property of the *caller*, not of any one call, and it has to
# reach every project-name lookup in the request. Threading it through ~40 route
# signatures and ~15 manager signatures would work, but one missed call site is
# a cross-tenant read, and a missed call site is invisible.
#
# So it rides a contextvar, set at the two places an AuthContext is minted
# (get_auth and get_mcp_auth) and read by the manager when it resolves a name.
# Anything that needs a different owner — a maintenance job sweeping all
# tenants — passes owner_id explicitly instead.
#
# asyncio copies the context per task, so concurrent requests do not share this.

from contextvars import ContextVar  # noqa: E402

_current_owner: ContextVar[str] = ContextVar(
    "epimneme_current_owner", default=DEFAULT_OWNER_ID
)


def current_owner() -> str:
    """The owner of the caller being served, or the default owner."""
    return _current_owner.get()


def set_current_owner(owner_id: str):
    """Bind the owner for this task. Returns the token to reset with."""
    return _current_owner.set(owner_id or DEFAULT_OWNER_ID)


def reset_current_owner(token) -> None:
    _current_owner.reset(token)
