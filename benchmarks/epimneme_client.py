"""Engram REST API client for benchmarking.

Thin async wrapper around epimneme's HTTP API. Handles:
  - Memory creation (single + bulk)
  - Search (with project scoping)
  - Project creation and cleanup
  - Connection pooling via aiohttp
"""

from __future__ import annotations

import asyncio
import aiohttp
from dataclasses import dataclass, field

_RETRY_DELAYS = (1.0, 2.0, 5.0)  # seconds between retries (3 attempts total)
_RETRY_ERRORS = (
    aiohttp.ClientConnectorError,
    aiohttp.ServerConnectionError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ClientOSError,
    ConnectionRefusedError,
)


async def _with_retry(coro_fn):
    """Retry *coro_fn()* up to len(_RETRY_DELAYS)+1 times on transient errors."""
    last_exc: Exception | None = None
    for attempt, delay in enumerate((_RETRY_DELAYS[0],) + _RETRY_DELAYS):
        if attempt > 0:
            await asyncio.sleep(delay)
        try:
            return await coro_fn()
        except _RETRY_ERRORS as exc:
            last_exc = exc
    raise last_exc


@dataclass
class EngramClient:
    """Async client for epimneme's REST API."""

    base_url: str = "http://localhost:8000"
    token: str = ""
    _session: aiohttp.ClientSession | None = field(default=None, repr=False)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=50)
            self._session = aiohttp.ClientSession(
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=600),
                connector=connector,
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Memory operations ────────────────────────────────────────────────

    async def create_memory(
        self,
        content: str,
        kind: str = "fact",
        project: str | None = None,
        subject: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Create a single memory. Returns the API response dict."""
        session = await self._ensure_session()
        payload: dict = {"content": content, "kind": kind}
        if project:
            payload["project"] = project
        if subject:
            payload["subject"] = subject
        if tags:
            payload["tags"] = tags
        async with session.post(f"{self.base_url}/api/memories", json=payload) as resp:
            return await resp.json()

    async def bulk_create(
        self,
        memories: list[dict],
        project: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Create up to 100 memories in one call.

        Each item in memories should have at minimum {"content": "..."}.
        Optional keys: kind, subject, tags, confidence.
        """
        session = await self._ensure_session()
        payload: dict = {"memories": memories}
        if project:
            payload["project"] = project
        if session_id:
            payload["session_id"] = session_id

        async def _do():
            async with session.post(f"{self.base_url}/api/bulk/memories", json=payload) as resp:
                return await resp.json()

        return await _with_retry(_do)

    # ── Search ───────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        project: str | None = None,
        limit: int = 50,
        kind: str | None = None,
        tags: str | None = None,
        reference_date: str | None = None,
    ) -> dict:
        """Search memories. Returns {count, total, has_more, results: [...]}."""
        session = await self._ensure_session()
        params: dict = {"query": query, "limit": limit}
        if project:
            params["project"] = project
        if kind:
            params["kind"] = kind
        if tags:
            params["tags"] = tags
        if reference_date:
            params["reference_date"] = reference_date

        async def _do():
            async with session.get(f"{self.base_url}/api/memories/search", params=params) as resp:
                if resp.status >= 500:
                    text = await resp.text()
                    raise aiohttp.ServerConnectionError(
                        f"Server error {resp.status}: {text[:200]}"
                    )
                return await resp.json()

        return await _with_retry(_do)

    # ── Project management ───────────────────────────────────────────────

    async def create_project(self, name: str) -> dict:
        session = await self._ensure_session()
        async with session.post(
            f"{self.base_url}/api/projects",
            json={"name": name},
        ) as resp:
            return await resp.json()

    async def list_projects(self) -> list[dict]:
        session = await self._ensure_session()
        async with session.get(f"{self.base_url}/api/projects") as resp:
            data = await resp.json()
            return data.get("projects", [])

    async def delete_memory(self, memory_id: str, hard: bool = True) -> dict:
        async def _do():
            session = await self._ensure_session()
            params = {"hard": "true"} if hard else {}
            async with session.delete(
                f"{self.base_url}/api/memories/{memory_id}",
                params=params,
            ) as resp:
                return await resp.json()
        return await _with_retry(_do)

    async def search_all(
        self, query: str, project: str | None = None, limit: int = 500
    ) -> list[dict]:
        """Fetch all results up to limit (handles pagination)."""
        result = await self.search(query, project=project, limit=limit)
        return result.get("results", [])

    async def clear_project(self, project_name: str, batch_size: int = 100):
        """Delete all memories in a project by searching and deleting in batches."""
        deleted = 0
        while True:
            results = await self.search(
                query="*", project=project_name, limit=batch_size
            )
            items = results.get("results", [])
            if not items:
                break
            for item in items:
                await self.delete_memory(item["id"], hard=True)
                deleted += 1
        return deleted


# ── Sync convenience wrapper ─────────────────────────────────────────────


def run_sync(coro):
    """Run an async coroutine from sync code."""
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — can't use asyncio.run
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)
