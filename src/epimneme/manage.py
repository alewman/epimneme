"""CLI for managing Engram API keys and projects.

Usage (inside the container):
    docker exec engram python -m engram.manage create-key --name "my-agent" --role agent --projects myproject
    docker exec engram python -m engram.manage list-keys
    docker exec engram python -m engram.manage revoke-key --name "my-agent"
    docker exec engram python -m engram.manage list-projects
    docker exec engram python -m engram.manage create-project --name "myproject" --description "My project"
    docker exec engram python -m engram.manage stats
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from epimneme.core.config import default_config
from epimneme.stores.postgresql import PostgresStore


async def get_store() -> PostgresStore:
    """Create and open a store connection using env-based config."""
    config = default_config()
    store = PostgresStore(dsn=config.pg_dsn, embedding_dim=config.embedding_dim, pool_timeout=config.pg_pool_timeout)
    await store.open()
    return store


async def cmd_create_key(args: argparse.Namespace) -> None:
    store = await get_store()
    try:
        raw_key = await store.create_api_key(
            name=args.name,
            role=args.role,
            projects=args.projects.split(",") if args.projects else [],
            expires_in_days=args.expires_in_days,
        )
        print(f"\n{'=' * 60}")
        print("  API Key created successfully!")
        print(f"  Name:     {args.name}")
        print(f"  Role:     {args.role}")
        print(f"  Projects: {args.projects or '(none — assign later)'}")
        if args.expires_in_days:
            print(f"  Expires:  in {args.expires_in_days} days")
        print(f"{'=' * 60}")
        print(f"\n  Key: {raw_key}\n")
        print("  ⚠  Save this key now — it cannot be retrieved later!")
        print(f"{'=' * 60}\n")
    finally:
        await store.close()


async def cmd_list_keys(args: argparse.Namespace) -> None:
    store = await get_store()
    try:
        keys = await store.list_api_keys()
        if not keys:
            print("No API keys found.")
            return

        print(f"\n{'Name':<20} {'Role':<8} {'Prefix':<14} {'Projects':<30} {'Status':<10} {'Last Used'}")
        print("-" * 100)
        for k in keys:
            status = "active"
            if k.get("revoked_at"):
                status = "revoked"
            elif k.get("expires_at") and k["expires_at"] < datetime.now(timezone.utc):
                status = "expired"

            projects = ", ".join(k.get("projects", []))
            last_used = str(k.get("last_used", "never"))[:19] if k.get("last_used") else "never"
            print(f"{k['name']:<20} {k['role']:<8} {k['key_prefix']:<14} {projects:<30} {status:<10} {last_used}")
        print()
    finally:
        await store.close()


async def cmd_revoke_key(args: argparse.Namespace) -> None:
    store = await get_store()
    try:
        if await store.revoke_api_key(args.name):
            print(f"API key '{args.name}' revoked.")
        else:
            print(f"API key '{args.name}' not found or already revoked.")
    finally:
        await store.close()


async def cmd_list_projects(args: argparse.Namespace) -> None:
    store = await get_store()
    try:
        rows = await store.list_projects()
        if not rows:
            print("No projects found.")
            return

        print(f"\n{'Name':<25} {'ID':<38} {'Description':<40} {'Created'}")
        print("-" * 120)
        for p in rows:
            desc = (p.description or "")[:38]
            created = str(p.created_at)[:19]
            print(f"{p.name:<25} {p.id:<38} {desc:<40} {created}")
        print()
    finally:
        await store.close()


async def cmd_create_project(args: argparse.Namespace) -> None:
    store = await get_store()
    try:
        from epimneme.core.models import Project

        project = Project(name=args.name, path=args.path, description=args.description)
        await store.create_project(project)
        print(f"Project '{args.name}' created (id: {project.id})")
    finally:
        await store.close()


async def cmd_stats(args: argparse.Namespace) -> None:
    store = await get_store()
    try:
        mem_count = await store.get_memory_count()
        vec_count = await store.get_vector_count()
        projects = await store.list_projects()
        entities = await store.list_entities()
        keys = await store.list_api_keys()

        print("\n  Engram Statistics")
        print(f"  {'=' * 30}")
        print(f"  Memories:  {mem_count}")
        print(f"  Vectors:   {vec_count}")
        print(f"  Entities:  {len(entities)}")
        print(f"  Projects:  {len(projects)}")
        print(f"  API Keys:  {len(keys)}")
        print()
    finally:
        await store.close()


async def cmd_re_embed(args: argparse.Namespace) -> None:
    """Re-embed all memories with the current embedding model.

    Handles dimension changes: if the new model produces a different
    dimension, the column is altered and the HNSW index is rebuilt.
    """
    store = await get_store()
    config = default_config()
    batch_size = args.batch_size or 100

    try:
        # Load the embedding model
        print(f"Loading embedding model: {config.embedding_model}")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.embedding_model)
        new_dim = model.get_sentence_embedding_dimension()
        print(f"Model dimension: {new_dim} (configured: {config.embedding_dim})")

        if new_dim != config.embedding_dim:
            print(f"\n  ⚠  Dimension change detected: {config.embedding_dim} → {new_dim}")
            print("  This will drop the HNSW index, null all existing embeddings,")
            print("  alter the column, re-embed everything, and rebuild the index.")
            if not args.yes:
                confirm = input("\n  Continue? (y/N) ")
                if confirm.lower() != "y":
                    print("Aborted.")
                    return
            print("Altering embedding column...")
            await store.alter_embedding_dimension(new_dim)
            print(f"Column altered to vector({new_dim})")

        # Get all memories needing embeddings
        memories = await store.get_memories_needing_embedding(limit=100000)
        if not memories:
            # If dimension didn't change, get ALL non-obsolete memories
            memories = await store.get_all_memories(include_obsolete=False, limit=100000)

        total = len(memories)
        if total == 0:
            print("No memories to re-embed.")
            return

        print(f"\nRe-embedding {total} memories in batches of {batch_size}...")
        updated = 0

        for i in range(0, total, batch_size):
            batch = memories[i:i + batch_size]
            texts = [m.content for m in batch]
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            for m, emb in zip(batch, embeddings):
                await store.update_embedding(m.id, emb.tolist())
                updated += 1

            pct = (updated / total) * 100
            print(f"  [{updated}/{total}] {pct:.0f}%")

        # Rebuild HNSW index
        print("Rebuilding HNSW index...")
        await store.rebuild_hnsw_index()

        print(f"\nDone. Re-embedded {updated} memories with {config.embedding_model}.")
        if new_dim != config.embedding_dim:
            print(f"\n  ⚠  Update EPIMNEME_EMBEDDING_DIM={new_dim} in your .env file!")
    finally:
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="engram.manage",
        description="Engram management CLI — API keys, projects, and stats",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # create-key
    p = sub.add_parser("create-key", help="Create a new API key")
    p.add_argument("--name", required=True, help="Unique name for this key")
    p.add_argument("--role", choices=["agent", "admin"], default="agent", help="Key role")
    p.add_argument("--projects", default="", help="Comma-separated project names")
    p.add_argument("--expires-in-days", type=int, default=None, help="Key expiration in days")
    p.set_defaults(func=cmd_create_key)

    # list-keys
    p = sub.add_parser("list-keys", help="List all API keys")
    p.set_defaults(func=cmd_list_keys)

    # revoke-key
    p = sub.add_parser("revoke-key", help="Revoke an API key")
    p.add_argument("--name", required=True, help="Key name to revoke")
    p.set_defaults(func=cmd_revoke_key)

    # list-projects
    p = sub.add_parser("list-projects", help="List all projects")
    p.set_defaults(func=cmd_list_projects)

    # create-project
    p = sub.add_parser("create-project", help="Create a new project")
    p.add_argument("--name", required=True, help="Project name")
    p.add_argument("--description", default="", help="Project description")
    p.add_argument("--path", default="", help="Filesystem path")
    p.set_defaults(func=cmd_create_project)

    # stats
    p = sub.add_parser("stats", help="Show overall statistics")
    p.set_defaults(func=cmd_stats)

    # re-embed
    p = sub.add_parser(
        "re-embed",
        help="Re-embed all memories with the current embedding model",
    )
    p.add_argument(
        "--batch-size", type=int, default=100,
        help="Batch size for embedding (default: 100)",
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt for dimension changes",
    )
    p.set_defaults(func=cmd_re_embed)

    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
