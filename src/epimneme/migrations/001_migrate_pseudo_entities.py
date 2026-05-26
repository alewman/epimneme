"""Migrate memory:xxxx pseudo-entities to memory_entities join table.

This migration finds all entities whose name starts with 'memory:' and
moves their relationships into the new memory_entities join table, then
removes the pseudo-entity nodes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def up(conn) -> None:
    """Migrate pseudo-entity relationships to memory_entities join table."""

    cur = await conn.execute("""
        SELECT e.id, e.name, e.project_id, e.properties
        FROM entities e
        WHERE e.name LIKE 'memory:%%'
    """)
    rows = await cur.fetchall()

    migrated = 0
    for row in rows:
        pseudo_id = row["id"]

        # The memory ID was stored as the entity's own ID
        memory_id = pseudo_id

        # Check if the memory still exists
        cur2 = await conn.execute(
            "SELECT id FROM memories WHERE id = %s", (memory_id,)
        )
        mem = await cur2.fetchone()
        if not mem:
            await conn.execute("DELETE FROM entities WHERE id = %s", (pseudo_id,))
            continue

        # Find all "about" relationships from this pseudo-entity
        cur3 = await conn.execute("""
            SELECT r.to_entity, e.id AS target_entity_id
            FROM relationships r
            JOIN entities e ON e.id = r.to_entity
            WHERE r.from_entity = %s AND r.relation = 'about'
        """, (pseudo_id,))
        rels = await cur3.fetchall()

        for rel in rels:
            target_id = rel["target_entity_id"]
            await conn.execute("""
                INSERT INTO memory_entities (memory_id, entity_id, relation)
                VALUES (%s, %s, 'about')
                ON CONFLICT (memory_id, entity_id) DO NOTHING
            """, (memory_id, target_id))
            migrated += 1

        await conn.execute(
            "DELETE FROM relationships WHERE from_entity = %s OR to_entity = %s",
            (pseudo_id, pseudo_id),
        )
        await conn.execute("DELETE FROM entities WHERE id = %s", (pseudo_id,))

    logger.info(f"Migrated {migrated} pseudo-entity links to memory_entities join table")
