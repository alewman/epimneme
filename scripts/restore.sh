#!/usr/bin/env bash
# Engram database restore script
# Usage: ./scripts/restore.sh <backup_file>
#
# Restores from a backup created by backup.sh.
# WARNING: This will DROP and recreate all tables!

set -euo pipefail

CONTAINER="${EPIMNEME_DB_CONTAINER:-epimneme-db}"
DB_NAME="${EPIMNEME_DB_NAME:-engram}"
DB_USER="${EPIMNEME_DB_USER:-engram}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    echo ""
    echo "Available backups:"
    SCRIPT_DIR="$(dirname "$0")"
    ls -lh "$SCRIPT_DIR/../backups"/engram_*.sql.gz 2>/dev/null || echo "  (none found)"
    exit 1
fi

BACKUP_FILE="$1"
if [[ ! -f "$BACKUP_FILE" ]]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "WARNING: This will replace all data in the engram database!"
echo "  Container: $CONTAINER"
echo "  Database:  $DB_NAME"
echo "  Backup:    $BACKUP_FILE"
echo ""
read -p "Continue? (y/N) " -r
if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo "Restoring..."
gunzip -c "$BACKUP_FILE" | docker exec -i "$CONTAINER" psql -U "$DB_USER" "$DB_NAME" --quiet

echo "Restore complete. Restart engram to reinitialize: docker compose restart engram"
