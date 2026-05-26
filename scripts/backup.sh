#!/usr/bin/env bash
# Engram database backup script
# Usage: ./scripts/backup.sh [output_dir]
#
# Backs up the PostgreSQL database using pg_dump inside the Docker container.
# Keeps the last 7 daily backups by default.
#
# Environment variables:
#   EPIMNEME_DB_CONTAINER  — Docker container name (default: epimneme-db-1)
#   EPIMNEME_DB_NAME       — Database name (default: engram)
#   EPIMNEME_DB_USER       — Database user (default: engram)
#   BACKUP_RETAIN_DAYS   — Days to retain backups (default: 7)

set -euo pipefail

CONTAINER="${EPIMNEME_DB_CONTAINER:-epimneme-db}"
DB_NAME="${EPIMNEME_DB_NAME:-engram}"
DB_USER="${EPIMNEME_DB_USER:-engram}"
RETAIN_DAYS="${BACKUP_RETAIN_DAYS:-7}"
OUTPUT_DIR="${1:-$(dirname "$0")/../backups}"

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$OUTPUT_DIR/engram_${TIMESTAMP}.sql.gz"

echo "Backing up engram database..."
echo "  Container: $CONTAINER"
echo "  Database:  $DB_NAME"
echo "  Output:    $BACKUP_FILE"

docker exec "$CONTAINER" pg_dump -U "$DB_USER" "$DB_NAME" \
    --no-owner --no-privileges --clean --if-exists \
    | gzip > "$BACKUP_FILE"

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup complete: $BACKUP_FILE ($SIZE)"

# Prune old backups
echo "Pruning backups older than $RETAIN_DAYS days..."
PRUNED=$(find "$OUTPUT_DIR" -name "engram_*.sql.gz" -mtime +"$RETAIN_DAYS" -delete -print | wc -l)
echo "Pruned $PRUNED old backup(s)"

# Summary
TOTAL=$(find "$OUTPUT_DIR" -name "engram_*.sql.gz" | wc -l)
echo "Total backups: $TOTAL"
