# Engram — Development Makefile
# Usage: make test | make lint | make build | make deploy
#
# Paths are relative to /docker/appdata/engram (where this Makefile lives).
# Docker commands use --env-file for the shared compose .env.

SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE_DIR  := /docker/compose
COMPOSE_FILE := $(COMPOSE_DIR)/dev/engram.yml
COMPOSE_ENV  := $(COMPOSE_DIR)/.env
COMPOSE      := docker compose --env-file $(COMPOSE_ENV) -f $(COMPOSE_FILE)
DOCKER_EXEC  := docker exec engram

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Testing ──────────────────────────────────────────────────────────────────

.PHONY: test test-local test-docker test-verbose test-cov

test: test-local ## Run tests (alias for test-local)

test-local: ## Run pytest locally (requires dev dependencies)
	python -m pytest tests/ -x -q

test-verbose: ## Run pytest with verbose output
	python -m pytest tests/ -x -v

test-cov: ## Run pytest with coverage report
	python -m pytest tests/ --cov=engram --cov-report=term-missing --cov-report=html -x

test-docker: ## Run tests inside the running container
	$(DOCKER_EXEC) python -m pytest tests/ -x -q

# ── Linting ──────────────────────────────────────────────────────────────────

.PHONY: lint lint-fix typecheck

lint: ## Run ruff linter
	python -m ruff check src/ tests/

lint-fix: ## Auto-fix lint issues
	python -m ruff check --fix src/ tests/

typecheck: ## Run mypy type checking (if installed)
	python -m mypy src/engram/ --ignore-missing-imports

# ── Build & Deploy ───────────────────────────────────────────────────────────

.PHONY: build up down restart logs

build: ## Build the Docker image
	$(COMPOSE) build engram

up: ## Start engram + db containers
	$(COMPOSE) up -d

down: ## Stop engram + db containers
	$(COMPOSE) down

restart: build ## Rebuild and restart
	$(COMPOSE) up -d engram

logs: ## Tail container logs
	$(COMPOSE) logs -f engram

# ── Database ─────────────────────────────────────────────────────────────────

.PHONY: db-logs db-shell db-backup db-stats

db-logs: ## Tail database logs
	$(COMPOSE) logs -f epimneme-db

db-shell: ## Open psql shell in the database
	docker exec -it epimneme-db psql -U engram

db-backup: ## Trigger a JSON backup via the API
	$(DOCKER_EXEC) python -c "import asyncio; from engram.backup import save_backup; from engram.stores.postgresql import PostgresStore; print('Use /api/backups/create endpoint instead')"

db-stats: ## Show engram stats
	$(DOCKER_EXEC) python -m engram.manage stats

# ── Management ───────────────────────────────────────────────────────────────

.PHONY: list-keys create-admin-key

list-keys: ## List API keys
	$(DOCKER_EXEC) python -m engram.manage list-keys

create-admin-key: ## Create a new admin API key
	$(DOCKER_EXEC) python -m engram.manage create-key --name admin --role admin

# ── All-in-one ───────────────────────────────────────────────────────────────

.PHONY: check ci

check: lint test ## Run lint + tests

ci: lint test-cov ## CI-style check: lint + tests with coverage
