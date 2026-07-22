# Implementation Plan — Reader Pipeline & Retrieval Improvements (July 2026)

**Executor:** coding agent (Sonnet 5).
**Repo:** `/data/emu/epimneme` — persistent memory service for AI agents (FastAPI + MCP, PostgreSQL + pgvector).
**Origin:** architectural review (Fable, 2026-07-22). Retrieval on LongMemEval-S is nearly saturated (R@10 ≈ 98.5–100% in every category except preference); the end-to-end reader pipeline loses ~38pp. This plan shifts effort to **context assembly + reader synthesis**, plus two targeted retrieval items.

---

## 1. Verified Baselines (do not trust prose docs — they are stale)

### Retrieval, LME-S, v500 (best current — `benchmarks/results_engram_lme_s_turnpair_v500_20260516.jsonl`)

| Category | n | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| knowledge-update | 78 | 0.962 | 1.000 | 1.000 |
| multi-session | 133 | 0.842 | 0.985 | 0.985 |
| single-session-assistant | 56 | 1.000 | 1.000 | 1.000 |
| single-session-preference | 30 | 0.400 | 0.800 | 0.833 |
| single-session-user | 70 | 0.900 | 0.986 | 0.986 |
| temporal-reasoning | 133 | 0.835 | 0.947 | 0.985 |
| **Overall** | 500 | **0.858** | **0.968** | **0.980** |

### Retrieval, LME-M, v402 (`results_engram_lme-m_v402-mmr-fix_20260515.jsonl`)

Overall R@1 = 0.672, R@10 = 0.902. Multi-session R@1 = 0.571. **This is where retrieval headroom lives.**

### End-to-end reader (Gemma via Ollama, top-10 rank-ordered chunks, v402 retrieval — `results_engram_lme_e2e_v402_20260512_corrected.rescored.jsonl`, score with the `hit` field)

| Category | Retrieval ceiling R@10 | E2E acc | Reader loss |
|---|---|---|---|
| single-session-assistant | 1.000 | 0.446 | −55pp |
| temporal-reasoning | 0.985 | 0.489 | −50pp |
| multi-session | 0.985 | 0.526 | −46pp |
| knowledge-update | 1.000 | 0.897 | −10pp |
| single-session-user | 0.986 | 0.929 | −6pp |
| single-session-preference | 0.833 | 0.100 | −73pp |
| **Overall** | ~0.98 | **0.596** | **−38pp** |

### Diagnosed root causes

1. **Temporal**: reader is asked to do date arithmetic from `[Date: …]` headers — small LLMs can't. Must be pre-computed in code.
2. **Knowledge-update**: chunks are presented in *rank* order; old and new values of a fact appear with no supersession cue; reader sometimes picks the stale value.
3. **Assistant**: retrieval is perfect but 1000-char chunking truncates long assistant turns; the asked-about detail sits in an adjacent, unshown chunk.
4. **Multi-session (counting)**: fixed K starves "how many" questions; `recall_any@k` masks incomplete evidence (only checks that *one* gold session is present — counting needs *all*).
5. **Preference**: genuine retrieval weakness (R@1 0.40) + reader mismatch. Smallest prize (n=30). Do last.

---

## 2. Ground Rules (non-negotiable)

1. **No regression on generalization benchmarks.** Any retrieval-path change must be validated on LoCoMo + BEAM (skip-ingest) and, when plausibly affected, BEIR SciFact. Budget: no benchmark drops by more than 1pp without explicit user sign-off.
2. **No benchmark-specific code in `src/epimneme/`.** All gating must be general-purpose query classification (existing style: `is_counting_query`, `is_vague_query`, `parse_target_date` in `src/epimneme/fusion.py`). The project's public claim is "zero benchmark tuning" — protect it.
3. **Iterative verify-then-commit.** One phase at a time: implement → unit tests pass (`pytest`) → benchmark gate passes → commit. Never batch multiple phases into one unverified commit.
4. **Every benchmark run goes through `benchmarks/run_bench.sh`** so it is appended to `benchmarks/results_history.tsv` (this ledger silently went stale after May 12 — resume it).
5. **New config knobs** follow the existing pattern: field in `EngramConfig` (`src/epimneme/core/config.py`) + `EPIMNEME_*` env override in `from_env` + row in `AI_TOOLBOX.md` config table.
6. **Do not modify** the core RRF fusion weights or existing signal defaults — those are the validated v402/v500 state.

---

## 3. Environment Setup & Gotchas

- Stack: `docker compose up -d --build` at repo root. App container is named `engram` (image `engram:latest` — rename leftover, fine); DB container `epimneme-db` (pgvector/pg16). Port 8000.
- Env prefix is **`EPIMNEME_`** (consistent in `core/config.py`). Server refuses to start with unset/default PG password unless `EPIMNEME_DEMO_MODE=1`. For local bench work: copy `.env.example` → `.env`, set a password, or use demo mode.
- Smoke test: `curl -s http://localhost:8000/health`, then create key/project via `docker exec engram python -m epimneme.manage create-key --name admin --role admin` and `create-project`.
- Unit tests: `pytest` (from repo root; `pythonpath=src` configured). Integration tests marked `integration` need live PG.
- **`AI_TOOLBOX.md` contains stale paths** (`/docker/appdata/engram`, `http://192.168.90.45:8000`) from the pre-rename private deployment. Benchmarks' Python defaults now point to `http://localhost:8000`. Pass `--engram-url` explicitly where needed. `run_bench.sh` may still hardcode the old LAN URL — **check and fix it first** (Phase 0.1).
- Benchmark data is already present in `benchmarks/data/` (`longmemeval_s_cleaned.json`, `longmemeval_m/`, `locomo10.json`, …). LME ingest can be pre-staged once, then `--skip-ingest` for fast iteration (see `AI_TOOLBOX.md` §Pre-Staging).
- E2E reader endpoint: `lme_e2e_bench.py` defaults to `OLLAMA_URL = http://10.10.20.167:11434`, model `gemma4:31b`. **Verify this endpoint is reachable; if not, ASK THE USER** which Ollama host/model to use for the reader (`--ollama-url`, `--model` flags exist). Prefer a current-generation small instruct model over gemma4 if the user has one pulled.
- Result-file schemas:
  - Retrieval JSONL: `retrieval_results.metrics.session.recall_any@{k}` per question; `retrieval_results.ranked_items` = `[{corpus_id, text, score}]`; chunk text convention: `[Date: YYYY/MM/DD (Day) HH:MM]\n[USER]: …\n[ASSISTANT]: …`; `corpus_id` convention: `{answer|noans}_{qid}_turn_{n}` (turn index = position in session).
  - E2E JSONL: score with `hit` (bool), `abs_hit` for unanswerable `_abs` questions.
- `MemoryResult` = `{memory: Memory, score, source}`; `Memory` has `session_id`, `session_ordinal`, `created_at`, `supersedes`, `version_of` (`src/epimneme/core/models.py`).

---

## 4. Phases

Execute in order. Each phase ends with a commit. Suggested commit prefixes shown.

### Phase 0 — Baseline & honest measurement  *(commit: `bench:`)*

**0.1** Fix `benchmarks/run_bench.sh` if it still hardcodes the old LAN URL; point it at `http://localhost:8000` (keep an `--engram-url` passthrough). Bring the stack up; run `pytest`; confirm green before touching anything.

**0.2** Add **`recall_all@k`** and **`evidence_completeness@k`** to the LME harness:
- `benchmarks/metrics.py` + `benchmarks/longmemeval_bench.py`.
- `recall_all@k` = 1.0 iff *every* gold answer session appears in top-k (LME questions carry multiple `answer_session_ids` for multi-session questions).
- `evidence_completeness@k` = |gold sessions in top-k| / |gold sessions|.
- Emit alongside existing `recall_any@k` — do not change existing keys (old result files must stay comparable).
- Unit-test with `benchmarks/data/test_lme_fixture.json`.

**0.3** Re-establish retrieval baseline on current code: full LME-S run via `run_bench.sh lme v600-baseline`. Pre-stage data first if not staged (see AI_TOOLBOX). Expect ≈ v500 numbers (±1pp run variance is known). Record the new `recall_all@10` — this is the honest multi-session evidence ceiling.

**0.4** Re-run **e2e baseline** against the 0.3 retrieval results with the confirmed reader model: `python3 benchmarks/lme_e2e_bench.py --retrieval-results <0.3 output> --judge --out benchmarks/results_epimneme_e2e_v600-baseline_<date>.jsonl`. This is the comparison line for Phases 1–3. Score per-category with the `hit` field.

### Phase 1 — Context assembly module  *(commit: `assembly:`)*

**New file `src/epimneme/assembly.py`** — deterministic, $0/query, standalone-testable (operates on a list of `(text, score, metadata)` items so both the server and the bench harness can use it). New file `tests/test_assembly.py`.

Components (each its own function, composed by `assemble_context(...)`):

1. **Date extraction** — parse a leading `[Date: …]` line from content when present (documented convention for imported transcripts), else fall back to `memory.created_at`.
2. **Temporal scaffolding** — reuse `parse_target_date(query, reference_date)` from `fusion.py`. When the question has a reference date: annotate every excerpt header with the pre-computed delta, e.g. `[Date: 2023/05/18 — 4 days before the question]`; when a target date resolves, prepend one line: `The question refers to approximately 2023/05/18.` **All date arithmetic happens here, in code — never delegated to the reader.**
3. **Chronological presentation** — selection stays rank-based; *presentation* order becomes chronological (stable tiebreak on rank). 
4. **Supersession pruning** — first use explicit links (`supersedes`, `version_of`, `obsolete`) when present; second, detect near-duplicate conflicting excerpts (reuse SimHash/semantic-similarity utilities from `dedup.py`) with different dates → keep newest, drop or annotate older `[SUPERSEDED 2023/06/01]`. Dropping saves tokens *and* removes the knowledge-update failure mode.
5. **Adaptive K / token budgeting** — classify query with existing gates: counting/aggregation (`is_counting_query`) → larger K (e.g. 20) grouped per session; resolved-date temporal → standard K; simple single-fact → small K (e.g. 5). Budget by total chars (default ≈ 12,000) rather than count alone. Config knobs: `assembly_budget_chars`, `assembly_k_single`, `assembly_k_counting`.
6. **Session grouping** — merge excerpts from the same session under one date header, ordered by turn index; eliminates repeated headers (token savings).

**Server integration** (backward compatible, default off): `assemble=true` option on the recall/search REST endpoint and the MCP `recall` tool → response gains an `assembled_context` string alongside the normal ranked results. Wire through `server.py` + `manager.py`.

**Acceptance gate:** unit tests cover date parsing, delta annotation, chrono ordering, supersession (linked + detected), budgeting, session grouping. `pytest` green. No change to any existing endpoint default behavior (existing tests untouched).

### Phase 2 — Parent-document (small-to-big) expansion  *(commit: `assembly:`)*

- At assembly time, for **single-session-type contexts** (few distinct sessions in results, non-counting query): fetch sibling chunks adjacent to each hit (same `session_id`, neighboring turn/creation order; in bench data, `corpus_id` `…_turn_{n±1}`) and merge into the excerpt.
- Applied **after** budgeting, capped by `assembly_budget_chars`; config gate `assembly_parent_expansion` (default on for assembled mode only).
- Server path: query the store for same-session neighbors. Bench path: look up neighbors in the staged corpus.
- **Acceptance gate:** unit tests; assembled context never exceeds budget; `pytest` green.

### Phase 3 — E2E harness upgrade & the payoff measurement  *(commit: `bench:`)*

- Modify `benchmarks/lme_e2e_bench.py`: replace the raw `"\n---\n".join(chunks)` with `epimneme.assembly.assemble_context(...)` (import directly; no server round-trip needed for ranked-items mode). Keep a `--no-assembly` flag to reproduce the old path.
- Rerun e2e on the Phase 0.3 retrieval results.
- **Acceptance gate vs Phase 0.4 baseline:**
  - temporal-reasoning e2e ≥ +15pp
  - knowledge-update e2e ≥ 93%
  - single-session-assistant e2e ≥ +15pp
  - no category regresses > 2pp
  - overall ≥ +10pp
- If a gate fails: analyze misses per category (there is prior art in `benchmarks/analyze_pref_misses.py` / `near_miss_analysis_v100.txt` for how misses were analyzed before), iterate on assembly parameters — **do not** touch retrieval to fix reader problems.

### Phase 4 — Temporal partition rerank (retrieval)  *(commit: `fusion:`)*

- In `manager.recall` final ordering (near existing step 11): when `parse_target_date` resolves a **day-precision** target, *partition* the final list — candidates within the window (`sigma` days, reuse `temporal_hard_filter_sigma`) ranked before out-of-window ones, stable order within partitions. This is a structural reorder, not an additive boost — the near-tie analysis (BENCHMARK_RESULTS.md §Near-tie Gap Analysis) proved boosts ≤0.015 cannot close median gaps.
- Config: `temporal_partition_enabled` (default **on**), distinct from the risky hard *filter* (which stays default-off).
- **Acceptance gate:** full LME-S: temporal-reasoning R@1 ≥ 0.87 (from 0.835) and overall R@1 not below baseline − 0.5pp. LoCoMo (`run_bench.sh locomo … --skip-ingest`) and BEAM 100K (`--skip-ingest`) within 1pp of baseline. BEIR spot-check unaffected (no dates in SciFact queries).

### Phase 5 — LME-M hierarchical retrieval (time-boxed investigation)  *(commit: `bench:` or `fusion:`)*

- Question: does two-stage retrieval (stage 1: rank *sessions* by aggregated chunk scores; stage 2: rank chunks within winning sessions) beat flat chunk search at LME-M scale (haystack ≫ LME-S)?
- Prototype as a benchmark-side experiment first (operate on over-fetched candidates; no schema change). Measure LME-M R@1/R@10 vs the 0.672/0.902 baseline.
- **Deliverable:** results table + go/no-go recommendation written into the PR/commit message. Only productionize (config-gated) if LME-M R@1 gains ≥ 3pp with LME-S neutral.

### Phase 6 — Preference extraction at ingest (optional — confirm with user before starting)

- Ingest-time detection of preference statements (general-purpose linguistic patterns, *not* LME-derived regexes) → store a derived `preference`-kind memory linked to the source. `extract_preference_terms` in `fusion.py` is a starting point.
- Target: preference R@1 0.40 → ≥ 0.60, R@10 ≥ 0.90. n=30, so treat all deltas with suspicion; re-run twice.

### Phase 7 — Bookkeeping & docs sync  *(commit: `docs:`)*

1. `CHANGELOG.md` → new Unreleased entries for assembly module, metrics, partition rerank.
2. `BENCHMARK_RESULTS.md` → append a dated July 2026 section with final verified numbers (retrieval + e2e, per category), including `recall_all@k`.
3. `README.md` → refresh the headline benchmark table (it still shows pre-RRF April numbers: R@1 79.0%); mention assembled-context mode.
4. `AI_TOOLBOX.md` → fix stale paths/URLs; add new config knobs to the table; update the version-history table.
5. Confirm `results_history.tsv` has rows for every run made in Phases 0–6.

---

## 5. Success Criteria (whole plan)

| Metric | Baseline | Target |
|---|---|---|
| LME-S e2e overall | ~0.60 (Gemma, v402) | ≥ 0.75 |
| LME-S e2e temporal-reasoning | 0.489 | ≥ 0.70 |
| LME-S e2e knowledge-update | 0.897 | ≥ 0.93 |
| LME-S retrieval temporal R@1 | 0.835 | ≥ 0.87 |
| LME-S retrieval overall R@1 | 0.858 | ≥ 0.858 (no regression) |
| LoCoMo top-10 | 0.914 | ≥ 0.904 (≤1pp drop) |
| BEAM 100K avg_recall | 0.4167 | ≥ 0.407 (≤1pp drop) |
| Context tokens per query | ~10KB fixed | −30% median (adaptive budgeting) |

Keep the story intact: **$0 per query, no LLM in the retrieval loop, no benchmark-specific tuning.** Everything in this plan is deterministic post-retrieval code or gated general-purpose retrieval logic.

## 6. Open Questions for the User (ask before Phase 0.4)

1. Which Ollama endpoint + reader model should e2e use? (Old default `10.10.20.167:11434` / `gemma4:31b` may be gone.)
2. Is Phase 6 (preference extractor) wanted, or skip?
3. Should the final docs sync (Phase 7) also bump the version to 0.8.0 for a PyPI release?
