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

1. Which Ollama endpoint + reader model should e2e use? **Answered 2026-09-07: `http://10.10.20.167:11434` (plain HTTP, port 11434), model `qwen3.8:27b` — verified responding with `num_ctx=16384`. `gemma4:31b` is gone; pass `--ollama-url` and `--model`. Phase 3 is unblocked.**
2. Is Phase 6 (preference extractor) wanted, or skip? **Answered: skip.**
3. Should the final docs sync (Phase 7) also bump the version to 0.8.0 for a PyPI release? **Answered: no, stayed at 0.7.0.**

---

## 7. Session Handoff — 2026-09-06 (start here for a clean session)

Phases 0/1/2/4/5/7 landed (see CHANGELOG.md and git log — `df2ef32`..`aa8dce8`). Phase 3 is
still blocked on an Ollama endpoint. Phase 6 skipped. This section is about what came
*after* that: a deep-dive into why LME-S turn-level evidence completeness sits around
70-90% instead of near-100%, done in response to production actually running this code.
Read this before touching retrieval again — several false leads were already run down.

### What's proven

1. **The `recall_all@k` "9.4%" number from the first pass of this investigation was
   overwhelmingly a metric-definition artifact**, not a retrieval problem — it counted
   every turn in an answer session as required evidence, when usually only one turn
   contains the answer. Fixed by scoring against the dataset's real per-turn
   `has_answer` ground-truth flag (`benchmarks/diagnose_evidence_gap3.py`) instead of a
   substring-match heuristic (which had a 45% fallback-contamination rate — see below).
   **True turn-level evidence_completeness@10 is 68-98% across question types.**
   `multi-session` (~68-75%, exact number moved after a real bug fix, see below) and
   `single-session-preference` (~68%, noisy, n=30, don't trust this one as strongly)
   are the two soft spots. Confirmed this metric actually matters: joined it against
   real e2e answer correctness (`results_engram_lme_e2e_v402_...jsonl`, matched against
   its own retrieval snapshot, not a different one) — completeness=1.0 questions hit
   70.9%, partial completeness 17.9%, zero completeness 3.7%. It's a real target.

2. **Two structural fixes ruled out, both confirmed on real (`has_answer`) gold, not
   the contaminated substring heuristic:**
   - Session-diversity reranking (hard cap or continuous penalty on how many slots one
     session can take in top-10) — tested both ways, **monotonically worse at every
     setting, for every question type, including multi-session itself.** Do not
     revisit this without new evidence; it was checked twice.
   - Embedding-truncation position (does the has_answer content sit past the
     ~1300-char / 256-token MiniLM window?) — **not the mechanism**: found and missed
     turns have statistically identical answer-position offsets (median 39 chars into
     the turn, both groups). The truncation issue is real (see below) but isn't what's
     causing *these specific* misses.

3. **A genuine RRF fusion pathology exists, with one clean proof case**
   (`benchmarks/results_channel_diag_20260906.jsonl`, built with a new opt-in
   `debug=` param on `manager.recall()` / `GET .../search?debug=true` — see
   `aa8dce8`). Question `gpt4_15e38248`: the item that won the top-10 slot ranked
   127th/128th/127th on semantic/entity/turn_pair but 2nd on BM25; the actual answer
   turn ranked 8th on those same three signals but 103rd on BM25. **9 of 25 sampled
   misses (multi-session + temporal-reasoning) show this same shape**: semantic +
   entity (an independent *lexical* signal — substring-matches proper
   nouns/numbers, not embedding-derived, see correction below) agree the gold turn
   is good; BM25 alone disagrees, sometimes by 100+ ranks, and drags the fused
   result down enough to lose.

   **What this is NOT**: not "3 independent signals vs. 1" — `turn_pair_rank` is
   confirmed to be a passenger, not a vote (it's a stable sort on a
   near-universally-true boolean, so it just inherits the semantic+fulltext merge
   order). It's closer to "semantic + entity (lexical) vs. BM25 (different
   lexical)," a 2-vs-1 with the two agreeing signals arrived at independently by
   different mechanisms (embedding similarity vs. exact substring counting).

   **What's NOT established**: prevalence (9/25 carries roughly a ±19pp interval —
   hypothesis-generating, not a real rate) and payoff size. **The proposed "simulate
   a fix against this data" idea is invalid as designed** — the sample is
   selected-for-being-a-miss, so any rule fit to it succeeds trivially, and there
   are zero cases in it where BM25 was *correctly* decisive, so there's no way to
   even detect a regression. A real test needs fresh samples of both misses and
   hits, specifically including hits where BM25 was the deciding signal.

   **Production-risk consideration, not yet resolved**: down-weighting BM25 would
   likely help LongMemEval (paraphrased natural language, lexical overlap is often
   noise) but is a plausible net-negative for real coding-agent traffic, where
   queries for identifiers/filenames/error strings are exactly the case where BM25
   being the lone dissenter means BM25 is right. The benchmark cannot see this cost
   — it doesn't contain that query distribution. Any fix needs production-realistic
   validation, not just an LME-S number.

   **Architectural consideration**: RRF's insensitivity to score *magnitude* (only
   rank matters) is what took LoCoMo from 61.5%→91.4% when it replaced the old
   linear merge (see BENCHMARK_RESULTS.md, April 2026). A "consensus override" that
   lets agreement between channels override a single channel's rank reintroduces
   exactly the magnitude-sensitivity RRF was adopted to escape. Not disqualifying,
   but it means this change is closer to the architectural core than it looks —
   treat it with the same care as touching the RRF weights themselves (ground rule
   6 territory, even though it's technically a new mechanism, not a weight change).

4. **The majority bucket is bigger and harder than the BM25 finding**: 14 of 25
   sampled misses are "uniformly weak" — no channel ranks the gold turn well,
   including one case absent from *every* channel's candidate list entirely. This
   points at `all-MiniLM-L6-v2` itself (a small, dated embedding model chosen for
   MemPalace comparability, not quality) as the real ceiling. No fusion-stage change
   touches this bucket. This is the bigger project, not a quick win — matches the
   embedding-profile-swap discussion from way back (`EPIMNEME_ARCHITECTURE_REVIEW.md`),
   including the standing caution that a model swap invalidates the LoCoMo apples-to-
   apples comparison unless the old encoder is pinned as a named "comparison" config.

### Bugs fixed during this investigation (already committed, `aa8dce8` and earlier)

- `_abs` (abstention-variant) questions are separate dataset entries with their own
  haystacks; an early pass in this investigation looked up the base question first,
  silently scoring 8 questions against the wrong haystack. Fixed (exact-ID lookup
  first). This moved multi-session's real number by several points — if you rerun
  any of the `diagnose_evidence_gap*.py` scripts, use gap3, not gap2 or gap1, they
  have this bug.
- Null-model math for "is the correct session enriched above chance in top-10 when
  its specific turn is missing" needs the *precise* per-question hypergeometric
  (actual per-session item counts in that question's own pool), not a coarse
  `10/n_distinct_sessions` approximation — the coarse version overstated enrichment
  1.34x vs the correct 1.09x (barely above chance).

### Concrete next steps, in priority order

1. **If continuing the BM25 thread**: design a properly-controlled sample first —
   misses AND hits, across question types, specifically including hits where BM25
   was the deciding signal — before proposing any weight or fusion-rule change.
   Validate any candidate fix against a held-out sample, not the discovery sample.
   Consider testing it against a synthetic identifier/filename-style query set (or
   real production query logs, if available and consented) before trusting an
   LME-S-only result, given the production-risk concern above.
2. **The embedder ceiling (56% of misses) is probably the higher-value target long
   term**, but it's a bigger lift (touches `embedding_model`/`embedding_dim` config,
   needs the pinned-comparison-config safeguard, needs re-validation across LME-S/
   LoCoMo/BEAM). Don't start this casually.
3. **Ask the user to identify the "magic number."** Aubrey's own recollection this
   session: *"I think we did all of these different ways to just increase our
   score... I feel we stuck a magic number in here too that seemed to increase our
   retrieval quality, but if we changed anything in the pipeline, the magic number
   would need to be redone or it would likely tank everything."* Likely candidates
   given the documented tuning history in `BENCHMARK_RESULTS.md` (the v0.9/v0.91/
   v1.00 temporal-boost tuning that turned out to be mostly noise; the RRF weight
   sweep table; `tiebreak_eps=0.005`): `EPIMNEME_RRF_KEYWORD_WEIGHT` (0.75),
   `bm25_signal_weight` (0.5, newly relevant given tonight's finding),
   `temporal_hard_filter_sigma`/boost `sigma`/`boost_cap`, or `tiebreak_eps`. Don't
   guess which one — ask, or grep the tuning history in BENCHMARK_RESULTS.md's
   "Weight Tuning Exploration" and "Near-tie Gap Analysis" sections for the exact
   value and re-derive why it was chosen before changing anything nearby.
4. Phase 3 (e2e reader) still needs a live Ollama endpoint from the user.

### Addendum — 2026-09-07 session (chunking investigation; pick up here)

**Decisions from Aubrey this session:** keep the current embedder (no goal-post moves);
pursue "the chunking fix". Committed tonight: the reader-side fix (see CHANGELOG
"Fixed": harness `[:2000]` text cap → full text; e2e `num_ctx` → 16384). The
embedder-side question is *not* decided — the decisive experiment is written but
could not finish (below).

**What the "chunking" problem actually is** (root cause 3 in §1 was mis-described):
- Ingest stores whole turn-pairs; there is no 1000-char chunker on the LME path.
- Reader side: the harness clipped stored text at 2,000 chars (65.7% of v700 top-10
  items) and the e2e run never set `num_ctx`. Fixed. Payoff needs an e2e rerun
  (Phase 3 — now unblocked, see §6 Q1). Expect the biggest movement in
  single-session-assistant, whose gold turns are the ones with answers in the tail.
- Embedder side: MiniLM window is 256 tokens (`SentenceTransformer.max_seq_length`);
  LME gold turn-pairs are median ≈630 tokens, 94.8% > 256; all docs 79% > 256
  (`benchmarks/diagnose_chunk_length.py`, corpus stats in its docstring). Only the
  semantic channel is affected — FTS/BM25/entity see full content. Gold recall@10
  drops with length (97.6% → 88.3% → 77.9% across ≤256 / 257–512 / >512 tokens;
  multi-session >512: 62.1%), but length is confounded with question type, so that
  is suggestive, not causal. The answer *start* is inside the window in 53/54
  assistant-answer cases, consistent with the earlier "position isn't the
  mechanism" finding — the live hypothesis is **dilution**: a short answer-bearing
  user statement packed with a long generic assistant reply blurs the vector.
- Production data is short (364 live projects, 3,385 memories, median 441 chars,
  2% > 1,300 chars), so an embedder-side fix protects imported transcripts and
  future long memories more than today's data. Bulk-import chat chunks use
  `User:` / `Assistant:` prefixes (no brackets) — a role-aware chunker must accept
  both that and the `[USER]:`/`[ASSISTANT]:` transcript convention.

**RESOLVED 2026-09-07 — NO-GO.** The simulation below ran to completion once the CPU
freed up (200 stratified questions, ~68 min/pass on ~10 cores). Result: best variant
(`role_win`) lifts semantic-channel gold-turn recall@10 76.3%→79.1% and session R@1
84.5%→88.5%; multi-session flat (67.4%→68.9%); `win` alone slightly negative on turn
recall; passes disagree on sign for temporal/preference. Inside noise, wrong bucket —
do not productionize. Full table and reading in `benchmarks/BENCHMARK_RESULTS.md`.
**Next: Phase 3** (e2e rerun on the v700 retrieval results with the fixed harness;
endpoint in §6 Q1) — that is where the remaining loss (−38pp reader gap) lives.

**The decisive experiment (as designed; kept for the record):**
`benchmarks/sem_chunking_sim.py` — semantic-channel-only ranking of each question's
own haystack under four indexing strategies (`head` = today, `win` = 254-token
windows max-pooled, `role` = {pair, user turn, assistant turn} max-pooled,
`role_win`), whole population, per question type, reporting turn-level found@10/50
and session-level R@1/R@10. Run it stratified (`every=5`, offsets 0 and 2) once the
CPU is free. **Go/no-go rule:** productionize (extra chunk vectors in a child table,
max-pooled into `search_semantic`, memory rows untouched, config-gated) only if a
variant lifts gold turn found@10 materially with session R@1 not lower; then
validate on a real LME-S run plus LoCoMo + BEAM per ground rule 1. If flat, drop the
embedder-side idea and keep only the harness fix.

**Why it didn't finish tonight:** the host was saturated by 30 `pypy3` z80
conformance jobs from `/data/emu/z80-python` (load avg 60–100 on 32 cores). The
in-container embedder fell to ~4 long-encodes/s and *got slower with more threads*;
two 4-hour passes produced nothing. Aubrey expects those jobs done ~03:00–04:00
2026-09-07. Check `uptime` before launching anything embedding-heavy.

**Environment gotchas found tonight:**
- Host Python was upgraded 3.10 → 3.14; `aiohttp`/`pytest` were stranded in
  `~/.local/lib/python3.10/site-packages`. Neither the host nor the app container
  can run the harness or pytest as-is. A gitignored `.venv` (3.14, torch-cpu,
  `-e .[dev]`, aiohttp, pytest-mock) was created at repo root — use
  `.venv/bin/python` / `.venv/bin/pytest`. `run_bench.sh` calls bare `python3`;
  activate the venv (or prefix `PATH=$PWD/.venv/bin:$PATH`) before using it.
- `docker run` from `engram:latest` for ad-hoc scripts needs `--no-healthcheck`:
  the image healthcheck probes :8000 and the host's `autoheal` container restarts
  anything that fails it.
- `all-MiniLM-L6-v2`'s `tokenizer.json` has a baked-in 128-token truncation; raw
  `tokenizers` counts are capped unless `tok.no_truncation()` is called. The first
  pass of tonight's length analysis was invalidated by this.
- The local docker container named `ollama` is an empty shell (no binary, no port).
  The real endpoint is on 10.10.20.167 (see §6 Q1).

### Addendum — 2026-09-09 (Phase 3 measured; assembly run in flight)

**Phase 3, part 1 (reader-side fix) is measured** — full table and miss analysis in
`benchmarks/BENCHMARK_RESULTS.md` § "Phase 3 — e2e reader-side fix". Like-for-like
(same v700 retrieval, same `qwen3.8:27b` reader, judged): control (clipped text,
`num_ctx=4096`) 0.340 → fixed (full text, `num_ctx=16384`) **0.718** overall, +37.8pp;
every category up; gates pass except knowledge-update (0.821 vs ≥ 0.93). Remaining
misses are reader-side with the gold sessions already in the pool: stale-vs-current
confusion (knowledge-update) and date arithmetic the reader refuses (temporal,
33/60 misses answer `Unknown`) — i.e. the assembly module's targets.

**Phase 3, part 2 (assembly in the harness) is implemented and running.**
`lme_e2e_bench.py` now defaults to `assemble_context` over the *same* candidate pool
(`--no-assembly` = old path; `--assembly-budget`). First run:
`results_engram_lme_e2e_v700-assembly-b48k-ctx16384_20260909.jsonl`, budget 48,000
chars (the raw multi-session median; the module default of 12,000 keeps only 3–5 of
10–20 excerpts and would confound the comparison with the full-text result), no
parent expansion (no neighbour fetcher offline). Started 2026-09-09 11:04 PDT,
~5–7 h. Then: `--rescore-only --judge` on it, `compare_e2e.py` against *fixed*,
write the verdict here. If knowledge-update still misses its gate, next knobs are
`prune_superseded` coverage (SimHash near-dup + entity divergence) and the
counting-query K, not retrieval.

**Reader endpoint facts (2026-09-09):** the Mac must be on MagSafe (or a 100 W USB-C
cable) for a run — on battery/5 W it throttles 6–10× and the reader starts answering
`Unknown`. `gemma4:31b` is gone; always pass `--ollama-url http://10.10.20.167:11434
--model qwen3.8:27b`. Rescore mode is `--rescore-only --judge --out <file>` (no
retrieval file needed now). Result files are gitignored; the local set now includes
`*_v700-clipped-ctx4096_20260907.{,rescored.}jsonl` and
`*_v700-fulltext-ctx16384_20260908.{,rescored.}jsonl`.

**Unrelated uncommitted work sitting in the tree (from the 2026-09-08 afternoon):**
`Dockerfile` workers 4→1, `ratelimit.py` rewritten as pure ASGI middleware (SSE
`http.response.start` crash), advisory lock in `_init_schema`. Not part of Phase 3;
review and commit separately.

### Addendum — 2026-09-10 (assembly measured; ablations running)

Assembly in the reader loop is **flat overall** (0.718 → 0.718 / 0.714) with
reproducible per-category moves: temporal +6.0pp, preference +13.3pp,
multi-session −10 to −12pp, knowledge-update 0. Pool-K/56k run rules out K and
budget as the cause of the multi-session loss. Full table and reading in
`benchmarks/BENCHMARK_RESULTS.md` § "Phase 3 — assembly module in the reader loop".

Harness now has `--assembly-k {adaptive,pool}`, `--assembly-parents`,
`--assembly-skip prune,group,chrono,dates`, `--types`; `assemble_context(skip=...)`
in `src/epimneme/assembly.py` (39 tests pass; skipping all steps == raw join).

Running (chained, judged automatically): multi-session-only ablations skipping
`group`, `chrono`, `dates` — result files `results_engram_lme_e2e_v700-ms-ablate-
{group,chrono,dates}-poolk-b56k_20260910.jsonl`, logs `/tmp/bench_e2e_ablation*.log`.
Compare each against the pool-K run's 133 multi-session rows with `compare_e2e.py`.

Findings that change the plan: supersession pruning changes 0/500 contexts (removal
candidate); parent expansion eligible on 28/500 and rescues ≤3 (shelved);
knowledge-update misses are turn-level retrieval gaps, not stale-value confusion —
Phase 3's "≥93% knowledge-update" gate is not reachable from presentation.

