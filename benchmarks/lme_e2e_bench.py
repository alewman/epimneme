"""
End-to-end LongMemEval benchmark.

Pipeline:
  1. Load retrieval results from a prior lme retrieval run (ranked_items)
  2. For each question: feed top-K chunks to Gemma via Ollama → generate answer
  3. Score: substring match (primary) + LLM judge pass (optional, for misses)
  4. Report per-type accuracy and save full results

Usage:
  # Full run using pre-existing retrieval results:
  python3 benchmarks/lme_e2e_bench.py \\
      --retrieval-results benchmarks/results_engram_lme_v402-mmr-fix_20260512_1106.jsonl \\
      --out /tmp/lme_e2e_v402.jsonl

  # With LLM judge pass on misses:
  python3 benchmarks/lme_e2e_bench.py \\
      --retrieval-results benchmarks/results_engram_lme_v402-mmr-fix_20260512_1106.jsonl \\
      --judge \\
      --out /tmp/lme_e2e_v402_judged.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import aiohttp


# =============================================================================
# CONFIG
# =============================================================================

OLLAMA_URL = "http://10.10.20.167:11434"
OLLAMA_MODEL = "gemma4:31b"
# Ollama's default context window (4096 in current releases, 2048 in older ones)
# is smaller than 10 full LME turn-pairs plus the prompt, and Ollama truncates the
# *front* of an over-long prompt silently. Set explicitly so the reader actually
# sees every chunk it is given. Overridable with --num-ctx.
OLLAMA_NUM_CTX = 16384
EPIMNEME_URL = "http://192.168.90.45:8000"

ANSWER_PROMPT = """\
You are answering a question about a person's past conversations stored as memory excerpts.
Use ONLY the provided memory excerpts. Answer as concisely as possible — a phrase or a few \
words, not a full sentence. Do not explain your reasoning.
If the answer cannot be found in the excerpts, say exactly: Unknown

Memory excerpts:
{context}

Question: {question}
Answer:"""

ANSWER_PROMPT_TEMPORAL = """\
You are answering a question about a person's past conversations stored as memory excerpts.
Each excerpt begins with a [Date: YYYY/MM/DD HH:MM] header — use these to establish
chronological order. When reasoning about time, compare dates explicitly.
Today's date is: {question_date}
Use ONLY the provided memory excerpts. Answer as concisely as possible — a phrase or a few \
words, not a full sentence. Do not explain your reasoning.
If the answer cannot be found in the excerpts, say exactly: Unknown

Memory excerpts:
{context}

Question: {question}
Answer:"""

# v2 of the temporal prompt. The v1 wording above ends with a blanket
# "if the answer cannot be found in the excerpts, say exactly: Unknown", but a
# temporal answer is DERIVED from the dates and is never literally present, so a
# literal reader is licensed to refuse. Measured on the 2026-09-13 run: 32 of 49
# temporal misses were refusals, 21 of them with the day-delta annotation already
# in the context. v1 also forbids explaining reasoning, which suppresses the
# intermediate steps arithmetic needs, and emits deltas only in days while 44% of
# week-questions and 38% of month-questions refuse against 23% of day-questions.
ANSWER_PROMPT_TEMPORAL_V2 = """\
You are answering a question about a person's past conversations stored as memory excerpts.
Each excerpt begins with a [Date: YYYY/MM/DD HH:MM — N days before the question] header.
Today's date is: {question_date}

The answer is usually NOT stated in the excerpts — you are expected to COMPUTE it from
the dates. Find the relevant event(s), then calculate the interval the question asks for,
converting days into weeks or months when that is the unit requested.

Answer "Unknown" ONLY if the underlying events themselves are absent from the excerpts.
Never answer "Unknown" merely because the number is not written down.

Reply with ONLY the final answer — a number or a short phrase. No working, no
explanation, no units of reasoning. (Generation is capped at 80 tokens, so any
visible reasoning would truncate the answer away.)

Memory excerpts:
{context}

Question: {question}
Answer:"""

# v3 fixes the two defects measured in v2 (+5.3pp judged, p=0.189, 14 fixes /
# 7 breaks). Five of those 7 regressions wanted a NAME and got a number — v2's
# "a number or a short phrase" pushed the reader to answer ordering questions as
# "2, 3, 1". v2 also dropped units ("7" for gold "7 days"), which exact-match
# scored as wrong even though the judge later rescued it; a bare number is worse
# than useless to a downstream consumer regardless of how it scores here.
ANSWER_PROMPT_TEMPORAL_V3 = """\
You are answering a question about a person's past conversations stored as memory excerpts.
Each excerpt begins with a [Date: YYYY/MM/DD HH:MM — N days before the question] header.
Today's date is: {question_date}

The answer is usually NOT stated in the excerpts — compute it from the dates. Find the
relevant event(s), then calculate the interval the question asks for, converting days
into weeks or months when that is the unit requested.

Answer "Unknown" ONLY if the underlying events themselves are absent from the excerpts.
Never answer "Unknown" merely because the number is not written down.

How to format the answer:
- For an elapsed time, give the number WITH its unit — "7 days", "3 weeks", "2 months".
  Never a bare number.
- For a question about order, or which came first or last, name the events themselves
  in the order asked. Never answer with positions, indices or numbers.
- Give only the answer. No preamble, no working, no explanation.

Memory excerpts:
{context}

Question: {question}
Answer:"""

HYDE_PROMPT = """\
You are helping with memory retrieval. Given a question about a person's past conversations,
generate a short hypothetical memory excerpt that would contain the answer.
Write it as if it were a real conversation turn from the user. Be specific — include names,
dates, places, and details that might appear in an actual memory.

Question: {question}
Hypothetical memory excerpt:"""

ANSWER_PROMPT_PREFERENCE = """\
You are answering a question about a person's past conversations stored as memory excerpts.
The question asks what this person would prefer or what suits them best.
Infer their preference from what they have done, chosen, praised, or complained about.
Start your answer with exactly: "The user would prefer responses that"
Then complete the sentence describing the type of content, approach, or suggestions they would want.
Keep it to 1-2 sentences. Do NOT say Unknown unless there is zero relevant information.

Memory excerpts:
{context}

Question: {question}
Answer:"""

ANSWER_PROMPT_ASSISTANT = """\
You are answering a question about a person's past conversations stored as memory excerpts.
Each excerpt may contain both [USER]: lines (what the person said) and [ASSISTANT]: lines \
(what the AI assistant said). The question asks about something the ASSISTANT said, recommended, \
or provided. Look at [ASSISTANT]: lines for the answer.
Use ONLY the provided memory excerpts. Answer as concisely as possible — a phrase or a few \
words, not a full sentence. Do not explain your reasoning.
If the answer cannot be found in the excerpts, say exactly: Unknown

Memory excerpts:
{context}

Question: {question}
Answer:"""

ANSWER_PROMPT_COUNTING = """\
You are answering a question about a person's past conversations stored as memory excerpts.
When the question asks "how many" or "how much total", do the following:
1. Search ALL excerpts for every relevant item or event.
2. List each one briefly (e.g. "- item 1\n- item 2").
3. On the LAST LINE, write ONLY the final answer as a number or total value, with no extra words.
   Example: if you found 3 items, last line = "3".
   Example: if you found a total of $185, last line = "$185".
If the answer cannot be found, say exactly: Unknown

Memory excerpts:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """\
You are evaluating whether a generated answer correctly answers a question, given a gold reference.
Be lenient: accept paraphrases, equivalent number formats (3 = three, 7 days = one week),
minor wording differences, and partial matches where the core fact is correct.
Answer YES if the generated answer is factually equivalent to the gold, NO otherwise.
Reply with exactly one word: YES or NO.

Question: {question}
Gold answer: {gold}
Generated answer: {generated}

Are these equivalent?"""

# Preference-specific judge: semantic equivalence of preference descriptions, not factual match.
# Gold answers describe the *type of response* the user wants; generated may use different framing.
JUDGE_PROMPT_PREFERENCE = """\
You are evaluating two descriptions of what a person would prefer.
The gold answer and the generated answer may differ in detail level or phrasing,
but both should describe the same underlying preference or interest.
Answer YES if both point to the same core preference, NO if they describe different things entirely.
Different levels of detail, extra context, or different phrasing = still YES.
Only answer NO if the preferences are genuinely about different topics.
Reply with exactly one word: YES or NO.

Question: {question}
Gold answer: {gold}
Generated answer: {generated}

Do these describe the same preference?"""

# =============================================================================
# SCORING
# =============================================================================

# Word-to-number map for normalization
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

def normalise(text: str) -> str:
    """Aggressive normalisation for substring matching."""
    text = str(text).lower().strip()
    # Strip framing phrases common in gold answers and in generated output
    text = re.sub(
        r"^(the user would (prefer|like) |i (prefer|would like) |you (prefer|mentioned|said) "
        r"|prefers |prefer |user prefers |the answer is |answer: )",
        "", text,
    )
    # Strip leading articles
    text = re.sub(r"^(a |an |the )", "", text)
    # Normalize word numbers to digits (word-boundary only)
    for word, digit in _NUM_WORDS.items():
        text = re.sub(r"\b" + word + r"\b", digit, text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Strip trailing punctuation
    text = re.sub(r'[.,;:!?]+$', "", text)
    return text


def is_abs_question(question_id: str) -> bool:
    """Return True for unanswerable _abs questions (question_id ends with '_abs')."""
    return str(question_id).endswith("_abs")


def score_substring(gold: str, generated: str) -> bool:
    """True if gold (normalised) appears in generated (normalised), or vice-versa.

    Also checks the last non-empty line of generated, which is where the counting
    prompt asks the model to place its final answer.
    """
    g = normalise(gold)
    gen = normalise(generated)
    if g in gen:
        return True
    # Bidirectional: accept if generated (non-trivial) is a substring of gold
    if len(gen) >= 3 and gen not in ("unknown", "n/a", "none", "yes", "no") and gen in g:
        return True
    # Check last non-empty line (counting prompt puts final answer on last line)
    last_line = normalise(generated.strip().rsplit("\n", 1)[-1])
    if last_line and last_line != gen:
        if g in last_line:
            return True
        if len(last_line) >= 3 and last_line not in ("unknown", "n/a", "none", "yes", "no") and last_line in g:
            return True
    # For short golds (1-4 significant words), accept if all tokens appear
    gold_tokens = [t for t in g.split() if len(t) > 2]
    if 1 <= len(gold_tokens) <= 4:
        if all(tok in gen for tok in gold_tokens):
            return True
    return False
# =============================================================================
# LLM CALL
# =============================================================================


async def _chat(session: aiohttp.ClientSession, prompt: str, max_tokens: int = 80,
                ollama_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL,
                retries: int = 3) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": OLLAMA_NUM_CTX},
    }
    last_exc: Exception = RuntimeError("no attempts")
    for attempt in range(retries):
        try:
            async with session.post(
                f"{ollama_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                d = await resp.json()
            return d.get("message", {}).get("content", "").strip()
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            last_exc = exc
            wait = 10 * (attempt + 1)
            print(f"  [retry {attempt+1}/{retries}] {type(exc).__name__} — waiting {wait}s", flush=True)
            await asyncio.sleep(wait)
    # All retries exhausted — return empty string so the question is scored as a miss, not a crash
    print(f"  [_chat] all retries failed: {last_exc}", flush=True)
    return ""


async def score_with_judge(
    session: aiohttp.ClientSession, question: str, gold: str, generated: str,
    ollama_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL,
    qtype: str = "",
) -> bool:
    # Use preference-specific judge prompt for single-session-preference questions
    template = JUDGE_PROMPT_PREFERENCE if qtype == "single-session-preference" else JUDGE_PROMPT
    prompt = template.format(question=question, gold=gold, generated=generated)
    verdict = await _chat(session, prompt, max_tokens=5, ollama_url=ollama_url, model=model)
    return verdict.strip().upper().startswith("YES")


# =============================================================================
# ENGRAM API (for HyDE)
# =============================================================================

async def _engram_search_hyde(
    session: aiohttp.ClientSession,
    query_text: str,
    project: str,
    top_k: int,
    engram_url: str = EPIMNEME_URL,
) -> list[str]:
    """Search memories via Engram's semantic search API using HyDE text as query.
    
    Returns list of chunk content strings.
    """
    params: dict[str, str | int] = {"query": query_text, "limit": top_k}
    if project:
        params["project"] = project
    try:
        async with session.get(
            f"{engram_url}/api/memories/search",
            params=params,  # type: ignore[arg-type]
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                d = await resp.json()
                return [r["content"] for r in d.get("results", []) if r.get("content")]
    except Exception:
        pass
    return []


async def generate_hyde_chunks(
    session: aiohttp.ClientSession,
    question_text: str,
    project: str,
    top_k: int = 10,
    ollama_url: str = OLLAMA_URL,
    model: str = OLLAMA_MODEL,
    engram_url: str = EPIMNEME_URL,
) -> list[str]:
    """Generate HyDE hypothetical, search Engram with it, return chunk texts."""
    # Step 1: Generate hypothetical memory excerpt via Ollama
    hyde_prompt = HYDE_PROMPT.format(question=question_text)
    hypothetical = await _chat(session, hyde_prompt, max_tokens=80,
                                ollama_url=ollama_url, model=model)
    if not hypothetical:
        return []

    # Step 2: Search Engram using the hypothetical text as query
    return await _engram_search_hyde(session, hypothetical, project, top_k, engram_url=engram_url)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

_CORPUS_ID_RE = re.compile(r"^(?P<session>.+?)_turn_(?P<turn>\d+)$")


def make_neighbor_fetcher(entry: dict):
    """Build a parent-document neighbour fetcher for one LME question.

    Rebuilds the exact turn-pair corpus the retrieval run ingested (same
    `build_corpus` as `longmemeval_bench.py`, so texts and corpus_ids match
    byte-for-byte) and returns the `..._turn_{n-1}` / `..._turn_{n+1}` siblings
    of a hit as Excerpts. `epimneme.assembly.expand_parents` decides *whether*
    to expand (not for counting queries; not when the pool spans > 3 sessions).
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from longmemeval_bench import build_corpus  # noqa: E402
    from epimneme.assembly import Excerpt

    docs, ids, _ = build_corpus(entry, granularity="turn-pair")
    by_id = dict(zip(ids, docs))

    def fetch(ex):
        m = _CORPUS_ID_RE.match(str(ex.metadata.get("memory_id", "")))
        if not m:
            return []
        sess, turn = m.group("session"), int(m.group("turn"))
        out = []
        for t in (turn - 1, turn + 1):
            cid = f"{sess}_turn_{t}"
            if t >= 0 and cid in by_id:
                out.append(Excerpt(text=by_id[cid], score=0.0,
                                   metadata={"memory_id": cid, "session_id": sess, "turn_index": t}))
        return out

    return fetch


def build_context(
    ranked: list[dict],
    question: str,
    qtype: str,
    question_date: str,
    top_k: int,
    use_assembly: bool,
    budget_chars: int | None = None,
    k_mode: str = "adaptive",
    fetch_neighbors=None,
    skip: tuple[str, ...] = (),
    recency_note: bool = False,
    recency_ungated: bool = False,
) -> tuple[str, int, dict]:
    """Turn ranked retrieval items into the reader's context block.

    Raw path (``use_assembly=False``): the historical ``"\\n---\\n".join`` of the
    top-K texts. Assembly path: the same candidate pool handed to
    ``epimneme.assembly.assemble_context`` — adaptive K, supersession pruning,
    char budget, session grouping, chronological order, precomputed date
    deltas. The pool is deliberately identical in both paths so the comparison
    isolates *presentation* from retrieval depth.

    Returns (context, excerpts_used, stats).
    """
    # Per-type top_k: multi-session counting benefits from wider context
    effective_top_k = top_k * 2 if qtype == "multi-session" else top_k
    pool = ranked[:effective_top_k]
    if not pool:
        return "(no memory found)", 0, {}
    if not use_assembly:
        return "\n---\n".join(item["text"] for item in pool), len(pool), {}

    from epimneme.assembly import DEFAULT_BUDGET_CHARS, Excerpt, assemble_context
    from epimneme.fusion import extract_logical_date

    excerpts = []
    for idx, item in enumerate(pool):
        cid = str(item.get("corpus_id", ""))
        m = _CORPUS_ID_RE.match(cid)
        md: dict = {"memory_id": cid}
        if m:
            md["session_id"] = m.group("session")
            md["turn_index"] = int(m.group("turn"))
        else:
            md["turn_index"] = idx
        excerpts.append(Excerpt(text=item["text"], score=float(item.get("score", 0.0)), metadata=md))

    # LME question_date is "YYYY/MM/DD (Dow) HH:MM"; reuse the fusion parser
    # (it understands the slash format) by feeding it as a fake header.
    ref = extract_logical_date(f"[Date: {question_date}]") if question_date else None
    # k_mode="pool": disable adaptive K (k_single/k_default/k_counting all = the
    # harness pool size) so only the presentation transforms differ from the raw
    # path. "adaptive": the module's own K selection (may narrow the pool).
    k_kwargs = {}
    if k_mode == "pool":
        k_kwargs = dict(k_single=len(pool), k_default=len(pool), k_counting=len(pool))
    assembled = assemble_context(
        excerpts, question,
        reference_date=ref,
        budget_chars=budget_chars or DEFAULT_BUDGET_CHARS,
        fetch_neighbors=fetch_neighbors,
        enable_parent_expansion=fetch_neighbors is not None,
        skip=skip,
        recency_note=recency_note,
        recency_note_gate=((lambda q: True) if recency_ungated else None),
        **k_kwargs,
    )
    stats = {
        "assembly_excerpts": assembled.excerpt_count,
        "assembly_chars": assembled.char_count,
        "assembly_truncated": assembled.truncated,
        "pool_chars": sum(len(item["text"]) for item in pool),
    }
    return assembled.text, assembled.excerpt_count, stats


def _pick_prompt(qtype: str, use_temporal: bool, temporal_version: str = "v1") -> str:
    """Select answer prompt based on question type."""
    temporal_tpl = {"v1": ANSWER_PROMPT_TEMPORAL,
                    "v2": ANSWER_PROMPT_TEMPORAL_V2,
                    "v3": ANSWER_PROMPT_TEMPORAL_V3}[temporal_version]
    if qtype == "single-session-preference":
        return ANSWER_PROMPT_PREFERENCE
    if qtype == "single-session-assistant":
        return ANSWER_PROMPT_ASSISTANT
    if qtype == "multi-session":
        return ANSWER_PROMPT_COUNTING
    if qtype == "temporal-reasoning":
        return temporal_tpl  # temporal prompt only for this type
    # For all other types, use temporal only if explicitly forced
    if use_temporal:
        return temporal_tpl
    return ANSWER_PROMPT


async def run(
    questions: list[dict],
    retrieval_map: dict[str, dict],
    top_k: int,
    use_judge: bool,
    use_temporal: bool,
    temporal_version: str,
    counting_max_tokens: int,
    use_hyde: bool,
    engram_project: str,
    out_path: Path,
    semaphore: asyncio.Semaphore,
    ollama_url: str = OLLAMA_URL,
    model: str = OLLAMA_MODEL,
    engram_url: str = EPIMNEME_URL,
    use_assembly: bool = True,
    assembly_budget: int | None = None,
    assembly_k_mode: str = "adaptive",
    assembly_parents: bool = False,
    assembly_skip: tuple[str, ...] = (),
    assembly_recency: bool = False,
    assembly_recency_ungated: bool = False,
) -> None:
    # answer_prompt is now picked per question in the loop (see _pick_prompt)
    # Resume: load already-completed question IDs from output file
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["question_id"])
                    except Exception:
                        pass
        if done_ids:
            print(f"Resuming: {len(done_ids)} questions already done, skipping.")

    remaining = [q for q in questions if q["question_id"] not in done_ids]
    n_total = len(questions)
    n = len(remaining)
    done_count = len(done_ids)

    # Recount hits from already-done results for accurate running acc
    exact_hits = 0
    judge_hits = 0
    type_total: dict[str, int] = {}
    type_exact: dict[str, int] = {}
    type_judge: dict[str, int] = {}
    if done_ids:
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    qt = row.get("question_type", "unknown")
                    type_total[qt] = type_total.get(qt, 0) + 1
                    type_exact.setdefault(qt, 0)
                    type_judge.setdefault(qt, 0)
                    if row.get("exact_match"):
                        exact_hits += 1
                        type_exact[qt] = type_exact.get(qt, 0) + 1
                    elif row.get("judge_match"):
                        judge_hits += 1
                        type_judge[qt] = type_judge.get(qt, 0) + 1
                except Exception:
                    pass

    connector = aiohttp.TCPConnector(limit=4)
    out_fh = open(out_path, "a")  # append so resume works
    async with aiohttp.ClientSession(connector=connector) as session:
        t_start = time.time()
        for i, q in enumerate(remaining, done_count + 1):
            qid = q["question_id"]
            gold = q["answer"]
            question_text = q["question"]
            qtype = q.get("question_type", "unknown")

            type_total[qtype] = type_total.get(qtype, 0) + 1
            type_exact.setdefault(qtype, 0)
            type_judge.setdefault(qtype, 0)

            # Get chunks from prior retrieval results
            r = retrieval_map.get(qid, {})
            ranked = r.get("retrieval_results", {}).get("ranked_items", [])
            question_date = q.get("question_date", "unknown")
            context, chunks_used, assembly_stats = build_context(
                ranked, question_text, qtype, question_date, top_k,
                use_assembly=use_assembly and not use_hyde,
                budget_chars=assembly_budget,
                k_mode=assembly_k_mode,
                fetch_neighbors=make_neighbor_fetcher(q) if (assembly_parents and use_assembly and not use_hyde) else None,
                skip=assembly_skip,
                recency_note=assembly_recency,
                recency_ungated=assembly_recency_ungated,
            )
            chunks = [item["text"] for item in ranked[: (top_k * 2 if qtype == "multi-session" else top_k)]]

            # HyDE: generate hypothetical, search Engram, prepend results
            hyde_chunks: list[str] = []
            if use_hyde:
                # Derive project name from question ID (LME convention: _lme_bench_{qid})
                hyde_project = engram_project or f"_lme_bench_{qid}"
                hyde_chunks = await generate_hyde_chunks(
                    session, question_text, hyde_project, top_k=top_k,
                    ollama_url=ollama_url, model=model, engram_url=engram_url,
                )
                if hyde_chunks:
                    # Prepend HyDE chunks (they're semantically closer to the answer)
                    chunks = hyde_chunks + chunks
                    # Trim to 2*top_k max total
                    chunks = chunks[: top_k * 2]

            if use_hyde:
                # HyDE path keeps the historical raw join (chunks were prepended above)
                context = "\n---\n".join(chunks) if chunks else "(no memory found)"
                chunks_used = len(chunks)
            prompt = _pick_prompt(qtype, use_temporal, temporal_version).format(
                context=context, question=question_text, question_date=question_date
            )

            async with semaphore:
                t0 = time.time()
                # The counting prompt (multi-session) asks for an enumerated list
                # and puts the final number on the LAST line, so an 80-token cap
                # truncates away the answer itself: measured 44/133 multi-session
                # answers cut mid-sentence, hit rate 0.39 against 0.71 for complete
                # ones, and zero truncation in any other type. Give that path room
                # rather than reordering the prompt, which would also remove the
                # enumerate-then-count deliberation.
                gen_cap = counting_max_tokens if qtype == "multi-session" else 80
                generated = await _chat(session, prompt, max_tokens=gen_cap,
                                        ollama_url=ollama_url, model=model)
                gen_time = time.time() - t0

            exact = score_substring(gold, generated)
            judged = False

            # _abs questions: LLM correctly outputs "Unknown" when answer not in memory.
            # Gold is a long "You did not mention..." sentence — substring never matches.
            # Treat generated="Unknown" as a hit for these questions.
            abs_hit = is_abs_question(qid) and normalise(generated) == "unknown"

            if exact or abs_hit:
                exact_hits += 1
                type_exact[qtype] = type_exact.get(qtype, 0) + 1
                hit = True
            elif use_judge and generated.lower() != "unknown":
                async with semaphore:
                    judged = await score_with_judge(session, question_text, gold, generated,
                                                    ollama_url=ollama_url, model=model,
                                                    qtype=qtype)
                if judged:
                    judge_hits += 1
                    type_judge[qtype] = type_judge.get(qtype, 0) + 1
                hit = judged
            else:
                hit = False

            row = {
                "question_id": qid,
                "question_type": qtype,
                "question": question_text,
                "gold": str(gold),
                "generated": generated,
                "chunks_used": chunks_used,
                "assembly": bool(use_assembly and not use_hyde),
                "assembly_k_mode": assembly_k_mode if (use_assembly and not use_hyde) else None,
                "assembly_parents": bool(assembly_parents and use_assembly and not use_hyde),
                "assembly_skip": list(assembly_skip),
                "assembly_recency": bool(assembly_recency),
                **assembly_stats,
                "exact_match": exact,
                "judge_match": judged,
                "hit": hit,
                "gen_time": round(gen_time, 2),
            }
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()

            # Progress log
            elapsed = time.time() - t_start
            completed_this_run = i - done_count
            rate = completed_this_run / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else 0
            total_hits = exact_hits + judge_hits
            print(
                f"[{i:3}/{n_total}] {qtype:30} gold={str(gold)[:20]:22} gen={generated[:22]:24} "
                f"{'✓' if hit else '✗'}  acc={total_hits/i:.3f}  {gen_time:.1f}s  ETA={eta/60:.1f}m",
                flush=True,
            )

    out_fh.close()

    # Summary — count from file
    total = 0
    exact_hits = judge_hits = 0
    type_total = {}; type_exact = {}; type_judge = {}
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            total += 1
            qt = row.get("question_type", "unknown")
            type_total[qt] = type_total.get(qt, 0) + 1
            type_exact.setdefault(qt, 0)
            type_judge.setdefault(qt, 0)
            # `hit` is the authoritative flag (it includes the _abs rule, which
            # neither exact_match nor judge_match records on its own).
            if row.get("judge_match"):
                judge_hits += 1
                type_judge[qt] = type_judge.get(qt, 0) + 1
            elif row.get("hit"):
                exact_hits += 1
                type_exact[qt] = type_exact.get(qt, 0) + 1
    total_hits = exact_hits + judge_hits
    print("\n" + "=" * 70)
    print(f"FINAL E2E SCORE: {total_hits}/{total} = {total_hits/total:.4f}")
    print(f"  Exact match:   {exact_hits}/{total} = {exact_hits/total:.4f}")
    if use_judge:
        print(f"  Judge rescue:  {judge_hits}/{total} = {judge_hits/total:.4f}")
    print()
    print(f"{'Type':<32} {'N':>4}  {'Exact':>6}  {'Total':>6}")
    print("-" * 55)
    for qtype in sorted(type_total):
        n_t = type_total[qtype]
        ex = type_exact.get(qtype, 0)
        jd = type_judge.get(qtype, 0)
        tot = ex + jd
        print(f"  {qtype:<30} {n_t:4}  {ex/n_t:6.3f}  {tot/n_t:6.3f}")
    print("=" * 70)
    print(f"Results saved to: {out_path}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    global OLLAMA_NUM_CTX
    ap = argparse.ArgumentParser(description="End-to-end LME benchmark via Ollama")
    ap.add_argument(
        "--retrieval-results", default="",
        help="Path to JSONL file from a prior lme retrieval run (ranked_items required)",
    )
    ap.add_argument(
        "--lme-data",
        default="benchmarks/data/longmemeval_s_cleaned.json",
        help="Path to LME question file",
    )
    ap.add_argument("--out", default="/tmp/lme_e2e.jsonl", help="Output JSONL path")
    ap.add_argument("--top-k", type=int, default=10, help="Chunks to feed per question")
    ap.add_argument("--limit", type=int, default=0, help="Limit to first N questions (0=all)")
    ap.add_argument(
        "--judge", action="store_true",
        help="Run LLM judge pass on substring-match misses",
    )
    ap.add_argument(
        "--temporal", action="store_true",
        help="Use temporal-aware prompt (instructs LLM to use [Date: ...] headers)",
    )
    ap.add_argument(
        "--hyde", action="store_true",
        help="Pre-generate hypothetical memory excerpt, embed, search via Engram API",
    )
    ap.add_argument(
        "--engram-project", default="",
        help="Engram project name for HyDE search (required if --hyde)",
    )
    ap.add_argument(
        "--engram-url", default=EPIMNEME_URL,
        help="Engram API base URL (for --hyde)",
    )
    ap.add_argument(
        "--ollama-url", default=OLLAMA_URL,
        help="Ollama API base URL",
    )
    ap.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    ap.add_argument(
        "--num-ctx", type=int, default=OLLAMA_NUM_CTX,
        help=f"Ollama context window in tokens (default: {OLLAMA_NUM_CTX}); must fit "
             "top-k full-length chunks plus the prompt or Ollama silently drops the front",
    )
    ap.add_argument(
        "--no-assembly", action="store_true",
        help="Reproduce the historical raw '---'-joined top-K context instead of "
             "epimneme.assembly.assemble_context (adaptive K, supersession pruning, "
             "char budget, session grouping, chronological order, date deltas)",
    )
    ap.add_argument(
        "--assembly-k", choices=["adaptive", "pool"], default="adaptive",
        help="adaptive = assembly's own K selection (5/10/20 by query shape; can narrow "
             "the pool); pool = keep the whole harness pool, apply only presentation transforms",
    )
    ap.add_argument(
        "--assembly-parents", action="store_true",
        help="Enable parent-document expansion: splice the n-1/n+1 turn-pair siblings "
             "of each hit (rebuilt from the LME haystack) before budgeting",
    )
    ap.add_argument(
        "--assembly-skip", default="",
        help="Comma-separated assembly steps to leave out (ablation): prune,group,chrono,dates",
    )
    ap.add_argument(
        "--assembly-recency", action="store_true",
        help="Prepend the recency note (latest-dated value is current) to the assembled context",
    )
    ap.add_argument(
        "--assembly-recency-ungated", action="store_true",
        help="Emit the recency note even on counting queries. The default gate also "
             "suppresses it on 'how many X do I have now', which is one updated fact "
             "rather than an aggregation, costing knowledge-update accuracy.")
    ap.add_argument(
        "--counting-max-tokens", type=int, default=80,
        help="Generation cap for multi-session (counting) questions. The counting "
             "prompt puts the answer on the LAST line, so the default 80 truncates it "
             "away on a third of those questions. 256 gives the list room to finish.")
    ap.add_argument(
        "--temporal-v3", action="store_true",
        help="v3 temporal prompt: v2 plus the two defects it exposed — keep the unit "
             "on elapsed times, and name events for ordering questions instead of "
             "answering with indices.")
    ap.add_argument(
        "--temporal-v2", action="store_true",
        help="Use the v2 temporal answer prompt: tells the reader the answer is DERIVED "
             "from dates and to refuse only when the underlying events are missing. v1's "
             "blanket 'say Unknown if not found' licenses refusal on every computed answer.")
    ap.add_argument(
        "--types", default="",
        help="Comma-separated question types to run (default: all), e.g. multi-session",
    )
    ap.add_argument(
        "--assembly-budget", type=int, default=0,
        help="Assembly char budget (0 = epimneme.assembly.DEFAULT_BUDGET_CHARS)",
    )
    ap.add_argument(
        "--rescore-only", action="store_true",
        help="Re-score an existing output JSONL (--out) without regenerating answers. "
             "Applies _abs fix and optionally runs --judge on misses. "
             "Writes rescored output to --out + '.rescored.jsonl'.",
    )
    args = ap.parse_args()
    OLLAMA_NUM_CTX = args.num_ctx

    # --rescore-only mode: re-score existing JSONL in-place
    if args.rescore_only:
        _rescore_jsonl(
            in_path=Path(args.out),
            use_judge=args.judge,
            ollama_url=args.ollama_url,
            model=args.model,
        )
        return

    if not args.retrieval_results:
        ap.error("--retrieval-results is required unless --rescore-only")

    # Load questions
    lme_path = Path(args.lme_data)
    if not lme_path.exists():
        # Try relative to this script's directory
        lme_path = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"
    with open(lme_path) as f:
        questions = json.load(f)
    if args.types:
        wanted = {t.strip() for t in args.types.split(",") if t.strip()}
        questions = [q for q in questions if q.get("question_type") in wanted]
    if args.limit > 0:
        questions = questions[: args.limit]

    # Load retrieval results
    retrieval_map: dict[str, dict] = {}
    with open(args.retrieval_results) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                retrieval_map[rec["question_id"]] = rec

    out_path = Path(args.out)
    semaphore = asyncio.Semaphore(1)  # Ollama is serial; 1 keeps pipeline clean

    print(f"E2E LME benchmark")
    print(f"  Model:       {args.model} (think=false)")
    print(f"  Top-K:       {args.top_k}")
    print(f"  Questions:   {len(questions)}")
    print(f"  Temporal:    {args.temporal}")
    print(f"  HyDE:        {args.hyde}")
    print(f"  Judge pass:  {args.judge}")
    print(f"  Assembly:    {not args.no_assembly}"
          + (f" (k={args.assembly_k}, parents={args.assembly_parents}, budget={args.assembly_budget or 'default'}, skip={args.assembly_skip or '-'}, recency={args.assembly_recency})" if not args.no_assembly else ""))
    print(f"  Types:       {args.types or 'all'}")
    print(f"  num_ctx:     {OLLAMA_NUM_CTX}")
    print(f"  Output:      {out_path}")
    print()

    asyncio.run(
        run(
            questions=questions,
            retrieval_map=retrieval_map,
            top_k=args.top_k,
            use_judge=args.judge,
            use_temporal=args.temporal,
            temporal_version=("v3" if args.temporal_v3 else "v2" if args.temporal_v2 else "v1"),
            counting_max_tokens=args.counting_max_tokens,
            use_hyde=args.hyde,
            engram_project=args.engram_project,
            out_path=out_path,
            semaphore=semaphore,
            ollama_url=args.ollama_url,
            model=args.model,
            engram_url=args.engram_url,
            use_assembly=not args.no_assembly,
            assembly_budget=args.assembly_budget or None,
            assembly_k_mode=args.assembly_k,
            assembly_parents=args.assembly_parents,
            assembly_skip=tuple(t.strip() for t in args.assembly_skip.split(",") if t.strip()),
            assembly_recency=args.assembly_recency,
            assembly_recency_ungated=args.assembly_recency_ungated,
        )
    )


def _rescore_jsonl(
    in_path: Path,
    use_judge: bool,
    ollama_url: str = OLLAMA_URL,
    model: str = OLLAMA_MODEL,
) -> None:
    """Re-score an existing result JSONL without regenerating answers.

    Applies:
    - _abs question fix (generated=Unknown → hit=True)
    - Optional LLM judge pass on substring misses
    """
    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(1)

    out_path = in_path.with_suffix(".rescored.jsonl")
    rows: list[dict] = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    print(f"Rescoring {len(rows)} rows from {in_path}")
    print(f"  Judge pass: {use_judge}")
    print(f"  Output:     {out_path}")

    async def _rescore() -> None:
        connector = aiohttp.TCPConnector(limit=4)
        async with aiohttp.ClientSession(connector=connector) as session:
            semaphore = asyncio.Semaphore(1)
            for row in rows:
                qid = row["question_id"]
                gold = row["gold"]
                generated = row["generated"]
                question_text = row["question"]

                # Preserve any hit already established by the original run.
                # Also re-apply score_substring with current (improved) logic
                # to rescue cases the old scorer missed.
                already_hit = row.get("hit", False)
                exact_now = score_substring(gold, generated)
                abs_hit = is_abs_question(qid) and normalise(generated) == "unknown"
                judged = False

                if already_hit or exact_now or abs_hit:
                    hit = True
                elif use_judge and generated.lower() != "unknown":
                    async with semaphore:
                        judged = await score_with_judge(
                            session, question_text, gold, generated,
                            ollama_url=ollama_url, model=model,
                            qtype=row.get("question_type", ""),
                        )
                    hit = judged
                else:
                    hit = False

                row["judge_match"] = judged
                row["hit"] = hit
                row["abs_hit"] = abs_hit

    asyncio.run(_rescore())

    # Write out
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Summary
    total = len(rows)
    total_hits = sum(1 for r in rows if r["hit"])
    abs_hits = sum(1 for r in rows if r.get("abs_hit"))
    judge_hits = sum(1 for r in rows if r["judge_match"])
    type_total: dict[str, int] = {}
    type_hits: dict[str, int] = {}
    for row in rows:
        qt = row.get("question_type", "unknown")
        type_total[qt] = type_total.get(qt, 0) + 1
        if row["hit"]:
            type_hits[qt] = type_hits.get(qt, 0) + 1

    print("\n" + "=" * 70)
    print(f"RESCORED: {total_hits}/{total} = {total_hits/total:.4f}")
    print(f"  _abs fix rescued:  {abs_hits}")
    if use_judge:
        print(f"  Judge rescued:     {judge_hits}")
    print()
    print(f"{'Type':<32} {'N':>4}  {'Hits':>5}  {'Acc':>6}")
    print("-" * 55)
    for qt in sorted(type_total):
        n_t = type_total[qt]
        h = type_hits.get(qt, 0)
        print(f"  {qt:<30} {n_t:4}  {h:5}  {h/n_t:6.3f}")
    print("=" * 70)
    print(f"Rescored results saved to: {out_path}")


if __name__ == "__main__":
    main()
