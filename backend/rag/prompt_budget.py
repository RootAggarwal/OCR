"""
prompt_budget.py — assemble the EVIDENCE block under a hard token ceiling.

Exists because the tables lane feeds whole HTML tables to the model, and HTML is far more
token-dense than prose. Measured against this model's own tokenizer:

    KJ-12 p9 mud table   2,612 chars -> 1,514 tokens   (1.73 chars/token)
    English prose          137 chars ->    35 tokens   (3.91 chars/token)

So one real table is ~24% of the entire input budget, and a char-based allocator sized on
the usual "~4 chars per token" intuition would overflow by roughly 2x.

Overflowing is not a degraded answer. Before the ContextOverflow fix in llm_client it was a
silently DIFFERENT MODEL: vLLM 400s, the client fell back to the local Qwen2.5-VL (32k
window, accepts the same prompt), and backend_name() still said "vllm:qwen3-14b-awq". That
is why this module counts real tokens and re-counts the finished string before returning.

Three contracts, in priority order:
  1. NEVER truncate a table mid-tag. A decapitated table is worse than no table: the model
     reads a half-written <td> as a value and reports it with full confidence. This is not
     hypothetical — it is the bug this whole lane was built to fix, where a 400-char clip
     delivered a 13-row table as 2 rows cut mid-cell.
  2. NEVER exceed max_model_len - output reserve - safety margin.
  3. Whatever is dropped is REPORTED. A table omitted for space must never be
     indistinguishable from a table that does not exist.
"""

import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

_LLM_URL = os.getenv("RAG_LLM_URL", "").rstrip("/")
_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3-14b-awq")

# Tokens reserved for the answer. Must match the max_tokens synthesize passes.
OUTPUT_RESERVE = int(os.getenv("RAG_OUTPUT_RESERVE", "2000"))
# Slack for chat-template scaffolding (role markers etc.) that /tokenize on a bare prompt
# does not see, plus drift between the tokenizer and the server's own accounting.
SAFETY_MARGIN = int(os.getenv("RAG_SAFETY_MARGIN", "256"))
# Never build a prompt with no room for prose at all.
PROSE_FLOOR_TOKENS = 300
TABLE_FLOOR_TOKENS = 200

_DEFAULT_MAX_LEN = 8192
_cache = threading.local()
_llm_probe_failed = False


def _tokenize_url() -> str:
    """/tokenize sits at the server root, NOT under the /v1 prefix that RAG_LLM_URL
    carries for the OpenAI-compatible routes."""
    base = _LLM_URL[:-3] if _LLM_URL.endswith("/v1") else _LLM_URL
    return base.rstrip("/") + "/tokenize"


def _probe(text: str) -> Optional[Dict[str, Any]]:
    global _llm_probe_failed
    if not _LLM_URL or _llm_probe_failed:
        return None
    try:
        req = urllib.request.Request(
            _tokenize_url(),
            data=json.dumps({"model": _MODEL, "prompt": text}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1.5) as r:
            return json.loads(r.read().decode())
    except Exception:
        _llm_probe_failed = True
        return None


def max_model_len() -> int:
    """The server's real context window. vLLM's /tokenize reports it, so we ask rather
    than hardcode — a redeployed container with a different --max-model-len must not
    silently invalidate every budget calculation here."""
    cached = getattr(_cache, "max_len", None)
    if cached:
        return cached
    d = _probe("x") or {}
    n = int(d.get("max_model_len") or 0) or _DEFAULT_MAX_LEN
    _cache.max_len = n
    return n


# Conservative chars-per-token when the tokenizer is unreachable. Deliberately BELOW the
# measured values (1.73 HTML / 3.91 prose) so the estimate over-counts and we under-fill.
_CPT_HTML = 1.5
_CPT_TEXT = 3.0


def count_tokens(text: str, kind: str = "text") -> int:
    """Real token count, falling back to a pessimistic per-kind estimate.

    The fallback must over-estimate: under-estimating produces a prompt the server
    rejects, and a rejected prompt is the failure mode this module exists to prevent.
    """
    if not text:
        return 0
    d = _probe(text)
    if d:
        n = d.get("count")
        if isinstance(n, int) and n >= 0:
            return n
        toks = d.get("tokens")
        if isinstance(toks, list):
            return len(toks)
    return int(len(text) / (_CPT_HTML if kind == "html" else _CPT_TEXT)) + 1


_TABLE_RE = re.compile(r"<table\b[\s\S]*?(?:</table>|$)", re.IGNORECASE)


def strip_tables(text: str) -> str:
    """Remove <table> blocks from prose text.

    Prose chunks and page_tables overlap: a table harvested into page_tables is usually
    still sitting inside the prose chunk it came from. Without this the same table is sent
    twice — once whole in the tables block and once decapitated in the excerpt — which
    wastes the scarcest resource in the prompt and invites the model to treat the truncated
    copy as a second, conflicting source.
    """
    return _TABLE_RE.sub(" [table shown above] ", text or "").strip()


def assemble_evidence(question: str, system_prompt: str,
                      table_hits: List[Dict[str, Any]],
                      prose_hits: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Build the EVIDENCE block. Returns (evidence_text, report).

    report = {included, dropped, malformed, prose_included, prose_clipped,
              input_tokens, limit}
    `dropped` entries carry a reason so synthesize can turn each into a named warning.
    """
    limit = max_model_len() - OUTPUT_RESERVE - SAFETY_MARGIN
    fixed = (count_tokens(system_prompt) + count_tokens(question) + 40)  # 40 = scaffolding
    report: Dict[str, Any] = {
        "included": [], "dropped": [], "malformed": [],
        "prose_included": 0, "prose_clipped": 0,
        "limit": limit, "input_tokens": fixed,
    }
    avail = limit - fixed
    if avail < TABLE_FLOOR_TOKENS + PROSE_FLOOR_TOKENS:
        # Structural, not data-dependent: the static prompt alone is eating the window.
        # Fail loudly rather than emit a prompt guaranteed to bounce.
        raise RuntimeError(
            f"only {avail} tokens left for evidence (limit {limit}, fixed overhead {fixed}) "
            f"— shorten the system prompt or raise --max-model-len")

    parts: List[str] = []

    # ── Tables first: they are the point of this lane. All-or-nothing. ──────────
    table_budget = avail - PROSE_FLOOR_TOKENS
    used = 0
    blocks: List[str] = []
    for h in table_hits:
        html = (h.get("html") or "").strip()
        if not html:
            continue
        # Rectangularize before showing it to the model. A rowspan group writes its
        # spanning cell only once, so rows beneath it carry fewer <td>s than there are
        # columns, and the model has to INFER which header each value belongs to. Measured:
        # a 14B model asked for "Yp" on such a row returned the neighbouring Pv value, and
        # after the prompt was tightened to warn about precisely that, it shifted one column
        # the other way instead — confidently wrong both times. Expanding the span states
        # what rowspan already means and turns alignment into a count. The raw HTML keeps
        # its spans in page_tables; only this prompt copy is rewritten. (It is also
        # smaller: 2,605 -> 2,092 chars on the reference table.)
        try:
            import table_extractor
            html = table_extractor.expand_table_for_prompt(html)
        except Exception:
            pass  # unexpandable is still worth showing
        label = f"{h.get('filename')} p{h.get('page')}"
        head = f"[T{len(blocks) + 1}] ({label}"
        if h.get("caption"):
            head += f" — {h['caption']}"
        head += ")"
        if not h.get("well_formed", True):
            head += "  [INCOMPLETE OCR READ: the transcription stops mid-table]"
            report["malformed"].append({"filename": h.get("filename"), "page": h.get("page")})
        block = f"{head}\n{html}"
        cost = count_tokens(block, kind="html")
        if used + cost > table_budget:
            report["dropped"].append({
                "filename": h.get("filename"), "page": h.get("page"),
                "caption": h.get("caption"),
                "reason": "too_large" if cost > table_budget else "no_room",
                "tokens": cost,
            })
            continue
        used += cost
        blocks.append(block)
        report["included"].append({
            "filename": h.get("filename"), "page": h.get("page"),
            "n_rows": h.get("n_rows"), "n_cols": h.get("n_cols"), "tokens": cost,
        })

    # ── Prose with whatever remains. ───────────────────────────────────────────
    prose_budget = avail - used
    lines: List[str] = []
    for h in prose_hits:
        body = strip_tables(h.get("content") or "")
        if not body:
            continue
        line = f"[{len(lines) + 1}] ({h.get('filename')} p{h.get('page')}) {body}"
        cost = count_tokens(line)
        if cost > prose_budget:
            room = max(0, prose_budget)
            if room < 60:
                break
            approx = int(room * _CPT_TEXT)
            line = line[:approx].rstrip() + " …[truncated]"
            cost = count_tokens(line)
            if cost > prose_budget:
                break
            report["prose_clipped"] += 1
        prose_budget -= cost
        used += cost
        lines.append(line)
        report["prose_included"] += 1

    is_summary = any(k in question.lower() for k in ("summar", "overview", "what is this", "about", "describe", "explain the pdf"))
    if is_summary:
        if lines:
            parts.append("DOCUMENT EXCERPTS:\n" + "\n".join(lines))
        if blocks:
            parts.append("TABLES (verbatim from the source pages; read values directly):\n"
                         + "\n\n".join(blocks))
    else:
        if blocks:
            parts.append("TABLES (verbatim from the source pages; read values directly):\n"
                         + "\n\n".join(blocks))
        if lines:
            parts.append("DOCUMENT EXCERPTS:\n" + "\n".join(lines))

    evidence = "\n\n".join(parts) if parts else "No evidence was retrieved."

    # Re-count the assembled string rather than trusting the running sum: per-part counts
    # do not include the joins, and tokenizers are not additive across concatenation.
    total = fixed + count_tokens(evidence, kind="html" if blocks else "text")
    report["input_tokens"] = total
    if total > limit:
        # Last-resort: shed whole tables, largest first, until it fits. Still never a
        # partial table.
        while blocks and total > limit:
            dropped_block = blocks.pop()
            moved = report["included"].pop() if report["included"] else {}
            report["dropped"].append({**moved, "reason": "no_room"})
            parts[0] = ("TABLES (verbatim from the source pages; read values directly):\n"
                        + "\n\n".join(blocks)) if blocks else ""
            evidence = "\n\n".join(p for p in parts if p) or "No evidence was retrieved."
            total = fixed + count_tokens(evidence, kind="html" if blocks else "text")
        report["input_tokens"] = total

    return evidence, report
