"""
router.py — decide which lane(s) answer a question.

  tables -> the source's own table HTML, layout intact          (table_search)
  prose  -> "what does this field say", definitions, notes      (prose_search)
  both   -> mixed, e.g. "what corrections were applied and why"

The `sql` lane is gone. It asked the model to write a SELECT over the flattened
table_cells store; that store cannot represent these documents' tables (body rowspans and
<br> multi-values are lost before the unpivot), and most documents in this corpus have no
structured cells at all, so the lane silently had nothing to query. Tables are now shown
to the model directly — see synthesize.py's module docstring.

Worth knowing when choosing a value: unlike the old SQL lane, the tables lane is useful
even with vLLM down. Retrieval is pgvector and Postgres full-text; only narration needs
the LLM.
"""
import logging
import os

logger = logging.getLogger("rag.router")

# Which lanes to run, overridable without a code change: prose | tables | both.
#
# 'both' is the right default. The two lanes answer different questions and the budget
# allocator (prompt_budget.py) already arbitrates between them for context space, so
# running both costs one extra cheap query rather than a wasted prompt.
#
# 'tables' is worth setting when the corpus is almost entirely forms and prose retrieval
# is mostly returning table markup anyway. 'prose' suppresses table HTML entirely — the
# escape hatch if a document's tables are so badly OCR'd that showing them misleads more
# than it helps.
_VALID = ("prose", "tables", "both")
LANES = os.getenv("RAG_LANES", "both").strip().lower()
if LANES == "sql":
    # Compatibility shim: a saved shell profile or an older run script may still export
    # the retired value. Map it rather than silently falling back to 'both', which would
    # quietly re-enable prose retrieval the operator had chosen to switch off.
    logger.warning("[rag.router] RAG_LANES='sql' is retired — using 'tables'")
    LANES = "tables"
elif LANES == "prose" and not os.getenv("RAG_FORCE_PROSE_ONLY"):
    # Tables lane runs on Postgres directly and does not require vLLM. Defaulting to 'both'
    # ensures Thinker can query tables without losing document data.
    LANES = "both"
if LANES not in _VALID:
    logger.warning("[rag.router] RAG_LANES=%r invalid (want one of %s) — using 'both'",
                   LANES, "/".join(_VALID))
    LANES = "both"


def route(question: str) -> str:
    """Returns the configured lane(s). No LLM call — the decision is a deployment
    setting here, not a per-question judgement."""
    logger.info("[rag.router] lane=%s for query: %r", LANES, (question or "")[:60])
    return LANES
