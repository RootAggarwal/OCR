"""
aggregate_guard.py — make the cost of retiring the text-to-SQL lane VISIBLE.

The old lane could answer "average Receiver correction across 147 rows" by running an
actual SELECT: Postgres computed the number and the model only copied it. The
HTML-in-context lane cannot. At ~1.7 chars/token, 147 rows do not fit in an 8192-token
window, and even when they do, the model is doing mental arithmetic over a table — which
it does unreliably and confidently.

The failure this module prevents is not "wrong average". It is "wrong average, stated in
exactly the same tone as a correct lookup". Every verdict here is computed in code from
the question text and the BUDGET REPORT — what actually reached the prompt — never by
asking the model, which can no more be trusted to notice it lacked the data than to notice
it lacked a citation.

Deliberately NOT included: a table_cells-based deterministic aggregate fallback. It would
only work for documents that have structured cells (on this corpus, one of three), so it
would answer some aggregates and refuse others with no user-visible reason — exactly the
silent inconsistency this module exists to prevent. Revisit only if cell coverage becomes
universal.
"""

import re
from typing import Any, Dict, List

_AGG_RE = re.compile(
    r"\b(average|avg|mean|median|sum|total|totals|aggregate|count|how many|number of|"
    r"minimum|maximum|min|max|highest|lowest|largest|smallest|most|least|"
    r"across all|every row|all rows|overall|percentage|per cent|distribution|trend)\b",
    re.IGNORECASE)

# Row count above which model arithmetic stops being trustworthy. Deliberately low: the
# point is not to find the true limit, it is to refuse well before reaching it.
_TRUSTWORTHY_ROWS = 25


def classify(question: str, report: Dict[str, Any], doc_scoped: bool = False) -> Dict[str, Any]:
    """-> {"is_aggregate", "verdict", "warnings": [...], "prompt_note": str}

    verdicts:
      not_aggregate — nothing to do.
      answerable    — one complete, small table is fully in the prompt. Allow the model to
                      compute, but label the result as model-computed, not looked up.
      partial       — evidence was dropped or is incomplete. Refuse, and name what is
                      missing so the user can ask a narrower question.
      unsupported   — the question ranges over more data than any prompt can hold.
    """
    if not _AGG_RE.search(question or ""):
        return {"is_aggregate": False, "verdict": "not_aggregate",
                "warnings": [], "prompt_note": ""}

    included: List[Dict] = report.get("included") or []
    dropped: List[Dict] = report.get("dropped") or []
    malformed: List[Dict] = report.get("malformed") or []
    rows = sum(int(t.get("n_rows") or 0) for t in included)

    if dropped or malformed or report.get("prose_clipped"):
        missing = ", ".join(sorted({f"{d.get('filename')} p{d.get('page')}" for d in dropped})) or "—"
        extra = (f" and {len(malformed)} is/are incomplete OCR reads" if malformed else "")
        return {
            "is_aggregate": True, "verdict": "partial",
            "warnings": [
                f"⚠ This asks for a calculation, but not all relevant data reached the "
                f"model: {len(dropped)} table(s) were left out ({missing}){extra}. Any "
                f"figure below is computed from a SUBSET and is not the total.",
                "⚠ Aggregates are no longer computed by a database engine. Ask about a "
                "single page or table, or export the parsed numeric cells from "
                "/rag/corpus-tables and compute them there.",
            ],
            "prompt_note": (
                "NOTE: the evidence above is INCOMPLETE — table(s) were omitted for space. "
                "Do NOT compute an average, sum, count or total. State which pages you can "
                "see and say the calculation cannot be completed from them."),
        }

    if not included:
        return {
            "is_aggregate": True, "verdict": "unsupported",
            "warnings": ["⚠ No table reached the model for this calculation, so nothing "
                         "below is computed from tabular data."],
            "prompt_note": ("NOTE: no table data is present. Do not compute or estimate "
                            "any total, average or count."),
        }

    if len(included) == 1 and rows and rows <= _TRUSTWORTHY_ROWS:
        t = included[0]
        return {
            "is_aggregate": True, "verdict": "answerable",
            "warnings": [f"ℹ This figure was calculated by the language model from the "
                         f"{rows} rows of {t['filename']} p{t['page']} shown to it — not by "
                         f"a database. Check it against the table before relying on it."],
            "prompt_note": (
                f"NOTE: you may compute over the {rows} rows of table [T1] because ALL of "
                "them are shown above. Show the individual values you used. Do not compute "
                "over anything else."),
        }

    scope = "the selected documents" if doc_scoped else "the corpus"
    return {
        "is_aggregate": True, "verdict": "unsupported",
        "warnings": [
            f"⚠ This asks for a calculation over ~{rows} rows across {len(included)} "
            f"table(s) — more than can be verified. This system reads tables as text and "
            f"does not run calculations across {scope}.",
            "⚠ For a real aggregate, export the parsed numeric cells from "
            "/rag/corpus-tables and compute them there.",
        ],
        "prompt_note": (
            "NOTE: do NOT compute an average, sum, count, minimum or maximum. Say plainly "
            "that this system reads tables as text and cannot calculate across them, then "
            "answer whatever part of the question IS a direct lookup."),
    }
