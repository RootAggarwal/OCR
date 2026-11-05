"""
llm_client.py — one LLM abstraction for the RAG lanes.

Two backends, chosen by env:
  * RAG_LLM_URL set  -> OpenAI-compatible /chat/completions (the production path: a vLLM
                        server hosting Qwen3-32B-AWQ). Deterministic (temperature 0).
  * RAG_LLM_URL unset-> LOCAL fallback via ocr_olm.run_plain_text_inference (Qwen2.5-VL,
                        already resident in the OCR server). Lets us build and validate
                        text-to-SQL before standing up the 32B server. Must be called from
                        inside the OCR server process (that's where the model lives).
"""

import json
import os
import threading
import urllib.error
import urllib.request

RAG_LLM_URL = os.getenv("RAG_LLM_URL", "").rstrip("/")
RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3-32b-awq")

# Opt-in escape hatch for running with no vLLM server at all.
#
# Lanes that must not silently degrade (text-to-SQL) call complete(allow_fallback=False),
# which raises PrimaryUnavailable when the primary is down — the whole Thinker analytical
# tab then reports "start the vLLM server" and answers nothing. That default is the right
# one when vLLM is merely *down*; it is the wrong one when vLLM is not installed on the
# box at all (no Windows build — the server runs in Docker), because it makes the lane
# permanently dead rather than degraded.
#
# Set RAG_SQL_LOCAL_FALLBACK=1 to let those lanes use the resident Qwen2.5-VL instead.
# Treat its SQL as advisory: the docstring below and run_project.sh disagree about how
# good the local model is at SQL ("cannot reliably write correct SQL" vs "~100% on the SQL
# eval"), and that claim is unverified against the current corpus. Answers carry a
# degraded backend name so the caller can tell which model produced them.
SQL_LOCAL_FALLBACK = os.getenv("RAG_SQL_LOCAL_FALLBACK", "0") == "1"


class ContextOverflow(Exception):
    """The primary REJECTED the request (HTTP 400) — usually an over-length prompt.

    Never falls back, deliberately. A 400 is our bug (evidence budget miscalculated, bad
    parameter), not an outage, and falling back here would be the most dangerous kind of
    silent failure this codebase has:

      * the local Qwen2.5-VL has a ~32k window, so it accepts the SAME oversized prompt
        and answers it, worse;
      * backend_name() would still report "vllm:<model>", because it infers from config
        and the vLLM server IS reachable — only this one request was too long.

    The user would receive a materially worse answer, correctly formatted, attributed to
    the wrong model, with nothing but a line in the server log. Surface it instead.
    """


class PrimaryUnavailable(Exception):
    """Raised by complete(..., allow_fallback=False) when RAG_LLM_URL is configured but
    unreachable. Callers that need the capable model for correctness (text-to-SQL) pass
    allow_fallback=False and must surface this rather than silently degrading: the local
    7B vision fallback cannot reliably write correct SQL, and a wrong/garbled query is
    worse than a clear "service unavailable" message."""


def complete(prompt: str, system: str = "", max_tokens: int = 400, temperature: float = 0.0,
            allow_fallback: bool = True) -> str:
    """
    Return the model's text completion. Deterministic by default (temperature 0).

    allow_fallback=False: if RAG_LLM_URL is set but the request fails, raise
    PrimaryUnavailable instead of silently degrading to the local model. Use this for
    lanes where a wrong answer is worse than a visible failure (text-to-SQL). Lanes where
    the local model is an acceptable degradation (routing, prose narration) should leave
    this at the default True.
    """
    # RAG_SQL_LOCAL_FALLBACK overrides an explicit allow_fallback=False — see the flag's
    # comment above: it exists for boxes where vLLM is absent, not merely unreachable.
    may_degrade = allow_fallback or SQL_LOCAL_FALLBACK

    if RAG_LLM_URL:
        if not _primary_reachable():
            if not may_degrade:
                raise PrimaryUnavailable(f"Primary LLM ({RAG_LLM_MODEL}) at {RAG_LLM_URL} is unreachable")
            _LAST_BACKEND.value = f"local:qwen2.5-vl (fallback; {RAG_LLM_MODEL} unreachable)"
            return _local_complete(prompt, system, max_tokens, temperature)
        try:
            out = _openai_complete(prompt, system, max_tokens, temperature)
            _LAST_BACKEND.value = f"vllm:{RAG_LLM_MODEL}"
            return out
        except urllib.error.HTTPError as e:
            # A rejected request and a dead server are different events and must not share
            # a policy. 4xx means the request itself was wrong (over-length prompt, bad
            # parameter) — degrading would hide our own bug behind a worse answer that is
            # still labelled as the good model. See ContextOverflow.
            if 400 <= e.code < 500:
                body = ""
                try:
                    body = e.read().decode(errors="replace")[:400]
                except Exception:
                    pass
                raise ContextOverflow(f"HTTP {e.code} from {RAG_LLM_MODEL}: {body}") from e
            _log_degrade(e, may_degrade)          # 5xx: server-side, treat as unavailable
            if not may_degrade:
                raise PrimaryUnavailable(str(e)) from e
        except Exception as e:
            _log_degrade(e, may_degrade)
            if not may_degrade:
                raise PrimaryUnavailable(str(e)) from e
        _LAST_BACKEND.value = f"local:qwen2.5-vl (fallback; {RAG_LLM_MODEL} unreachable)"
        return _local_complete(prompt, system, max_tokens, temperature)

    if not may_degrade:
        # No vLLM configured at all: there is no "primary" to have failed, but the caller
        # explicitly opted out of the weak local model, so still refuse rather than guess.
        raise PrimaryUnavailable("RAG_LLM_URL is not configured")
    _LAST_BACKEND.value = "local:qwen2.5-vl"
    return _local_complete(prompt, system, max_tokens, temperature)


def _log_degrade(e: Exception, may_degrade: bool) -> None:
    import logging
    logging.getLogger("rag.llm_client").warning(
        "vLLM connection failed (%s).%s", e,
        "" if not may_degrade else " Falling back to local model.")


# What actually served the most recent complete() on THIS thread. The server answers
# requests from a threadpool, so thread-local is the right scope.
_LAST_BACKEND = threading.local()


def last_backend() -> str:
    """The backend that ACTUALLY ran, not the one the configuration implies.

    Prefer this over backend_name() anywhere the value is shown to a user.
    backend_name() only guesses: with RAG_SQL_LOCAL_FALLBACK unset it reports
    "vllm:<model>" even when the local model wrote the answer, because it has no record
    of what happened — it just re-reads env vars and pings the server.
    """
    return getattr(_LAST_BACKEND, "value", None) or backend_name()


def backend_name() -> str:
    """The backend the CONFIGURATION implies. Inferential — see last_backend(), which
    reports what actually ran. Kept for callers that need a label before any call."""
    if not RAG_LLM_URL:
        return "local:qwen2.5-vl"
    if SQL_LOCAL_FALLBACK and not _primary_reachable():
        return f"local:qwen2.5-vl (fallback; {RAG_LLM_MODEL} unreachable)"
    return f"vllm:{RAG_LLM_MODEL}"


def _primary_reachable() -> bool:
    """Cheap liveness probe for the label above. Short timeout: this runs on the response
    path and must never add latency comparable to the generation it is describing."""
    try:
        urllib.request.urlopen(RAG_LLM_URL + "/models", timeout=1.5).read()
        return True
    except Exception:
        return False


def _openai_complete(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    body = {"model": RAG_LLM_MODEL, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            # Qwen3 is a hybrid-reasoning model: without this it emits <think>...</think>
            # before the answer, which corrupts SQL extraction and wastes tokens. Turn the
            # thinking mode off for the structured lanes. (Ignored by non-Qwen servers.)
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        RAG_LLM_URL + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"]


def _local_complete(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    # Uses the Qwen2.5-VL text path already loaded by the OCR server. Low temperature
    # approximates greedy decoding (run_plain_text_inference always samples).
    import ocr_olm
    full = (system + "\n\n" if system else "") + prompt
    return ocr_olm.run_plain_text_inference(
        full, max_new_tokens=max_tokens, temperature=max(temperature, 0.01))
