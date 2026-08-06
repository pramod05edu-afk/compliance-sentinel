"""
sentinel_groq.py — Free-tier Groq adapter for the LLM rung.

sentinel_llm.py was written against the Anthropic client. Rather than rewrite it, this thin
adapter exposes the SAME small interface (a client object with .messages.create(...) and a
classify function) but routes to Groq's free API. All the actual logic — prompt, injection
defense, JSON parsing, batching — is reused unchanged from sentinel_llm.py.

Why Groq: free, no credit card, and (unlike some free tiers) it does NOT train on your prompts,
which is the right call for a data-ethics project.

Model note: Groq deprecated llama-3.3-70b-versatile (June 2026). Default here is the current
free recommended model 'openai/gpt-oss-120b'. If that ever errors, check console.groq.com/docs/models
and swap MODEL below.
"""
from __future__ import annotations
import os
import time
import groq

import sentinel_core as sc
import sentinel_llm as sl

MODEL = "openai/gpt-oss-120b"     # current free Groq model; fallback: "llama-3.3-70b-versatile"


def get_client(api_key: str | None = None) -> groq.Groq:
    """Build a Groq client. Reads GROQ_API_KEY from the environment if no key is passed."""
    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])


def classify_with_groq(text: str, client: groq.Groq, system_prompt: str,
                       model: str = MODEL, max_retries: int = 3) -> dict:
    """
    Same contract as sentinel_llm.classify_with_llm, but via Groq's chat.completions API.
    Returns {violates, policies, confidence, reasoning, _raw} or {error, violates=None}.
    """
    user_msg = f"<proposal>\n{text}\n</proposal>"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=300,
                temperature=0,                      # deterministic classification
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content
            result = sl.parse_llm_json(raw)          # reuse the tested parser
            result["_raw"] = raw
            return result
        except Exception as e:                       # groq raises various error types
            etype = type(e).__name__
            if "RateLimit" in etype:
                time.sleep(2 ** attempt)
                continue
            if attempt == max_retries - 1:
                return {"error": f"{etype}: {e}", "violates": None}
            time.sleep(1)
    return {"error": "exhausted retries", "violates": None}


def classify_batch(texts, client: groq.Groq, system_prompt: str,
                   model: str = MODEL, delay: float = 0.5) -> list[dict]:
    """Classify many proposals. delay=0.5s stays comfortably under the 30 req/min free limit."""
    return [_sleep_then(classify_with_groq(t, client, system_prompt, model), delay) for t in texts]


def _sleep_then(result, delay):
    time.sleep(delay)
    return result


def evaluate_injection_defense(client: groq.Groq, system_prompt: str,
                                model: str = MODEL) -> "pd.DataFrame":
    """Same injection battery as sentinel_llm, routed through Groq."""
    import pandas as pd
    rows = []
    for name, text in sl.INJECTION_ATTACKS:
        r = classify_with_groq(text, client, system_prompt, model)
        time.sleep(0.5)
        rows.append({
            "attack": name,
            "model_said_violates": r.get("violates"),
            "defended": r.get("violates") is True,
            "reasoning": r.get("reasoning", r.get("error", "")),
        })
    return pd.DataFrame(rows)
