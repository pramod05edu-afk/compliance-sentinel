"""Reusable cleaning functions extracted from the Week-1 pipeline (Steps 1-11).

Only logic lives here -- no file paths, no plotting -- so every function is
unit-testable and importable by both the notebook and the pytest suite.
"""
import re
import pandas as pd


def dedup_tickets(df: pd.DataFrame, key: str = "Issue key") -> pd.DataFrame:
    """Remove the 49x row duplication found in Week 1.

    Keeps the first copy of every ticket. Safe because the Week-1 variation
    check proved copies differ only in access-timestamp metadata that is
    dropped later anyway.
    """
    if key not in df.columns:
        raise KeyError(f"'{key}' column not found -- is this the raw export?")
    return df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)


_WIKI_PATTERNS = [
    (re.compile(r"\{code(?::[^}]*)?\}.*?\{code\}", re.DOTALL), " [code] "),
    (re.compile(r"\{noformat\}.*?\{noformat\}", re.DOTALL), " [code] "),
    (re.compile(r"\{quote\}|\{color(?::[^}]*)?\}|\{color\}"), " "),
    (re.compile(r"!\S+?(?:\|[^!]*)?!"), " [image] "),
    (re.compile(r"\[([^|\]]+)(?:\|[^\]]*)?\]"), r"\1"),
    (re.compile(r"h[1-6]\.\s*"), ""),
    (re.compile(r"\*([^*\n]+)\*"), r"\1"),
    (re.compile(r"\s+"), " "),
]


def strip_wiki_markup(text) -> str:
    """Strip Jira wiki markup so detectors see prose, not formatting noise."""
    if not isinstance(text, str):
        return ""
    for pattern, repl in _WIKI_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


def content_guard(text, min_chars: int = 30, max_chars: int = 5000):
    """Ingestion guard from Analysis 5 (descriptions ran 0 -> 86k chars).

    Returns (status, text) where status is one of
    'too_short' | 'truncated' | 'ok'. Truncation keeps the head of the text.
    """
    if not isinstance(text, str):
        text = ""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return "too_short", stripped
    if len(stripped) > max_chars:
        return "truncated", stripped[:max_chars]
    return "ok", stripped
