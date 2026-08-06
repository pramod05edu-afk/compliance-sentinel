import pandas as pd
import pytest

from src.cleaning import dedup_tickets, strip_wiki_markup, content_guard


def _stacked_export(n_tickets=5, copies=49):
    """Miniature of the real defect: every ticket repeated `copies` times."""
    base = pd.DataFrame({"Issue key": [f"SRCTREE-{i}" for i in range(n_tickets)],
                         "summary": [f"bug {i}" for i in range(n_tickets)]})
    return pd.concat([base] * copies, ignore_index=True)


def test_dedup_removes_the_49x_stack():
    raw = _stacked_export()
    assert len(raw) == 5 * 49                      # the defect
    clean = dedup_tickets(raw)
    assert len(clean) == 5                         # the fix
    assert clean["Issue key"].is_unique


def test_dedup_keeps_first_copy_and_resets_index():
    raw = _stacked_export(n_tickets=2, copies=3)
    raw.loc[2, "summary"] = "changed in a later copy"   # 2nd copy of ticket 0
    clean = dedup_tickets(raw)
    assert list(clean.index) == [0, 1]
    assert clean.loc[0, "summary"] == "bug 0"      # first copy won


def test_dedup_raises_on_wrong_frame():
    with pytest.raises(KeyError):
        dedup_tickets(pd.DataFrame({"not_a_key": [1]}))


def test_strip_wiki_markup_removes_code_blocks_and_formatting():
    raw = "h2. Title *bold* {code:java}secret(){code} see [docs|http://x] !img.png!"
    out = strip_wiki_markup(raw)
    assert "{code" not in out and "h2." not in out and "*bold*" not in out
    assert "secret()" not in out                   # code content replaced
    assert "docs" in out                           # link text kept


def test_strip_wiki_markup_handles_non_strings():
    assert strip_wiki_markup(None) == ""
    assert strip_wiki_markup(float("nan")) == ""


def test_content_guard_all_three_statuses():
    assert content_guard("hi")[0] == "too_short"
    status, text = content_guard("x" * 6000)
    assert status == "truncated" and len(text) == 5000
    assert content_guard("a perfectly reasonable description here")[0] == "ok"


def test_content_guard_non_string_is_too_short():
    assert content_guard(None)[0] == "too_short"
