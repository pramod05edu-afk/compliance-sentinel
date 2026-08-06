import pytest

from src.detectors import FIX_TEMPLATES, suggest_fix, write_back


def test_suggest_fix_template_covers_every_policy():
    for policy in FIX_TEMPLATES:
        fix = suggest_fix(policy, "any proposal text", use_llm=False)
        assert fix["engine"] == "template"
        assert len(fix["suggestion"]) > 10


def test_suggest_fix_unknown_policy_has_a_fallback():
    fix = suggest_fix("P99_unknown", "any text", use_llm=False)
    assert fix["engine"] == "template"
    assert "manually" in fix["suggestion"]


def test_suggest_fix_never_calls_llm_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fix = suggest_fix("P3_consent", "any text", use_llm=True, api_key=None)
    assert fix["engine"] == "template"           # graceful fallback, no crash


def test_write_back_blocks_without_approval():
    result = write_back("TICKET-1", "some fix", approved=False)
    assert result["status"] == "blocked"
    assert "not approved" in result["reason"]


def test_write_back_writes_when_approved():
    result = write_back("TICKET-1", "some fix", approved=True)
    assert result["status"] == "written"
    assert "TICKET-1" in result["message"]
    assert "some fix" in result["message"]


def test_write_back_never_writes_on_rejection_even_with_a_real_ticket():
    # Guards the Art. 14 human-oversight gate: no code path may flip this.
    for approved in (False, 0, None, ""):
        result = write_back("TICKET-42", "irrelevant", approved=approved)
        assert result["status"] == "blocked"
