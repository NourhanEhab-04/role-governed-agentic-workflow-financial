# tests/test_selector.py

from types import SimpleNamespace
from orchestrator.selector import make_selector


def sp(name):
    """Make a fake last_speaker with just a .name attribute."""
    return SimpleNamespace(name=name)


def valid_state():
    """A pipeline_state that passes all five validators."""
    return {
        "client_profile": {
            "financial_knowledge": "moderate",
            "risk_tolerance_score": 5,
            "investment_horizon": 5,
            "liquid_assets": 10000.0,
            "income": 50000.0,
            "investment_amount": 5000.0,
            "can_afford_total_loss": True,
            "financial_vulnerability": "LOW",
        },
        "product_profile": {
            "product_name": "Example Fund",
            "risk_class": 4,
            "complexity_tier": "NON-COMPLEX",
            "requires_knowledge_level": "basic",
            "minimum_horizon": 3,
            "potential_loss": "partial",
            "leverage": False,
        },
        "rule_verdict": {
            "rules": {r: "PASS" for r in ["R1","R2","R3","R4","R5","R6","R7"]},
            "score": 100,
            "decision": "SUITABLE",
            "failed_rules": [],
        },
        "conflict_report": {
            "flags": [],
            "escalate": False,
            "summary": "ok",
        },
        "suitability_report": {
            "decision": "SUITABLE",
            "summary": "ok",
            "rule_findings": [
                {"rule_id": r, "status": "PASS", "explanation": "ok"}
                for r in ["R1","R2","R3","R4","R5","R6","R7"]
            ],
            "flags_addressed": [],
            "regulatory_basis": "Article 25(2)",
            "client_facing_summary": "Suitable.",
        },
    }


# ── 1. First call with no speaker → A1 ──────────────

def test_first_call_routes_to_a1():
    fn = make_selector({}, {})
    assert fn(sp(None), []) == "client_profiler"


# ── 2. Full clean forward path ───────────────────────

def test_clean_forward_path():
    state = valid_state()
    fn = make_selector(state, {})
    assert fn(sp("client_profiler"),    []) == "product_classifier"
    assert fn(sp("product_classifier"), []) == "rule_engine_agent"
    assert fn(sp("rule_engine_agent"),  []) == "conflict_detector"
    assert fn(sp("conflict_detector"),  []) == "disclosure_agent"
    assert fn(sp("disclosure_agent"),   []) == "TERMINATE"


# ── 3. First retry on bad A1 ────────────────────────

def test_first_retry_on_bad_a1():
    state = {}          # no client_profile → validation fails
    retries = {}
    fn = make_selector(state, retries)
    result = fn(sp("client_profiler"), [])
    assert result == "client_profiler"
    assert retries["A1"] == 1


# ── 4. Halt after second bad A1 ─────────────────────

def test_halt_after_second_bad_a1():
    state = {}
    retries = {"A1": 1}  # already used the one retry
    fn = make_selector(state, retries)
    result = fn(sp("client_profiler"), [])
    assert result == "PIPELINE_HALT"
    assert state["halt"] is True
    assert "client_profiler" in state["halt_reason"]


# ── 5. First retry on bad A3 ────────────────────────

def test_first_retry_on_bad_a3():
    state = valid_state()
    state["rule_verdict"]["decision"] = "INVALID"
    retries = {}
    fn = make_selector(state, retries)
    result = fn(sp("rule_engine_agent"), [])
    assert result == "rule_engine_agent"
    assert retries["A3"] == 1


# ── 6. Halt after second bad A3 ─────────────────────

def test_halt_after_second_bad_a3():
    state = valid_state()
    state["rule_verdict"]["decision"] = "INVALID"
    retries = {"A3": 1}
    fn = make_selector(state, retries)
    result = fn(sp("rule_engine_agent"), [])
    assert result == "PIPELINE_HALT"
    assert state["halt"] is True


# ── 7. Escalation routes to A5, sets escalated flag ─

def test_escalation_routes_to_disclosure_and_sets_flag():
    state = valid_state()
    state["conflict_report"]["escalate"] = True
    fn = make_selector(state, {})
    result = fn(sp("conflict_detector"), [])
    assert result == "disclosure_agent"
    assert state.get("escalated") is True
    assert state.get("halt_reason") is not None


# ── 8. No escalation → normal route to A5 ───────────

def test_no_escalation_routes_normally():
    state = valid_state()
    state["conflict_report"]["escalate"] = False
    fn = make_selector(state, {})
    result = fn(sp("conflict_detector"), [])
    assert result == "disclosure_agent"
    assert state.get("escalated") is not True


# ── 9. A5 TERMINATE after valid output ──────────────

def test_a5_terminates():
    state = valid_state()
    fn = make_selector(state, {})
    assert fn(sp("disclosure_agent"), []) == "TERMINATE"


# ── 10. Retry counter is per-stage, not global ───────

def test_retry_counts_are_per_stage():
    state = valid_state()
    state["rule_verdict"]["decision"] = "INVALID"   # A3 bad
    retries = {"A1": 1}                              # A1 already retried
    fn = make_selector(state, retries)
    fn(sp("rule_engine_agent"), [])
    assert retries["A1"] == 1   # unchanged
    assert retries["A3"] == 1   # incremented independently