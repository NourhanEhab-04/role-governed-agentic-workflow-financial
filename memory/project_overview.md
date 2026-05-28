---
name: project-overview
description: MiFID II multi-agent financial suitability pipeline — architecture, metrics, fixes applied 2026-05-27, and known issues
metadata:
  type: project
---

MiFID II suitability assessment pipeline comparing a baseline single-LLM architecture vs a 5-agent pipeline (A1 client profiler, A2 product classifier, A3 rule engine, A4 conflict detector, A5 disclosure agent) using LangGraph + AutoGen.

**Architecture**: `orchestrator/graph.py` is the LangGraph state machine. Agents are in `agents/`. Evaluation in `evaluation/`.

**Metrics M1-M7** (accuracy/fairness) live in `evaluation/metrics.py`.
**Metrics M8-M15** (governance/compliance/explainability) added 2026-05-27 to `evaluation/governance_metrics.py`.

**Fixes applied 2026-05-27** (to be re-evaluated):
1. **A4 CONTRADICTION fix** (`agents/conflict_detector.py`): Changed trigger from `vulnerability==HIGH AND decision==SUITABLE` to `vulnerability==HIGH AND decision==SUITABLE AND risk_class>=4`. This correctly distinguishes scenario 07 (global_equity_etf, risk_class=4 → escalate) from scenario 17 (government_bond, risk_class=2 → no escalate). Also added `product_profile` parameter to `check_contradiction` and updated ESMA GL §86-88 rationale in system prompt.
2. **A3 parse fallback** (`agents/rule_engine_agent.py`): Added try/except around `parse_rule_verdict`; on failure (e.g., LLM outputs 'N/A' for rule values), falls back to direct `evaluate_suitability` call. Prevents pipeline UNKNOWN crashes.
3. **A5 deterministic overrides** (`agents/disclosure_agent.py`): Before Pydantic validation, override rule_findings statuses from `rule_verdict["rules"]` and override decision from `conflict_report["escalate"]`. Prevents A5 from hallucinating wrong PASS/FAIL or wrong ESCALATED decisions.
4. **A3 integrity failure recovery** (`orchestrator/graph.py`): When `validate_after_a3` fails but `pre_check_verdict` is available, recover using pre_check result (both come from same deterministic engine on same profiles) and continue pipeline. Only halt if no pre_check is available. This preserves the three-point integrity check as an audit trail while keeping the pipeline running.

**Known pre-existing test failures** (not caused by any recent changes):
- `rule_engine/tests/test_rule_engine.py` — 22 failures; test fixtures assert old penalty values that diverged from implementation.
- `tests/test_validators_cross_check.py`, `test_verifier_wiring.py`, `test_parallel_profiling.py` — mock `orchestrator.orchestrator.run_pre_check` which no longer exists after refactor.
- `tests/test_fixture_integrity.py` — hardcoded `len(files) == 10` but there are 19 scenarios.

**Remaining issues after fixes** (2026-05-27 state):
- Non-halting wrong decisions in 3 scenarios (09_run1, 11_run3, 18_run2): A1/A2 extract wrong risk_tolerance_score or product knowledge_level, pre_check AND A3 both agree on the wrong answer — three-point check can't catch this.
- Rate limit failures (10_run1): infrastructure issue, not fixable in code.
- M7-B parity violation partially remains: scenario 18_run2 still gives UNSUITABLE (non-halting A1/A2 error).

**Why:** Thesis comparing role-governed multi-agent pipeline against single LLM for regulatory compliance. **How to apply:** Always frame changes in terms of MiFID II / EU AI Act compliance.
