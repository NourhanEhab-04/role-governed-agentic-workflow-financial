# tests/test_a3_agent.py
"""
Tests for run_rule_engine_agent (A3).

A3 is fully deterministic — it calls evaluate_suitability directly with no LLM.
These tests verify the output format, correctness, and error handling.
"""
import pytest
import asyncio
from unittest.mock import MagicMock

from agents.rule_engine_agent import run_rule_engine_agent
from rule_engine.rule_engine import evaluate_suitability

BASE_CLIENT = {
    "financial_knowledge": "moderate",
    "risk_tolerance_score": 5,
    "investment_horizon": 5,
    "liquid_assets": 20000.0,
    "income": 60000.0,
    "investment_amount": 3000.0,
    "can_afford_total_loss": True,
    "financial_vulnerability": "LOW",
}

BASE_PRODUCT = {
    "risk_class": 4,
    "complexity_tier": "NON-COMPLEX",
    "requires_knowledge_level": "basic",
    "minimum_horizon": 2,
    "potential_loss": "partial",
    "leverage": False,
}


def test_valid_verdict_returned():
    """A3 returns a valid rule verdict for known-good inputs."""
    result = asyncio.run(run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, MagicMock()))
    assert result["score"] == 100
    assert result["decision"] == "SUITABLE"
    assert all(v == "PASS" for v in result["rules"].values())


def test_output_uses_short_rule_ids():
    """Rules dict must use short IDs R1..R7, not long-form IDs."""
    result = asyncio.run(run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, MagicMock()))
    assert set(result["rules"].keys()) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}


def test_tool_output_matches_direct_rule_engine():
    """A3 output score/decision must match evaluate_suitability directly."""
    direct = evaluate_suitability(BASE_CLIENT, BASE_PRODUCT)
    result = asyncio.run(run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, MagicMock()))

    assert result["score"] == direct["score"]
    assert result["decision"] == direct["decision"]
    assert all(v in {"PASS", "FAIL"} for v in result["rules"].values())


def test_a3_is_deterministic():
    """Same inputs always produce identical output (no LLM randomness)."""
    results = [
        asyncio.run(run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, MagicMock()))
        for _ in range(5)
    ]
    assert all(r == results[0] for r in results)


def test_unsuitable_verdict_returned_correctly():
    """Clearly unsuitable profile surfaces UNSUITABLE with score < 40."""
    bad_client = {**BASE_CLIENT,
                  "can_afford_total_loss": False,
                  "risk_tolerance_score": 2,
                  "financial_knowledge": "none"}
    bad_product = {**BASE_PRODUCT,
                   "potential_loss": "total",
                   "risk_class": 6,
                   "requires_knowledge_level": "advanced",
                   "leverage": True}
    result = asyncio.run(run_rule_engine_agent(bad_client, bad_product, MagicMock()))
    assert result["decision"] == "UNSUITABLE"
    assert result["score"] < 40
    # At least one FAIL rule should be present
    assert any(v == "FAIL" for v in result["rules"].values())


def test_model_client_argument_is_ignored():
    """model_client is accepted for API compatibility but not used by A3."""
    result1 = asyncio.run(run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, MagicMock()))
    result2 = asyncio.run(run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, None))
    assert result1 == result2
