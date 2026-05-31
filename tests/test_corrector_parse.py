# tests/test_corrector_parse.py
"""
Unit tests for the corrector agent helpers in agents/verifier_agent.py.

Verifies:
  1. _build_corrector_message produces the right sections
  2. run_corrector_on_a1 passes its output through parse_client_profile
  3. run_corrector_on_a2 passes its output through parse_product_profile
  4. A corrected output that fails schema validation raises ValueError
  5. Only truly failed fields appear in the corrector message
No live LLM calls — agent.on_messages is mocked.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.verifier_agent import _build_corrector_message


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _good_client():
    return {
        "age": 40,
        "financial_knowledge": "moderate",
        "risk_tolerance_score": 5,
        "investment_horizon": 5,
        "liquid_assets": 20000.0,
        "income": 50000.0,
        "investment_amount": 5000.0,
        "can_afford_total_loss": False,
        "financial_vulnerability": "LOW",
    }


def _good_product():
    return {
        "product_name": "Global Equity ETF",
        "risk_class": 4,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 3,
        "potential_loss": "partial",
        "leverage": False,
    }


def _verification_two_failed():
    """Verification result with two failed fields."""
    return {
        "passed": False,
        "confidence": 0.5,
        "issues": ["risk_tolerance_score is wrong", "income looks wrong"],
        "field_checks": {
            "financial_knowledge": {"supported": True,  "reason": "Clear from input."},
            "risk_tolerance_score": {"supported": False, "reason": "Client said 3 but 8 returned."},
            "income":               {"supported": False, "reason": "Monthly figure not annualised."},
            "investment_horizon":   {"supported": True,  "reason": "5 years stated."},
        },
    }


def _verification_all_passed():
    return {
        "passed": True,
        "confidence": 0.95,
        "issues": [],
        "field_checks": {
            "financial_knowledge": {"supported": True, "reason": "ok"},
        },
    }


# ── _build_corrector_message ──────────────────────────────────────────────────

class TestBuildCorrectorMessage:

    def test_contains_original_input(self):
        msg = _build_corrector_message("client text here", _good_client(), _verification_two_failed())
        assert "client text here" in msg

    def test_contains_previous_output(self):
        msg = _build_corrector_message("input", _good_client(), _verification_two_failed())
        assert "PREVIOUS_OUTPUT" in msg
        assert "financial_knowledge" in msg

    def test_contains_failed_fields_section(self):
        msg = _build_corrector_message("input", _good_client(), _verification_two_failed())
        assert "FAILED_FIELDS" in msg

    def test_only_failed_fields_in_failed_section(self):
        msg = _build_corrector_message("input", _good_client(), _verification_two_failed())
        # Parse what's between FAILED_FIELDS and VERIFIER_ISSUES
        failed_section = msg.split("FAILED_FIELDS:")[1].split("VERIFIER_ISSUES:")[0]
        failed_data = json.loads(failed_section.strip())
        assert set(failed_data.keys()) == {"risk_tolerance_score", "income"}
        # Passing field must NOT appear
        assert "financial_knowledge" not in failed_data

    def test_verifier_issues_appear(self):
        msg = _build_corrector_message("input", _good_client(), _verification_two_failed())
        assert "risk_tolerance_score is wrong" in msg
        assert "income looks wrong" in msg

    def test_no_failed_fields_produces_empty_failed_section(self):
        msg = _build_corrector_message("input", _good_client(), _verification_all_passed())
        failed_section = msg.split("FAILED_FIELDS:")[1].split("VERIFIER_ISSUES:")[0]
        failed_data = json.loads(failed_section.strip())
        assert failed_data == {}

    def test_missing_field_checks_key_handled(self):
        verification = {"passed": False, "confidence": 0.4, "issues": ["x"], "field_checks": {}}
        msg = _build_corrector_message("input", _good_client(), verification)
        assert "FAILED_FIELDS" in msg  # should not raise


# ── run_corrector_on_a1 ───────────────────────────────────────────────────────

class TestRunCorrectorA1:

    @pytest.mark.asyncio
    async def test_valid_corrected_output_returned(self):
        """Corrector returns valid JSON → parsed and returned as dict."""
        from agents.verifier_agent import run_corrector_on_a1

        corrected = dict(_good_client())
        corrected["risk_tolerance_score"] = 3  # the fix

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(corrected)
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent):
            result = await run_corrector_on_a1(
                "original text",
                _good_client(),
                _verification_two_failed(),
                model_client=MagicMock(),
            )

        assert result["risk_tolerance_score"] == 3
        assert result["financial_knowledge"] == "moderate"

    @pytest.mark.asyncio
    async def test_invalid_corrected_output_raises(self):
        """Corrector returns JSON missing required keys → ValueError from parse_client_profile."""
        from agents.verifier_agent import run_corrector_on_a1

        bad_output = {"financial_knowledge": "moderate"}  # missing 7 keys

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(bad_output)
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent):
            with pytest.raises(ValueError):
                await run_corrector_on_a1(
                    "original",
                    _good_client(),
                    _verification_two_failed(),
                    model_client=MagicMock(),
                )

    @pytest.mark.asyncio
    async def test_invalid_enum_raises(self):
        """Corrector returns invalid enum value → ValueError."""
        from agents.verifier_agent import run_corrector_on_a1

        bad_output = dict(_good_client())
        bad_output["financial_knowledge"] = "expert"  # invalid enum

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(bad_output)
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent):
            with pytest.raises(ValueError, match="Invalid financial_knowledge"):
                await run_corrector_on_a1(
                    "original",
                    _good_client(),
                    _verification_two_failed(),
                    model_client=MagicMock(),
                )

    @pytest.mark.asyncio
    async def test_corrector_uses_corrector_system_prompt(self):
        """The agent must be built with CORRECTOR_SYSTEM_PROMPT_A1, not verifier prompt."""
        from agents.verifier_agent import run_corrector_on_a1, CORRECTOR_SYSTEM_PROMPT_A1

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(_good_client())
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent) as MockAgent:
            await run_corrector_on_a1("input", _good_client(), _verification_two_failed(), MagicMock())

        _, kwargs = MockAgent.call_args
        assert kwargs["system_message"] == CORRECTOR_SYSTEM_PROMPT_A1

    @pytest.mark.asyncio
    async def test_corrector_agent_name_is_corrector(self):
        """Agent name must be CorrectorA1."""
        from agents.verifier_agent import run_corrector_on_a1

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(_good_client())
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent) as MockAgent:
            await run_corrector_on_a1("input", _good_client(), _verification_two_failed(), MagicMock())

        _, kwargs = MockAgent.call_args
        assert kwargs["name"] == "CorrectorA1"


# ── run_corrector_on_a2 ───────────────────────────────────────────────────────

class TestRunCorrectorA2:

    @pytest.mark.asyncio
    async def test_valid_corrected_output_returned(self):
        from agents.verifier_agent import run_corrector_on_a2

        corrected = dict(_good_product())
        corrected["risk_class"] = 5  # the fix

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(corrected)
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent):
            result = await run_corrector_on_a2(
                "product text",
                _good_product(),
                _verification_two_failed(),
                model_client=MagicMock(),
            )

        assert result["risk_class"] == 5

    @pytest.mark.asyncio
    async def test_invalid_corrected_output_raises(self):
        from agents.verifier_agent import run_corrector_on_a2

        bad_output = {"product_name": "ETF"}  # missing keys

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(bad_output)
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent):
            with pytest.raises(ValueError):
                await run_corrector_on_a2(
                    "original",
                    _good_product(),
                    _verification_two_failed(),
                    model_client=MagicMock(),
                )

    @pytest.mark.asyncio
    async def test_corrector_uses_corrector_system_prompt(self):
        from agents.verifier_agent import run_corrector_on_a2, CORRECTOR_SYSTEM_PROMPT_A2

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(_good_product())
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent) as MockAgent:
            await run_corrector_on_a2("input", _good_product(), _verification_two_failed(), MagicMock())

        _, kwargs = MockAgent.call_args
        assert kwargs["system_message"] == CORRECTOR_SYSTEM_PROMPT_A2

    @pytest.mark.asyncio
    async def test_corrector_agent_name_is_corrector(self):
        from agents.verifier_agent import run_corrector_on_a2

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(_good_product())
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent) as MockAgent:
            await run_corrector_on_a2("input", _good_product(), _verification_two_failed(), MagicMock())

        _, kwargs = MockAgent.call_args
        assert kwargs["name"] == "CorrectorA2"

    @pytest.mark.asyncio
    async def test_invalid_risk_class_raises(self):
        from agents.verifier_agent import run_corrector_on_a2

        bad_output = dict(_good_product())
        bad_output["risk_class"] = 99  # out of range

        mock_response = MagicMock()
        mock_response.chat_message.content = json.dumps(bad_output)
        mock_agent = MagicMock()
        mock_agent.on_messages = AsyncMock(return_value=mock_response)

        with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent):
            with pytest.raises(ValueError, match="risk_class"):
                await run_corrector_on_a2(
                    "original",
                    _good_product(),
                    _verification_two_failed(),
                    model_client=MagicMock(),
                )
