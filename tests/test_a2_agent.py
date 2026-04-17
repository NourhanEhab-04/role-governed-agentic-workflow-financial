import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch
import autogen_agentchat.agents as agents_module
from agents.product_classifier import run_product_classifier

MOCK_ETF_RESPONSE = """{
    "product_name": "Vanguard FTSE All-World ETF",
    "risk_class": 4,
    "complexity_tier": "NON-COMPLEX",
    "requires_knowledge_level": "basic",
    "minimum_horizon": 3,
    "potential_loss": "partial",
    "leverage": false
}"""

MOCK_LEVERAGED_RESPONSE = """{
    "product_name": "ProShares UltraPro QQQ 3x Leveraged ETF",
    "risk_class": 7,
    "complexity_tier": "COMPLEX",
    "requires_knowledge_level": "advanced",
    "minimum_horizon": 1,
    "potential_loss": "total",
    "leverage": true
}"""

MOCK_UNCLEAR_RESPONSE = """{
    "status": "needs_clarification",
    "missing": ["product_type"]
}"""


def make_mock_on_messages(response_text: str):
    async def mock_on_messages(self, messages, cancellation_token):
        msg = MagicMock()
        msg.content = response_text
        resp = MagicMock()
        resp.chat_message = msg
        return resp
    return mock_on_messages


def test_etf_returns_valid_dict():
    with patch.object(agents_module.AssistantAgent, "on_messages",
                      new=make_mock_on_messages(MOCK_ETF_RESPONSE)):
        result = asyncio.run(
            run_product_classifier("Plain vanilla global equity ETF", MagicMock())
        )
    assert result["risk_class"] == 4
    assert result["complexity_tier"] == "NON-COMPLEX"
    assert result["leverage"] is False


def test_leveraged_etf_returns_valid_dict():
    with patch.object(agents_module.AssistantAgent, "on_messages",
                      new=make_mock_on_messages(MOCK_LEVERAGED_RESPONSE)):
        result = asyncio.run(
            run_product_classifier("3x leveraged NASDAQ ETF", MagicMock())
        )
    assert result["risk_class"] == 7
    assert result["complexity_tier"] == "COMPLEX"
    assert result["leverage"] is True
    assert result["potential_loss"] == "total"


def test_unclear_product_raises_value_error():
    with patch.object(agents_module.AssistantAgent, "on_messages",
                      new=make_mock_on_messages(MOCK_UNCLEAR_RESPONSE)):
        with pytest.raises(ValueError):
            asyncio.run(
                run_product_classifier("something financial", MagicMock())
            )
