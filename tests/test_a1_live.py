"""
Live integration tests for run_client_profiler.
These tests make real LLM calls — run selectively, not in CI.

    python -m pytest agents/tests/test_a1_live.py -v

Note: uses the free Gemini tier which has a low RPM limit.
Tests are run sequentially with a 5-second delay between calls
to avoid hitting the rate limit.
"""
import asyncio
import time
import pytest

from config.llm_config import get_model_client
from agents.client_profiler import run_client_profiler
from schemas.client_profile import REQUIRED_CLIENT_KEYS


LIVE_TEST_INPUTS = [
    # 1 — clean structured JSON input
    '{"age": 34, "knowledge": "basic", "risk": "conservative", '
    '"horizon": "3 years", "liquid": 8000, "income": 42000, "invest": 5000}',

    # 2 — natural language, all info present
    "I am 45 years old, work as a teacher earning 35000 EUR a year. "
    "I have 15000 EUR in savings. I want to invest 5000 EUR for 5 years. "
    "I consider myself a cautious investor and I cannot afford to lose everything.",

    # 3 — high risk, sophisticated investor
    "Professional trader, 38 years old, annual income 120000 EUR, "
    "liquid assets 80000 EUR. Looking to invest 20000 EUR. "
    "Very high risk tolerance, 10-year horizon, can afford total loss.",

    # 4 — elderly / vulnerability flag expected
    "I am 74 years old, retired, pension income 18000 EUR per year. "
    "I have 22000 EUR in savings and want to invest 10000 EUR. "
    "I am quite worried about losing money.",

    # 5 — missing investment horizon → needs_clarification expected
    "I have 30000 EUR in savings, earn 55000 EUR per year. "
    "I want to invest 8000 EUR. Moderate risk appetite.",

    # 6 — ambiguous risk description
    "I like to take some risks but not too many. "
    "I earn 60000 EUR, have 25000 EUR liquid. "
    "Invest 7000 EUR for roughly 4 years.",

    # 7 — very low income, high investment amount → vulnerability expected
    "Freelancer, irregular income around 15000 EUR per year. "
    "Savings of 6000 EUR. Want to put in 5000 EUR for 2 years.",

    # 8 — all fields explicitly stated in mixed format
    "Age 29. Knowledge level: advanced (I trade derivatives). "
    "Risk score: 8/10. Horizon: 7 years. "
    "Liquid assets: EUR 40,000. Salary: EUR 75,000/year. "
    "Investment: EUR 10,000. Can afford total loss: yes.",

    # 9 — completely missing financial situation → needs_clarification expected
    "I want to invest in something safe for my retirement.",

    # 10 — contradictory signals (claims high risk but fears losses)
    "I consider myself an aggressive investor with high risk tolerance. "
    "But I really cannot afford to lose this money — it is my emergency fund. "
    "Income 48000 EUR, liquid 9000 EUR, invest 8000 EUR, 3-year horizon.",
]


@pytest.mark.parametrize("raw_input", LIVE_TEST_INPUTS)
def test_output_schema_valid(raw_input):
    time.sleep(5)  # respect free-tier RPM limit between parametrized calls
    client = get_model_client()
    try:
        result = asyncio.run(run_client_profiler(raw_input, client))
        # If we get a result, validate schema
        assert set(result.keys()) >= REQUIRED_CLIENT_KEYS
        assert isinstance(result["risk_tolerance_score"], int)
        assert isinstance(result["investment_horizon"], int)
        assert isinstance(result["liquid_assets"], float)
        assert isinstance(result["income"], float)
        assert isinstance(result["investment_amount"], float)
        assert isinstance(result["can_afford_total_loss"], bool)
        assert result["financial_knowledge"] in {"none", "basic", "moderate", "advanced"}
        assert result["financial_vulnerability"] in {"LOW", "MEDIUM", "HIGH"}
    except ValueError as e:
        # needs_clarification or parse failure is acceptable for
        # inputs 5 and 9 which intentionally have missing fields
        assert "missing" in str(e).lower() or "clarification" in str(e).lower(), \
            f"Unexpected ValueError for input: {raw_input}\nError: {e}"
