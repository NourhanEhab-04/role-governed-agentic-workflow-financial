"""
Live integration tests for run_product_classifier.
These tests make real LLM calls — run selectively, not in CI.

    python -m pytest agents/tests/test_a2_live.py -v
"""
import pytest
import asyncio
from dotenv import load_dotenv
load_dotenv()

from config.llm_config import get_model_client
from agents.product_classifier import run_product_classifier
from schemas.product_profile import REQUIRED_PRODUCT_KEYS

LIVE_PRODUCTS = [
    # 1 — plain vanilla broad market ETF
    (
        "iShares Core MSCI World ETF — tracks the MSCI World Index, "
        "invests in large and mid-cap equities across 23 developed markets, "
        "no leverage, listed on Euronext, UCITS compliant.",
        {
            "expected_complexity": "NON-COMPLEX",
            "expected_leverage": False,
            "expected_risk_class_range": (3, 5),
            "expected_potential_loss": "partial",
        }
    ),

    # 2 — government bond
    (
        "German Federal Government Bond (Bund), 10-year maturity, "
        "fixed coupon 2.5%, AAA rated by S&P and Moody's, "
        "issued in EUR, listed on Deutsche Borse.",
        {
            "expected_complexity": "NON-COMPLEX",
            "expected_leverage": False,
            "expected_risk_class_range": (1, 3),
            "expected_potential_loss": "partial",
        }
    ),

    # 3 — equity derivative (call option)
    (
        "European call option on Apple Inc. (AAPL) stock, "
        "strike price $195, expiry 3 months, traded on Eurex, "
        "buyer pays premium upfront, no margin required beyond premium.",
        {
            "expected_complexity": "COMPLEX",
            "expected_leverage": False,   # buying options: limited to premium paid
            "expected_risk_class_range": (5, 7),
            "expected_potential_loss": "total",
        }
    ),

    # 4 — leveraged ETF 3x
    (
        "ProShares UltraPro QQQ ETF — seeks daily investment results "
        "of 3x the daily performance of the NASDAQ-100 Index. "
        "Uses swap agreements and futures contracts to achieve leverage. "
        "Intended for short-term trading, not long-term holding.",
        {
            "expected_complexity": "COMPLEX",
            "expected_leverage": True,
            "expected_risk_class_range": (6, 7),
            "expected_potential_loss": "total",
        }
    ),

    # 5 — money market fund
    (
        "Fidelity Money Market Fund — invests in short-term, "
        "high-quality EUR-denominated money market instruments including "
        "treasury bills and commercial paper. Daily liquidity. "
        "Capital preservation objective. UCITS compliant.",
        {
            "expected_complexity": "NON-COMPLEX",
            "expected_leverage": False,
            "expected_risk_class_range": (1, 2),
            "expected_potential_loss": "partial",
        }
    ),
]


@pytest.mark.parametrize("raw_input,expectations", LIVE_PRODUCTS)
def test_product_schema_valid_and_classification_correct(raw_input, expectations):
    client = get_model_client()
    result = asyncio.run(run_product_classifier(raw_input, client))

    # Schema check — all required keys present
    assert set(result.keys()) >= REQUIRED_PRODUCT_KEYS

    # Type checks
    assert isinstance(result["risk_class"], int)
    assert 1 <= result["risk_class"] <= 7
    assert isinstance(result["minimum_horizon"], int)
    assert result["minimum_horizon"] >= 1
    assert isinstance(result["leverage"], bool)
    assert result["complexity_tier"] in {"NON-COMPLEX", "COMPLEX"}
    assert result["requires_knowledge_level"] in {"none", "basic", "moderate", "advanced"}
    assert result["potential_loss"] in {"partial", "total"}

    # Classification correctness checks
    assert result["complexity_tier"] == expectations["expected_complexity"], (
        f"Expected complexity {expectations['expected_complexity']}, "
        f"got {result['complexity_tier']}"
    )

    assert result["leverage"] == expectations["expected_leverage"], (
        f"Expected leverage {expectations['expected_leverage']}, "
        f"got {result['leverage']}"
    )

    low, high = expectations["expected_risk_class_range"]
    assert low <= result["risk_class"] <= high, (
        f"Expected risk_class between {low} and {high}, "
        f"got {result['risk_class']}"
    )

    assert result["potential_loss"] == expectations["expected_potential_loss"], (
        f"Expected potential_loss {expectations['expected_potential_loss']}, "
        f"got {result['potential_loss']}"
    )
