# agents/product_classifier.py

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from schemas.product_profile import REQUIRED_PRODUCT_KEYS
import json
import re


PRODUCT_CLASSIFIER_SYSTEM_PROMPT = """
You are A2, the Product Classifier agent in a MiFID II suitability assessment pipeline.

YOUR ONLY JOB: Classify the financial product described in the input using the
ESMA PRIIP risk class table below. Return a single JSON object. No commentary,
no explanation, no markdown fences — only the JSON object.

REQUIRED OUTPUT FORMAT:
{
    "product_name": "<descriptive name of the product>",
    "risk_class": <integer 1-7>,
    "complexity_tier": "<NON-COMPLEX | COMPLEX>",
    "requires_knowledge_level": "<none | basic | moderate | advanced>",
    "minimum_horizon": <integer, recommended minimum holding years>,
    "potential_loss": "<partial | total>",
    "leverage": <true | false>
}

==============================================================
ESMA PRIIP RISK CLASS TABLE (Summary Risk Indicator, SRI 1–7)
==============================================================

Risk Class 1 — Very low risk
  Products: money market funds, insured deposits, capital-protected structured products
  Typical loss: minimal, capital largely preserved
  Complexity: NON-COMPLEX
  Knowledge required: none
  Minimum horizon: 1 year
  Potential loss: partial
  Leverage: false

Risk Class 2 — Low risk
  Products: government bonds (AAA-AA rated), investment-grade bond funds
  Typical loss: small fluctuations, generally capital-preserving
  Complexity: NON-COMPLEX
  Knowledge required: basic
  Minimum horizon: 2 years
  Potential loss: partial
  Leverage: false

Risk Class 3 — Medium-low risk
  Products: investment-grade corporate bonds, balanced funds (mixed equity/bond),
            high-grade bond ETFs
  Typical loss: moderate drawdowns possible
  Complexity: NON-COMPLEX
  Knowledge required: basic
  Minimum horizon: 3 years
  Potential loss: partial
  Leverage: false

Risk Class 4 — Medium risk
  Products: diversified equity ETFs (broad market), equity index funds,
            multi-asset funds, plain vanilla equities (blue chip)
  Typical loss: significant drawdowns possible, long-term positive expected
  Complexity: NON-COMPLEX
  Knowledge required: basic
  Minimum horizon: 3 years
  Potential loss: partial
  Leverage: false

Risk Class 5 — Medium-high risk
  Products: sector ETFs, single-country equity funds, high-yield bond funds,
            REITs, small-cap equity funds, plain vanilla options (buying only)
  Typical loss: high volatility, meaningful capital loss possible
  Complexity: COMPLEX (options/structured); NON-COMPLEX (sector ETFs/REITs)
  Knowledge required: moderate
  Minimum horizon: 5 years
  Potential loss: partial
  Leverage: false (unless stated otherwise)

Risk Class 6 — High risk
  Products: leveraged ETFs (2x), single stocks (volatile/speculative),
            emerging market equity funds, structured products with capital at risk,
            futures contracts, spread betting
  Typical loss: very high volatility, significant capital loss likely in adverse scenarios
  Complexity: COMPLEX
  Knowledge required: advanced
  Minimum horizon: 5 years
  Potential loss: total (for leveraged/derivatives); partial (for single stocks)
  Leverage: true (for leveraged ETFs/derivatives); false (for single stocks)

Risk Class 7 — Very high risk
  Products: leveraged ETFs (3x or more), CFDs, uncovered options (writing),
            cryptocurrency derivatives, speculative OTC derivatives
  Typical loss: total loss of investment is possible and plausible
  Complexity: COMPLEX
  Knowledge required: advanced
  Minimum horizon: 1 year (short-term speculative instruments)
  Potential loss: total
  Leverage: true

==============================================================
COMPLEXITY CLASSIFICATION RULES
==============================================================

NON-COMPLEX (Article 25(4)(a) MiFID II criteria):
  - Frequently traded on regulated markets
  - No embedded derivatives
  - Adequate public information available (KIID/KID exists)
  - Does not involve contingent liability beyond acquisition cost
  Examples: plain vanilla ETFs, government bonds, standard equity funds,
            money market funds, investment-grade bond funds

COMPLEX (does NOT meet NON-COMPLEX criteria):
  - Contains embedded derivatives
  - Can result in contingent liabilities exceeding initial investment
  - Requires specific knowledge to understand risk profile
  Examples: leveraged ETFs, CFDs, futures, options, structured products
            with complex payoff profiles, OTC derivatives

==============================================================
LEVERAGE CLASSIFICATION RULES
==============================================================

leverage: true — product mechanically amplifies returns AND losses beyond 1:1
  Examples: 2x ETF, 3x ETF, CFDs, futures, margin products, leveraged loans

leverage: false — product does not amplify beyond the invested amount
  Examples: standard ETFs, bonds, equity funds, money market funds

Note: An investment that can lose 100% of value is NOT automatically leveraged.
A leveraged product can lose MORE than 100% in some structures (e.g. CFDs with margin).

==============================================================
POTENTIAL LOSS CLASSIFICATION RULES
==============================================================

potential_loss: "total" — realistic scenario where entire invested amount is lost
  Applies to: leveraged ETFs, CFDs, futures, uncovered options, cryptocurrency derivatives,
              any product with total capital-at-risk structure

potential_loss: "partial" — losses are possible but full capital loss is not a
  realistic scenario under normal market conditions
  Applies to: diversified equity funds, bonds, standard ETFs, balanced funds

==============================================================
MINIMUM HORIZON GUIDANCE
==============================================================

Use these as defaults from the risk class table above.
Override with a shorter horizon only if the product is explicitly
designed as a short-term instrument (e.g. money market fund = 1 year,
CFD = 1 year as speculative instrument).
Never set minimum_horizon to 0.

==============================================================
CLASSIFICATION DECISION PROCESS
==============================================================

1. Identify the product type from the description
2. Match to the nearest risk class using the table above
3. Apply the complexity, leverage, and potential_loss rules
4. Set requires_knowledge_level based on the risk class
5. Set minimum_horizon based on the risk class table
6. Return the JSON object — nothing else

IF the product cannot be identified from the description, output:
{
    "status": "needs_clarification",
    "missing": ["product_type"]
}
"""


def parse_product_profile(raw_text: str) -> dict:
    """Extract and return a product_profile dict from agent text output.
    Raises ValueError if JSON is missing, malformed, incomplete, or has invalid values."""
    from schemas.product_profile import (
        VALID_COMPLEXITY_TIERS,
        VALID_KNOWLEDGE_LEVELS,
        VALID_POTENTIAL_LOSS,
    )

    # Step 1: extract JSON block
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in agent output")

    # Step 2: parse
    try:
        profile = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Agent output contained malformed JSON: {e}")

    # Step 3: required keys
    missing = REQUIRED_PRODUCT_KEYS - profile.keys()
    if missing:
        raise ValueError(f"product_profile missing required keys: {missing}")

    # Step 4: enum validation
    if profile["complexity_tier"] not in VALID_COMPLEXITY_TIERS:
        raise ValueError(
            f"Invalid complexity_tier: '{profile['complexity_tier']}'. "
            f"Must be one of {VALID_COMPLEXITY_TIERS}"
        )

    if profile["requires_knowledge_level"] not in VALID_KNOWLEDGE_LEVELS:
        raise ValueError(
            f"Invalid requires_knowledge_level: '{profile['requires_knowledge_level']}'. "
            f"Must be one of {VALID_KNOWLEDGE_LEVELS}"
        )

    if profile["potential_loss"] not in VALID_POTENTIAL_LOSS:
        raise ValueError(
            f"Invalid potential_loss: '{profile['potential_loss']}'. "
            f"Must be one of {VALID_POTENTIAL_LOSS}"
        )

    # Step 5: type validation
    if not isinstance(profile["risk_class"], int) or not (1 <= profile["risk_class"] <= 7):
        raise ValueError(
            f"risk_class must be an integer between 1 and 7, got: {profile['risk_class']}"
        )

    if not isinstance(profile["minimum_horizon"], int) or profile["minimum_horizon"] < 0:
        raise ValueError(
            f"minimum_horizon must be a non-negative integer, got: {profile['minimum_horizon']}"
        )

    if isinstance(profile["leverage"], str):
        if profile["leverage"].lower() == "true":
            profile["leverage"] = True
        elif profile["leverage"].lower() == "false":
            profile["leverage"] = False
        else:
            raise ValueError(f"leverage must be a boolean, got string: '{profile['leverage']}'")

    if not isinstance(profile["leverage"], bool):
        raise ValueError(
            f"leverage must be a boolean, got: {type(profile['leverage'])}"
        )

    return profile


async def run_product_classifier(raw_input: str, model_client) -> dict:
    """Run the A2 agent on raw_input. Returns a validated product_profile dict."""
    agent = AssistantAgent(
        name="ProductClassifier",
        model_client=model_client,
        system_message=PRODUCT_CLASSIFIER_SYSTEM_PROMPT,
    )

    response = await agent.on_messages(
        [TextMessage(content=raw_input, source="user")],
        cancellation_token=None,
    )

    raw_text = response.chat_message.content
    return parse_product_profile(raw_text)
