# agents/product_classifier.py

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
import json


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
  Products: diversified equity ETFs tracking DEVELOPED-MARKET broad indices
            (e.g. MSCI World, S&P 500, STOXX 600, EURO STOXX 50),
            developed-market equity index funds, multi-asset funds,
            plain vanilla blue-chip equities
  NOTE: RC4 applies to DEVELOPED-MARKET broad indices ONLY.
        Do NOT assign RC4 to emerging-market funds — those are RC5 (see below).
  Typical loss: significant drawdowns possible, long-term positive expected
  Complexity: NON-COMPLEX
  Knowledge required: basic
  Minimum horizon: 3 years
  Potential loss: partial
  Leverage: false

Risk Class 5 — Medium-high risk
  Products: sector ETFs (technology, energy, healthcare, financials, etc.),
            diversified EMERGING-MARKET equity ETFs and index funds
            (e.g. MSCI Emerging Markets Index ETF, multi-country EM equity funds),
            single-country equity funds, high-yield bond funds,
            REITs, small-cap equity funds, plain vanilla options (buying only)
  NOTE: Any ETF with concentrated sector exposure OR significant emerging-market
        equity exposure belongs here, NOT in RC4.
        requires_knowledge_level for this class ranges from basic (NON-COMPLEX
        sector ETFs) to moderate — read the EXACT value from the product description
        text; do NOT default to "moderate" when the product text says "basic".
  Typical loss: high volatility, meaningful capital loss possible
  Complexity: COMPLEX (options/structured); NON-COMPLEX (sector ETFs/REITs/EM ETFs)
  Knowledge required: basic to moderate (read from product text — see NOTE above)
  Minimum horizon: 5 years (or as stated in product text)
  Potential loss: partial
  Leverage: false (unless stated otherwise)

Risk Class 6 — High risk
  Products: leveraged ETFs (2x), single stocks (volatile/speculative),
            single-country FRONTIER or highly concentrated emerging-market funds,
            structured products with capital at risk,
            futures contracts, spread betting
  NOTE: Broad multi-country emerging-market equity ETFs (MSCI EM, 20+ countries)
        are RC5, NOT RC6. RC6 applies to concentrated single-country EM exposure
        or frontier markets only.
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

PRIORITY RULE — text wins over the table:
If the product input EXPLICITLY states a minimum or recommended holding
period (e.g. "minimum holding period of three years", "recommended
investment horizon is five years", "minimum 3-year horizon"), use that
EXACT value regardless of the risk class table default.

Only fall back to the risk class table default when NO explicit horizon
is stated in the input.

Never set minimum_horizon to 0.

==============================================================
CLASSIFICATION DECISION PROCESS
==============================================================

1. Identify the product type from the description
2. Match to the nearest risk class using the table above
   - DEVELOPED-MARKET broad index ETF/fund → RC4
   - SECTOR ETF or EMERGING-MARKET multi-country index ETF/fund → RC5
   - Leveraged (2x) ETF, frontier/single-country EM, structured product at risk → RC6
   - Leveraged (3x+) ETF, CFD, uncovered option → RC7
3. Apply the complexity, leverage, and potential_loss rules
4. Set requires_knowledge_level STRICTLY from the product description text.
   If the text says "basic financial knowledge is required" → use "basic".
   If the text says "moderate knowledge" → use "moderate".
   If the text says "advanced" → use "advanced".
   Do NOT override the text with a table default — the text is always authoritative.
5. Set minimum_horizon using explicit product text first. If no explicit horizon is stated, use the risk class table default.
6. Return the JSON object — nothing else

IF the product cannot be identified from the description, output:
{
    "status": "needs_clarification",
    "missing": ["product_type"]
}
"""


def parse_product_profile(raw_text: str) -> dict:
    """Extract and return a validated product_profile dict from agent text output."""
    from agents.parsing import extract_json_object
    from schemas.output_models import ProductProfileModel
    from pydantic import ValidationError
    data = extract_json_object(raw_text)
    try:
        return ProductProfileModel.model_validate(data).model_dump()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


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


async def run_product_classifier_with_feedback(
    raw_input: str,
    previous_output: dict,
    verification_result: dict,
    model_client,
    attempt_history: list | None = None,
) -> dict:
    """
    Retry A2 with surgical verifier feedback.

    Passing fields are locked — their current values are copied verbatim into
    the LOCKED_FIELDS section so the model cannot accidentally regress them.
    Only the failed fields are presented for re-reasoning.

    attempt_history: all prior (output, verifier_issues, failed_fields) tuples from
    earlier rounds; included in the prompt so the agent can't go in circles.
    """
    agent = AssistantAgent(
        name="ProductClassifier",
        model_client=model_client,
        system_message=PRODUCT_CLASSIFIER_SYSTEM_PROMPT,
    )

    field_checks = verification_result.get("field_checks", {})
    failed_fields = {
        field: check
        for field, check in field_checks.items()
        if check.get("supported") is False
    }
    locked_fields = {
        field: previous_output[field]
        for field, check in field_checks.items()
        if check.get("supported") is True and field in previous_output
    }
    issues = verification_result.get("issues", [])

    locked_note = (
        f"\n\nLOCKED_FIELDS (already verified correct — copy these values EXACTLY into your output, "
        f"do NOT change them):\n{json.dumps(locked_fields, indent=2)}"
        if locked_fields else ""
    )

    # Include full transcript of prior rounds so the agent can see what it already
    # tried and why those attempts were rejected — prevents circular corrections.
    history_note = ""
    if attempt_history:
        parts = []
        for entry in attempt_history:
            parts.append(
                f"Attempt {entry['attempt']} output:\n{json.dumps(entry['output'], indent=2)}\n"
                f"Verifier rejected these fields:\n{json.dumps(entry['failed_fields'], indent=2)}\n"
                f"Issues: " + "; ".join(entry["verifier_issues"])
            )
        history_note = (
            "\n\nPRIOR_ATTEMPTS — do NOT repeat these mistakes:\n"
            + "\n---\n".join(parts)
            + "\n"
        )

    feedback_prompt = (
        f"PRODUCT INPUT:\n{raw_input}\n"
        + history_note
        + f"\nYOUR LATEST OUTPUT (flagged by the verifier):\n"
        f"{json.dumps(previous_output, indent=2)}\n\n"
        f"VERIFIER ISSUES:\n"
        + "\n".join(f"- {i}" for i in issues)
        + f"\n\nFAILED FIELDS (fix ONLY these):\n{json.dumps(failed_fields, indent=2)}"
        + locked_note
        + f"\n\nFocus your reasoning on the FAILED FIELDS. "
        f"Re-read the PRODUCT INPUT and correct only the failed fields. "
        f"Output the complete JSON with all required fields."
    )

    response = await agent.on_messages(
        [TextMessage(content=feedback_prompt, source="user")],
        cancellation_token=None,
    )
    return parse_product_profile(response.chat_message.content)
