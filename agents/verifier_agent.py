# agents/verifier_agent.py
"""
VerifierAgent (AV) — LLM-as-judge that checks A1 and A2 outputs.

For each agent, it receives:
  - The original raw input text the agent processed
  - The parsed JSON dict the agent returned

It returns a structured verification_result dict (see schemas/verification_result.py).

Two entry points:
  run_verifier_on_a1(original_input, client_profile, model_client) -> dict
  run_verifier_on_a2(original_input, product_profile, model_client) -> dict
"""

import json
import re

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from schemas.verification_result import CONFIDENCE_THRESHOLD


# ── System prompts ────────────────────────────────────────────────────────────

VERIFIER_SYSTEM_PROMPT_A1 = """
You are AV, the Verifier Agent for A1 (Client Profiler) in a MiFID II pipeline.

You will receive:
  ORIGINAL_INPUT — the raw text description the client profiler received.
  PARSED_OUTPUT  — the JSON the client profiler returned.

Your job: re-apply the EXACT SAME rules the client profiler uses (listed below)
to check whether each field in PARSED_OUTPUT is the correct value for the given
ORIGINAL_INPUT. If the correct value by these rules differs from what was output,
mark that field unsupported.

════════════════════════════════════════
CLIENT PROFILER FIELD RULES (same rules A1 uses — apply them here)
════════════════════════════════════════

── financial_knowledge ──────────────────────────────────────────────────────
Map ONLY to one of: none | basic | moderate | advanced

  none     → client has NO investing experience at all, or explicitly states
             they have never invested, don't understand financial products,
             or are a complete beginner. Phrases like "no experience",
             "never invested", "don't know much", "first time" → none.

  basic    → client has experience ONLY with bank deposits, savings accounts,
             fixed-term deposits, or government savings bonds. No stocks, no
             funds, no anything beyond plain bank products.

  moderate → client has bought or sold individual stocks, mutual funds, ETFs,
             REITs, or similar listed instruments. Acknowledges understanding
             of market risk.

  advanced → client has experience with derivatives (options, futures, CFDs,
             structured products, leveraged instruments, or alternatives).

  DEFAULT RULE: When in doubt, assign the LOWER category. Do NOT upgrade a
  client's knowledge level based on vocabulary alone — only on described
  instruments or explicit self-declaration.

── risk_tolerance_score ─────────────────────────────────────────────────────
Integer 1-10. Verbal mapping:
  1=no risk/capital preservation, 2=conservative, 3=cautious/very low,
  4=moderately cautious, 5=balanced/moderate, 6=moderately growth-oriented,
  7=growth/willing to accept significant losses, 8=high risk/aggressive-leaning,
  9=aggressive, 10=maximum risk/speculative.
  If client gives a numeric score directly, use it verbatim (clamp 1–10).
  Mixed descriptions → average mapped values, round down.

── investment_horizon ───────────────────────────────────────────────────────
Output in YEARS as an integer. Conversion rules:
  "X months" → floor(X/12). "X weeks" → floor(X/52). "X years" → X.
  "short-term" → 1. "medium-term" → 5. "long-term" → 10.
  Range given → use LOWER bound.

── liquid_assets ────────────────────────────────────────────────────────────
Total immediately accessible cash/savings in EUR (float). Exclude illiquid
assets (real estate, pension, locked deposits). Sum multiple accounts.

── income ───────────────────────────────────────────────────────────────────
Gross annual income in EUR (float).
  Monthly → multiply by 12. Net/take-home → gross up by ~1.3.
  Zero income (unemployed, retired with no pension stated) → 0.0.

── investment_amount ────────────────────────────────────────────────────────
Amount client wants to invest NOW, in EUR (positive float).
If expressed as % of savings, compute absolute amount.

── can_afford_total_loss ────────────────────────────────────────────────────
true  → ONLY if client EXPLICITLY states they can lose the entire amount,
        OR clearly implies it ("purely speculative money", "won't miss it").
false → ANY other case, including silence, vague language, or uncertainty.
        Default is false.

── financial_vulnerability ──────────────────────────────────────────────────
HIGH   → ANY of: over age 70; heavily in debt relative to income; unemployed
         (income=0 and not retired with pension); money needed for essential
         expenses (rent, medical, education, living costs); serious health
         issues affecting finances.
MEDIUM → ANY of: within 5 years of stated retirement age; irregular income
         (freelance, commission, seasonal); significant upcoming expenses but
         investment_amount does not deplete liquid_assets entirely.
LOW    → None of the above.
Apply top-down, stop at FIRST match.

════════════════════════════════════════
REQUIRED OUTPUT FORMAT
════════════════════════════════════════
Output ONLY a single JSON object — no preamble, no markdown fences.

{
    "passed": <true | false>,
    "confidence": <float 0.0–1.0>,
    "issues": ["<issue description>", ...],
    "field_checks": {
        "<field_name>": {
            "supported": <true | false>,
            "reason": "<one sentence citing the rule and the evidence>"
        },
        ...
    }
}

"supported": false when the field value does not match what the rules above
produce for the given ORIGINAL_INPUT. "supported": true otherwise.

"passed" rules:
  - true  → ALL fields are supported AND confidence >= 0.70
  - false → ANY field is unsupported OR confidence < 0.70

"confidence":
  - 1.0  → every field clearly correct per rules and input
  - 0.8  → all correct; minor ambiguity in one field
  - 0.6  → one field is wrong or significantly ambiguous
  - 0.4  → multiple fields are wrong
  - 0.0  → output is fabricated or wildly inconsistent

Output ONLY the JSON object. Nothing else.
"""


VERIFIER_SYSTEM_PROMPT_A2 = """
You are AV, the Verifier Agent for A2 (Product Classifier) in a MiFID II pipeline.

You will receive:
  ORIGINAL_INPUT — the raw text description of the financial product.
  PARSED_OUTPUT  — the JSON the product classifier returned.

Your job: re-apply the EXACT SAME rules the product classifier uses (listed below)
to check whether each field in PARSED_OUTPUT is the correct value for the given
ORIGINAL_INPUT. If the correct value by these rules differs from what was output,
mark that field unsupported.

════════════════════════════════════════
PRODUCT CLASSIFIER RULES (same rules A2 uses — apply them here)
════════════════════════════════════════

ESMA PRIIP RISK CLASS TABLE:
  Class 1 — money market funds, insured deposits, capital-protected structures
            → leverage=false, potential_loss=partial, complexity=NON-COMPLEX,
              knowledge=none, min_horizon=1
  Class 2 — government bonds (AAA-AA), investment-grade bond funds
            → leverage=false, potential_loss=partial, complexity=NON-COMPLEX,
              knowledge=basic, min_horizon=2
  Class 3 — investment-grade corporate bonds, balanced funds, high-grade bond ETFs
            → leverage=false, potential_loss=partial, complexity=NON-COMPLEX,
              knowledge=basic, min_horizon=3
  Class 4 — diversified equity ETFs, equity index funds, multi-asset, blue-chip equities
            → leverage=false, potential_loss=partial, complexity=NON-COMPLEX,
              knowledge=basic, min_horizon=3
  Class 5 — sector ETFs, single-country equity funds, high-yield bonds, REITs,
             small-cap equity funds, plain vanilla options (buying only)
            → leverage=false (unless stated), potential_loss=partial,
              complexity=COMPLEX for options/structured; NON-COMPLEX for ETFs/REITs,
              knowledge=moderate, min_horizon=5
  Class 6 — leveraged ETFs (2x), single volatile/speculative stocks, emerging market
             equity funds, structured products with capital at risk, futures, spread betting
            → complexity=COMPLEX, knowledge=advanced, min_horizon=5
              leverage=true for leveraged ETFs/derivatives; false for single stocks
              potential_loss=total for leveraged/derivatives; partial for single stocks
  Class 7 — leveraged ETFs (3x+), CFDs, uncovered options, crypto derivatives,
             speculative OTC derivatives
            → leverage=true, potential_loss=total, complexity=COMPLEX,
              knowledge=advanced, min_horizon=1

COMPLEXITY RULES:
  NON-COMPLEX → frequently traded on regulated markets, no embedded derivatives,
                adequate public info, no contingent liability beyond purchase cost.
                Examples: plain vanilla ETFs, government bonds, standard equity funds.
  COMPLEX     → contains embedded derivatives, contingent liability, or requires
                specific expertise. Examples: leveraged ETFs, CFDs, futures, options,
                structured products, OTC derivatives.

LEVERAGE RULES:
  true  → mechanically amplifies returns AND losses beyond 1:1
           (2x ETF, 3x ETF, CFDs, futures, margin products)
  false → does not amplify beyond the invested amount
           (standard ETFs, bonds, equity funds, money market funds)
  Note: a product that can lose 100% is NOT automatically leveraged.

POTENTIAL LOSS RULES:
  total   → realistic scenario of losing entire invested amount
             (leveraged ETFs, CFDs, futures, uncovered options, crypto derivatives)
  partial → full capital loss is NOT realistic under normal market conditions
             (diversified equity funds, bonds, standard ETFs, balanced funds)

MINIMUM HORIZON: use risk class defaults above. Override only if product is
explicitly designed as short-term (e.g. CFD = 1 year). Never set to 0.

════════════════════════════════════════
REQUIRED OUTPUT FORMAT
════════════════════════════════════════
Output ONLY a single JSON object — no preamble, no markdown fences.

{
    "passed": <true | false>,
    "confidence": <float 0.0–1.0>,
    "issues": ["<issue description>", ...],
    "field_checks": {
        "<field_name>": {
            "supported": <true | false>,
            "reason": "<one sentence citing the rule and the evidence>"
        },
        ...
    }
}

"supported": false when the field value does not match what the rules above
produce for the given ORIGINAL_INPUT. "supported": true otherwise.

"passed" rules:
  - true  → ALL fields are supported AND confidence >= 0.70
  - false → ANY field is unsupported OR confidence < 0.70

"confidence":
  - 1.0  → every field clearly correct per rules and input
  - 0.8  → all correct; minor ambiguity in one field
  - 0.6  → one field is wrong or significantly ambiguous
  - 0.4  → multiple fields are wrong
  - 0.0  → output is fabricated or wildly inconsistent

Output ONLY the JSON object. Nothing else.
"""


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_verification_result(raw_text: str) -> dict:
    """
    Extract and validate a verification_result dict from verifier agent output.
    Raises ValueError on missing/malformed JSON or invalid structure.
    """
    from schemas.verification_result import (
        REQUIRED_VERIFICATION_KEYS,
        CONFIDENCE_THRESHOLD,
        validate_verification_result,
    )

    # Extract JSON block (handles accidental markdown fences)
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in verifier output")

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Verifier output contained malformed JSON: {e}")

    # Structural validation
    ok, err = validate_verification_result(result)
    if not ok:
        raise ValueError(f"verification_result invalid: {err}")

    # Enforce: if confidence is below threshold, passed must be False
    if result["confidence"] < CONFIDENCE_THRESHOLD and result["passed"] is True:
        result["passed"] = False
        result["issues"].append(
            f"passed forced to False: confidence {result['confidence']:.2f} "
            f"is below threshold {CONFIDENCE_THRESHOLD}"
        )

    return result


# ── Agent runners ─────────────────────────────────────────────────────────────

def _build_verifier_message(original_input: str, parsed_output: dict) -> str:
    return (
        f"ORIGINAL_INPUT:\n{original_input}\n\n"
        f"PARSED_OUTPUT:\n{json.dumps(parsed_output, indent=2)}"
    )


async def run_verifier_on_a1(
    original_input: str,
    client_profile: dict,
    model_client,
) -> dict:
    """
    Verify A1's client_profile against the original client input text.
    Returns a verification_result dict.
    """
    agent = AssistantAgent(
        name="VerifierA1",
        model_client=model_client,
        system_message=VERIFIER_SYSTEM_PROMPT_A1,
    )
    message = _build_verifier_message(original_input, client_profile)
    response = await agent.on_messages(
        [TextMessage(content=message, source="user")],
        cancellation_token=None,
    )
    return parse_verification_result(response.chat_message.content)


async def run_verifier_on_a2(
    original_input: str,
    product_profile: dict,
    model_client,
) -> dict:
    """
    Verify A2's product_profile against the original product input text.
    Returns a verification_result dict.
    """
    agent = AssistantAgent(
        name="VerifierA2",
        model_client=model_client,
        system_message=VERIFIER_SYSTEM_PROMPT_A2,
    )
    message = _build_verifier_message(original_input, product_profile)
    response = await agent.on_messages(
        [TextMessage(content=message, source="user")],
        cancellation_token=None,
    )
    return parse_verification_result(response.chat_message.content)


# ── Corrector system prompts ──────────────────────────────────────────────────

CORRECTOR_SYSTEM_PROMPT_A1 = """
You are AC1, the Correction Agent for A1 (Client Profiler) in a MiFID II pipeline.

A previous agent extracted a client profile from the input text, but the Verifier
found that specific fields contain wrong values. Your job is to re-extract ONLY
the flagged fields using the EXACT SAME rules the client profiler uses.

You will receive:
  ORIGINAL_INPUT   — the raw client description text
  PREVIOUS_OUTPUT  — the full JSON the previous agent returned
  FAILED_FIELDS    — the fields the Verifier flagged as wrong, with reasons

════════════════════════════════════════
YOUR TASK
════════════════════════════════════════
1. Read ORIGINAL_INPUT carefully.
2. For each field in FAILED_FIELDS, derive the correct value using the rules below
   (the same rules A1 uses).
3. Return the COMPLETE corrected JSON — all 8 fields — with the wrong fields fixed.
   Keep all non-flagged fields exactly as they appear in PREVIOUS_OUTPUT.

════════════════════════════════════════
REQUIRED OUTPUT FORMAT
════════════════════════════════════════
Output ONLY the corrected JSON object. No preamble, no explanation, no fences.

{
    "financial_knowledge": "<none | basic | moderate | advanced>",
    "risk_tolerance_score": <integer 1–10>,
    "investment_horizon": <integer, YEARS>,
    "liquid_assets": <float, EUR>,
    "income": <float, EUR annual>,
    "investment_amount": <float, EUR>,
    "can_afford_total_loss": <true | false>,
    "financial_vulnerability": "<LOW | MEDIUM | HIGH>"
}

════════════════════════════════════════
CLIENT PROFILER FIELD RULES (apply these when correcting)
════════════════════════════════════════

── financial_knowledge ──────────────────────────────────────────────────────
  none     → no investing experience, never invested, "don't know much", "first time"
  basic    → experience ONLY with bank deposits, savings accounts, fixed-term
             deposits, or government savings bonds
  moderate → has bought/sold individual stocks, mutual funds, ETFs, REITs
  advanced → experience with derivatives (options, futures, CFDs, structured
             products, leveraged instruments, or alternatives)
  DEFAULT: when in doubt, assign the LOWER category.

── risk_tolerance_score ─────────────────────────────────────────────────────
  1=no risk, 2=conservative, 3=cautious, 4=moderately cautious, 5=balanced,
  6=moderately growth, 7=growth, 8=high risk, 9=aggressive, 10=maximum/speculative.
  Numeric score given → use verbatim (clamp 1–10).
  Mixed description → average mapped values, round down.

── investment_horizon ───────────────────────────────────────────────────────
  "X months" → floor(X/12). "X weeks" → floor(X/52). "X years" → X.
  "short-term"→1, "medium-term"→5, "long-term"→10. Range → LOWER bound.

── liquid_assets ────────────────────────────────────────────────────────────
  Immediately accessible cash/savings in EUR (float). Exclude illiquid assets.
  Sum multiple accounts.

── income ───────────────────────────────────────────────────────────────────
  Gross annual EUR (float). Monthly → ×12. Net → gross up by ~1.3. Unemployed/no pension → 0.0.

── investment_amount ────────────────────────────────────────────────────────
  Amount to invest now, EUR (positive float). % of savings → compute absolute.

── can_afford_total_loss ────────────────────────────────────────────────────
  true  → ONLY if client EXPLICITLY states they can lose the entire amount,
          or clearly implies it ("purely speculative", "won't miss it").
  false → any other case including silence. Default is false.

── financial_vulnerability ──────────────────────────────────────────────────
  HIGH   → over 70; heavily in debt relative to income; unemployed (income=0,
           not retired with pension); money needed for essential expenses;
           serious health issues affecting finances.
  MEDIUM → within 5 years of retirement; irregular income; significant upcoming
           expenses but investment_amount doesn't deplete liquid_assets entirely.
  LOW    → none of the above. Apply top-down, stop at FIRST match.

Output ONLY the JSON. Nothing else.
"""

CORRECTOR_SYSTEM_PROMPT_A2 = """
You are AC2, the Correction Agent for A2 (Product Classifier) in a MiFID II pipeline.

A previous agent classified a financial product, but the Verifier found that specific
fields contain wrong values. Your job is to re-classify ONLY the flagged fields using
the EXACT SAME rules the product classifier uses.

You will receive:
  ORIGINAL_INPUT   — the raw product description text
  PREVIOUS_OUTPUT  — the full JSON the previous agent returned
  FAILED_FIELDS    — the fields the Verifier flagged as wrong, with reasons

════════════════════════════════════════
YOUR TASK
════════════════════════════════════════
1. Read ORIGINAL_INPUT carefully.
2. For each field in FAILED_FIELDS, derive the correct value using the rules below
   (the same rules A2 uses).
3. Return the COMPLETE corrected JSON — all 7 fields — with the wrong fields fixed.
   Keep all non-flagged fields exactly as they appear in PREVIOUS_OUTPUT.

════════════════════════════════════════
REQUIRED OUTPUT FORMAT
════════════════════════════════════════
Output ONLY the corrected JSON object. No preamble, no explanation, no fences.

{
    "product_name": "<descriptive name>",
    "risk_class": <integer 1–7>,
    "complexity_tier": "<NON-COMPLEX | COMPLEX>",
    "requires_knowledge_level": "<none | basic | moderate | advanced>",
    "minimum_horizon": <integer, years>,
    "potential_loss": "<partial | total>",
    "leverage": <true | false>
}

════════════════════════════════════════
PRODUCT CLASSIFIER RULES (apply these when correcting)
════════════════════════════════════════

ESMA PRIIP RISK CLASS TABLE:
  Class 1 — money market funds, insured deposits, capital-protected structures
            → leverage=false, potential_loss=partial, complexity=NON-COMPLEX,
              knowledge=none, min_horizon=1
  Class 2 — government bonds (AAA-AA), investment-grade bond funds
            → leverage=false, potential_loss=partial, complexity=NON-COMPLEX,
              knowledge=basic, min_horizon=2
  Class 3 — investment-grade corporate bonds, balanced funds, high-grade bond ETFs
            → leverage=false, potential_loss=partial, complexity=NON-COMPLEX,
              knowledge=basic, min_horizon=3
  Class 4 — diversified equity ETFs, equity index funds, multi-asset, blue-chip equities
            → leverage=false, potential_loss=partial, complexity=NON-COMPLEX,
              knowledge=basic, min_horizon=3
  Class 5 — sector ETFs, single-country equity funds, high-yield bonds, REITs,
             small-cap equity funds, plain vanilla options (buying only)
            → leverage=false (unless stated), potential_loss=partial,
              complexity=COMPLEX for options/structured; NON-COMPLEX for ETFs/REITs,
              knowledge=moderate, min_horizon=5
  Class 6 — leveraged ETFs (2x), volatile/speculative single stocks, emerging market
             equity funds, structured products with capital at risk, futures, spread betting
            → complexity=COMPLEX, knowledge=advanced, min_horizon=5
              leverage=true for leveraged ETFs/derivatives; false for single stocks
              potential_loss=total for leveraged/derivatives; partial for single stocks
  Class 7 — leveraged ETFs (3x+), CFDs, uncovered options, crypto derivatives,
             speculative OTC derivatives
            → leverage=true, potential_loss=total, complexity=COMPLEX,
              knowledge=advanced, min_horizon=1

LEVERAGE: true → amplifies beyond 1:1 (2x/3x ETF, CFD, futures, margin).
          false → does not amplify (standard ETFs, bonds, equity funds).
          A product that can lose 100% is NOT automatically leveraged.

POTENTIAL LOSS: total → realistic full loss scenario.
                partial → full capital loss not realistic under normal conditions.

MINIMUM HORIZON: use risk class defaults. Override only for explicitly short-term
instruments (e.g. CFD=1 year). Never set to 0.

Output ONLY the JSON. Nothing else.
"""


# ── Corrector message builder ─────────────────────────────────────────────────

def _build_corrector_message(
    original_input: str,
    previous_output: dict,
    verification_result: dict,
) -> str:
    """Build the user message for the correction agent.
    Extracts only the failed fields from verification_result to keep the message focused."""
    failed_fields = {
        field: check
        for field, check in verification_result.get("field_checks", {}).items()
        if check.get("supported") is False
    }
    # Also include top-level issues for context
    top_issues = verification_result.get("issues", [])

    return (
        f"ORIGINAL_INPUT:\n{original_input}\n\n"
        f"PREVIOUS_OUTPUT:\n{json.dumps(previous_output, indent=2)}\n\n"
        f"FAILED_FIELDS:\n{json.dumps(failed_fields, indent=2)}\n\n"
        f"VERIFIER_ISSUES:\n" + "\n".join(f"- {i}" for i in top_issues)
    )


# ── Corrector runners ─────────────────────────────────────────────────────────

async def run_corrector_on_a1(
    original_input: str,
    bad_profile: dict,
    verification_result: dict,
    model_client,
) -> dict:
    """
    Run the A1 correction agent to fix fields flagged by the verifier.
    Returns a corrected client_profile dict (validated through parse_client_profile).
    Raises ValueError if the corrected output is still invalid.
    """
    from agents.client_profiler import parse_client_profile

    agent = AssistantAgent(
        name="CorrectorA1",
        model_client=model_client,
        system_message=CORRECTOR_SYSTEM_PROMPT_A1,
    )
    message = _build_corrector_message(original_input, bad_profile, verification_result)
    response = await agent.on_messages(
        [TextMessage(content=message, source="user")],
        cancellation_token=None,
    )
    # Re-use the same strict parser that A1 uses — corrected output must pass it
    return parse_client_profile(response.chat_message.content)


async def run_corrector_on_a2(
    original_input: str,
    bad_profile: dict,
    verification_result: dict,
    model_client,
) -> dict:
    """
    Run the A2 correction agent to fix fields flagged by the verifier.
    Returns a corrected product_profile dict (validated through parse_product_profile).
    Raises ValueError if the corrected output is still invalid.
    """
    from agents.product_classifier import parse_product_profile

    agent = AssistantAgent(
        name="CorrectorA2",
        model_client=model_client,
        system_message=CORRECTOR_SYSTEM_PROMPT_A2,
    )
    message = _build_corrector_message(original_input, bad_profile, verification_result)
    response = await agent.on_messages(
        [TextMessage(content=message, source="user")],
        cancellation_token=None,
    )
    return parse_product_profile(response.chat_message.content)
