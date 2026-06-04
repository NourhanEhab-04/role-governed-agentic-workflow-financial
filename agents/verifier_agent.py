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

When verification fails, the orchestrator feeds the verifier's issues back to
the original A1/A2 agent so it can retry — there is no separate corrector.
"""

import json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage



# ── System prompts ────────────────────────────────────────────────────────────

VERIFIER_SYSTEM_PROMPT_A1 = """
You are AV, a STRICT VERIFIER for the Client Profiler (A1).

You will receive:
- ORIGINAL_INPUT (raw client text)
- PARSED_OUTPUT (JSON produced by A1)

Your ONLY job:
→ Check whether PARSED_OUTPUT correctly follows the Client Profiler rules below.

You are NOT allowed to fix, improve, reinterpret, or regenerate values.

════════════════════════════════════════
🚫 CRITICAL VERIFICATION MODE (READ FIRST)
════════════════════════════════════════

You are NOT a profiler. You are NOT generating values.

DO:
✔ Check if each field is supported by the input + rules  
✔ Compare values EXACTLY  
✔ Use explicit evidence  

DO NOT:
✘ Recompute everything from scratch  
✘ Override explicit numbers  
✘ Infer missing data  
✘ “Improve” outputs  

If a value is acceptable under the rules → supported = true  
If it contradicts rules or input → supported = false  

════════════════════════════════════════
🔥 PRIORITY RULE (VERY IMPORTANT)
════════════════════════════════════════

EXPLICIT INPUT ALWAYS WINS.

If ORIGINAL_INPUT explicitly states:
- a number (e.g. 7, 5 years, €10,000)
→ that exact value MUST appear in PARSED_OUTPUT.

If it does not match exactly → supported = false

DO NOT reinterpret explicit numbers. EVER.

════════════════════════════════════════
📏 CLIENT PROFILER RULES (REFERENCE ONLY — DO NOT RE-GENERATE)
════════════════════════════════════════

Use these ONLY to verify correctness — not to recompute freely.

── financial_knowledge ─────────────────
none → no experience  
basic → only deposits/bonds  
moderate → stocks/ETFs/funds  
advanced → derivatives  

DEFAULT → choose LOWER if unclear

── risk_tolerance_score ────────────────
1–10 mapping

IMPORTANT:
- If numeric is explicitly given → MUST match exactly
- If mixed description → average + round down

── investment_horizon ─────────────────
Convert to YEARS:
- months → floor(/12)
- weeks → floor(/52)
- short-term → 1
- medium → 5
- long → 10
- range → LOWER bound

VERIFY conversion only — do not reinterpret intent

── liquid_assets ──────────────────────
Only liquid cash/savings  
Sum if multiple  

── income ─────────────────────────────
Annual EUR  
Monthly → ×12  
Net → ×1.3  

── investment_amount ──────────────────
Amount invested NOW  
Must be positive  

── can_afford_total_loss ──────────────
true ONLY if client explicitly states they can lose the ENTIRE amount
("write it off", "I won't miss it", "purely speculative money").

false → default for ALL other cases:
  • Partial-loss tolerance ("I can handle losing some") → false
  • Explicit statement that total loss WOULD harm/affect them → false
  • Silence → false

CRITICAL (false positive): If the client says "losing everything would seriously
affect me financially" (or similar) and A1 output is true → supported = false.
Partial-loss tolerance does NOT imply total-loss affordability.

CRITICAL (false negative): If ORIGINAL_INPUT is a JSON object that explicitly
contains "can_afford_total_loss": true and A1 output is false → supported = false.
An explicit boolean true in the input overrides the default-false rule. This is
NOT a case of "silence" or "default applied correctly" — it directly contradicts
an explicit field value.

── age ────────────────────────────────
Integer if stated in input, null otherwise. Never inferred.

── financial_vulnerability ────────────
HIGH → debt, unemployed, essential money, age field > 70, serious health issue
MEDIUM → irregular income, near retirement (<5 yrs), explicit upcoming
         large expense (home purchase, tuition)
LOW → none of the above

Top-down — stop at first match.

CRITICAL AGE RULE: if the age field is present and > 70,
financial_vulnerability MUST be HIGH. Any other value → supported = false.

CRITICAL: Do NOT accept MEDIUM unless one of its conditions appears
explicitly in the input. The following are NOT MEDIUM triggers:
  • Investing a large fraction of liquid_assets
  • Client expressing concern about total loss
  • Stable employment with no other signals
If MEDIUM is output but none of its conditions appear → supported = false.

════════════════════════════════════════
⚠️ WHAT "FABRICATED" MEANS — READ BEFORE VERIFYING
════════════════════════════════════════

A value is FABRICATED only when it:
  ✗ Directly CONTRADICTS something the client explicitly stated, OR
  ✗ Violates the field rules with NO basis at all

A value is NOT fabricated when:
  ✓ It correctly applies the DEFAULT rule to a missing field
  ✓ It correctly maps a verbal description to a number via the rule table
  ✓ It is a reasonable inference from stated information

KEY RULE:
"The client did not mention field X" → does NOT mean X is fabricated.
It means A1 applied the DEFAULT or rule mapping.
Your job: check whether the default/mapping was applied CORRECTLY — not flag it as invented.

EXAMPLES:
  • Client says "moderate risk" → A1 outputs risk_tolerance_score: 5 → supported = true (rule mapping)
  • Client doesn't mention can_afford_total_loss → A1 outputs false → supported = true (default = false)
  • Client says "7 out of 10 risk" → A1 outputs risk_tolerance_score: 5 → supported = false (contradicts explicit number)

════════════════════════════════════════
🧠 HOW TO VERIFY (STRICT PROCESS)
════════════════════════════════════════

For EACH field:

1. Look for EXPLICIT evidence in ORIGINAL_INPUT
2. If explicit number/value → check EXACT match; mismatch = supported: false
3. If verbal description → check whether the rule MAPPING was applied correctly
4. If field not mentioned at all → check whether the DEFAULT was applied correctly
5. Only mark supported: false if the value CONTRADICTS the input or VIOLATES the rule

⚠ "Not mentioned" ≠ "fabricated". Do NOT flag correctly-defaulted fields.
⚠ If you find yourself flagging everything → STOP → you are misapplying the rules.

════════════════════════════════════════
📊 OUTPUT FORMAT (STRICT)
════════════════════════════════════════

Return ONLY a single JSON object with exactly these four top-level keys.
CRITICAL: "passed" and "confidence" MUST be top-level keys — never omit them.
Do NOT return just the field_checks dict. The outer wrapper is mandatory.

{
  "passed": <true | false>,
  "confidence": <float>,
  "issues": ["..."],
  "field_checks": {
    "<field>": {
      "supported": <true | false>,
      "reason": "<cite the explicit input text OR the rule/default that was applied>"
    }
  }
}

════════════════════════════════════════
📉 CONFIDENCE RULES
════════════════════════════════════════

1.0 → all fields clearly supported by input or correct rule application
0.8 → minor ambiguity in one field, but value is within acceptable range
0.6 → one field clearly contradicts the input or rule
0.4 → multiple fields contradict the input or rules
0.0 → output contradicts most explicit inputs (genuine fabrication)

DO NOT use 0.0 just because the input is short or some fields are inferred.

════════════════════════════════════════
🚨 FINAL RULE
════════════════════════════════════════

You are a VALIDATOR, not a THINKER.

Do NOT:
- reinterpret tone
- adjust numbers
- "improve" anything
- flag correct defaults as errors

Just check:
→ Does this value contradict the input or violate the rule? If yes → false. Otherwise → true.
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

MINIMUM HORIZON — evidence-first:
First, search ORIGINAL_INPUT for any explicit statement of a minimum or
recommended holding period (e.g. "three years", "minimum 5-year horizon",
"recommended investment horizon is N years"). If such a statement is present,
the correct value is that exact number — mark any other value as supported: false.
Only if NO explicit horizon is stated should you fall back to the risk class
table default. Never accept 0.

════════════════════════════════════════
REQUIRED OUTPUT FORMAT
════════════════════════════════════════
Output ONLY a single JSON object — no preamble, no markdown fences.
CRITICAL: "passed" and "confidence" MUST be top-level keys — never omit them.
Do NOT return just the field_checks dict. The outer wrapper is mandatory.

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

def _extract_all_json_objects(text: str) -> list:
    """
    Walk through text and return every complete, balanced JSON object found.
    Used so we can pick the best candidate instead of blindly taking the first.
    """
    import json as _json
    objects = []
    pos = 0
    while True:
        start = text.find('{', pos)
        if start == -1:
            break
        depth = 0
        in_string = False
        escape_next = False
        found_end = None
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    found_end = i
                    break
        if found_end is None:
            break
        candidate = text[start:found_end + 1]
        try:
            obj = _json.loads(candidate)
            if isinstance(obj, dict):
                objects.append(obj)
        except Exception:
            pass
        pos = found_end + 1
    return objects


def _recover_flat_field_checks(data: dict) -> dict:
    """
    Recover when the LLM returns only the field_checks content at the top level,
    omitting the required passed / confidence / issues wrapper.

    Detects the pattern: no 'passed' key, and all values are dicts with 'supported'.
    Reconstructs a valid VerificationResultModel-shaped dict from the field checks.
    """
    if "passed" in data or "confidence" in data:
        return data  # already has the wrapper — nothing to fix
    if not data:
        return data
    all_field_checks = all(
        isinstance(v, dict) and "supported" in v for v in data.values()
    )
    if not all_field_checks:
        return data  # not a field_checks-only dict — let validation handle it
    failed = [
        (k, v) for k, v in data.items() if v.get("supported") is False
    ]
    num_failed = len(failed)
    confidence = round(max(0.4, 1.0 - 0.2 * num_failed), 2)
    issues = [f"{k}: {v.get('reason', '')}" for k, v in failed]
    return {
        "passed": num_failed == 0,
        "confidence": confidence,
        "issues": issues,
        "field_checks": data,
    }


def parse_verification_result(raw_text: str) -> dict:
    """Extract and return a validated verification_result dict from verifier output."""
    from agents.parsing import extract_json_object
    from schemas.output_models import VerificationResultModel
    from pydantic import ValidationError
    import re
    # Sanitize control characters that break json.loads inside string values
    sanitized = re.sub(r'[\x00-\x1f]', ' ', raw_text)
    # Try all JSON objects in the response; prefer the one that already has
    # the required top-level keys so a stray field-check fragment doesn't win.
    candidates = _extract_all_json_objects(sanitized)
    data = next(
        (c for c in candidates if "passed" in c and "confidence" in c),
        candidates[0] if candidates else extract_json_object(sanitized),
    )
    data = _recover_flat_field_checks(data)
    try:
        return VerificationResultModel.model_validate(data).model_dump()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


# ── Agent runners ─────────────────────────────────────────────────────────────

def _build_verifier_message(
    original_input: str,
    parsed_output: dict,
    consistency_issues: list | None = None,
    prior_verification: dict | None = None,
) -> str:
    msg = (
        f"ORIGINAL_INPUT:\n{original_input}\n\n"
        f"PARSED_OUTPUT:\n{json.dumps(parsed_output, indent=2)}"
    )
    if consistency_issues:
        issues_text = "\n".join(f"- {i}" for i in consistency_issues)
        msg += (
            f"\n\nDETERMINISTIC_PRE_CHECKS (already confirmed by rule engine — "
            f"treat these as definite errors, focus your LLM reasoning on the remaining fields):\n"
            f"{issues_text}"
        )
    if prior_verification:
        prior_failed = {
            field: check
            for field, check in prior_verification.get("field_checks", {}).items()
            if check.get("supported") is False
        }
        prior_issues = prior_verification.get("issues", [])
        msg += (
            f"\n\nPRIOR_VERIFICATION_CONTEXT (a previous verifier pass found these problems — "
            f"check whether the corrected PARSED_OUTPUT has addressed them):\n"
            f"Previously failed fields:\n{json.dumps(prior_failed, indent=2)}\n"
            f"Prior issues:\n" + "\n".join(f"- {i}" for i in prior_issues)
        )
    return msg


async def run_verifier_on_a1(
    original_input: str,
    client_profile: dict,
    model_client,
    consistency_issues: list | None = None,
    prior_verification: dict | None = None,
) -> dict:
    """
    Verify A1's client_profile against the original client input text.
    consistency_issues: pre-confirmed errors from the deterministic check layer.
    prior_verification: result from a previous verifier pass (retry context).
    Returns a verification_result dict.
    """
    agent = AssistantAgent(
        name="VerifierA1",
        model_client=model_client,
        system_message=VERIFIER_SYSTEM_PROMPT_A1,
    )
    message = _build_verifier_message(
        original_input, client_profile, consistency_issues, prior_verification
    )
    response = await agent.on_messages(
        [TextMessage(content=message, source="user")],
        cancellation_token=None,
    )
    return parse_verification_result(response.chat_message.content)


async def run_verifier_on_a2(
    original_input: str,
    product_profile: dict,
    model_client,
    consistency_issues: list | None = None,
    prior_verification: dict | None = None,
) -> dict:
    """
    Verify A2's product_profile against the original product input text.
    consistency_issues: pre-confirmed errors from the deterministic check layer.
    prior_verification: result from a previous verifier pass (retry context).
    Returns a verification_result dict.
    """
    agent = AssistantAgent(
        name="VerifierA2",
        model_client=model_client,
        system_message=VERIFIER_SYSTEM_PROMPT_A2,
    )
    message = _build_verifier_message(
        original_input, product_profile, consistency_issues, prior_verification
    )
    response = await agent.on_messages(
        [TextMessage(content=message, source="user")],
        cancellation_token=None,
    )
    return parse_verification_result(response.chat_message.content)


# ── Corrector helpers ─────────────────────────────────────────────────────────

CORRECTOR_SYSTEM_PROMPT_A1 = (
    "You are CorrectorA1. You receive a client input, a previous flawed output, "
    "the failed fields, and verifier issues. Re-read the client input from scratch "
    "and produce a fully corrected JSON output following all Client Profiler rules exactly. "
    "Output ONLY the JSON object."
)

CORRECTOR_SYSTEM_PROMPT_A2 = (
    "You are CorrectorA2. You receive a product description, a previous flawed output, "
    "the failed fields, and verifier issues. Re-read the product description from scratch "
    "and produce a fully corrected JSON output following all Product Classifier rules exactly. "
    "Output ONLY the JSON object."
)


def _build_corrector_message(
    original_input: str,
    previous_output: dict,
    verification_result: dict,
) -> str:
    """Build the message sent to the corrector agent."""
    failed_fields = {
        field: check
        for field, check in verification_result.get("field_checks", {}).items()
        if check.get("supported") is False
    }
    issues = verification_result.get("issues", [])
    return (
        f"ORIGINAL_INPUT:\n{original_input}\n\n"
        f"PREVIOUS_OUTPUT:\n{json.dumps(previous_output, indent=2)}\n\n"
        f"FAILED_FIELDS:\n{json.dumps(failed_fields, indent=2)}\n\n"
        f"VERIFIER_ISSUES:\n" + "\n".join(f"- {i}" for i in issues)
    )


async def run_corrector_on_a1(
    original_input: str,
    previous_output: dict,
    verification_result: dict,
    model_client,
) -> dict:
    """Run the corrector agent for A1. Returns a validated client_profile dict."""
    from agents.client_profiler import parse_client_profile
    agent = AssistantAgent(
        name="CorrectorA1",
        model_client=model_client,
        system_message=CORRECTOR_SYSTEM_PROMPT_A1,
    )
    message = _build_corrector_message(original_input, previous_output, verification_result)
    response = await agent.on_messages(
        [TextMessage(content=message, source="user")],
        cancellation_token=None,
    )
    return parse_client_profile(response.chat_message.content)


async def run_corrector_on_a2(
    original_input: str,
    previous_output: dict,
    verification_result: dict,
    model_client,
) -> dict:
    """Run the corrector agent for A2. Returns a validated product_profile dict."""
    from agents.product_classifier import parse_product_profile
    agent = AssistantAgent(
        name="CorrectorA2",
        model_client=model_client,
        system_message=CORRECTOR_SYSTEM_PROMPT_A2,
    )
    message = _build_corrector_message(original_input, previous_output, verification_result)
    response = await agent.on_messages(
        [TextMessage(content=message, source="user")],
        cancellation_token=None,
    )
    return parse_product_profile(response.chat_message.content)


