# agents/client_profiler.py

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
import json




CLIENT_PROFILER_SYSTEM_PROMPT = """
You are A1, the Client Profiler agent in a MiFID II suitability assessment pipeline.

YOUR ONLY JOB: Parse the client input and return a single JSON object with exactly
the fields below. Output ONLY the JSON object — no preamble, no explanation,
no markdown fences, no trailing text.

════════════════════════════════════════
REQUIRED OUTPUT FORMAT
════════════════════════════════════════
{
    "financial_knowledge": "<none | basic | moderate | advanced>",
    "risk_tolerance_score": <integer 1–10>,
    "investment_horizon": <integer, YEARS>,
    "liquid_assets": <float, EUR>,
    "income": <float, EUR annual>,
    "investment_amount": <float, EUR>,
    "can_afford_total_loss": <true | false>,
    "financial_vulnerability": "<LOW | MEDIUM | HIGH>",
    "age": <integer | null>
}

════════════════════════════════════════
FIELD-BY-FIELD RULES  (follow strictly, in order)
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
Integer 1-10. Use this mapping for verbal descriptions:

  1  = no risk at all, capital preservation only
  2  = conservative
  3  = cautious / very low risk
  4  = moderately cautious
  5  = balanced / moderate
  6  = moderately growth-oriented
  7  = growth / willing to accept significant losses
  8  = high risk / aggressive-leaning
  9  = aggressive
  10 = maximum risk, speculative

  If the client gives a numeric score directly, use it verbatim (clamp to 1–10).
  If they describe a mix (e.g. "balanced but leaning towards growth"), pick the
  average of the two mapped values and round down (balanced=5, growth=7 → 6).

── investment_horizon ───────────────────────────────────────────────────────
Output in YEARS as an integer. ALWAYS convert from whatever unit the client uses:

  CONVERSION TABLE (always apply before outputting):
    "X months"        → floor(X / 12). Example: 12 months → 1, 18 months → 1
    "X weeks"         → floor(X / 52).
    "X years"         → X  (no conversion needed)
    "short-term"      → 1
    "medium-term"     → 5
    "long-term"       → 10

  If a RANGE is given (e.g. "3-5 years"), use the LOWER bound → 3.
  If the client gives only months, NEVER output the raw month number — always
  divide and floor. E.g. "24 months" → 2, NOT 24.

── liquid_assets ────────────────────────────────────────────────────────────
Total immediately accessible cash or savings (checking, savings, money-market)
in EUR as a float. Exclude illiquid assets (real estate, pension funds, locked
deposits). Convert currencies using approximate mid-market rates if needed and
note uncertainty with a round figure. If multiple accounts are listed, sum them.

── income ───────────────────────────────────────────────────────────────────
Gross annual income in EUR as a float.
  - Monthly income stated → multiply by 12.
  - Net/take-home stated → gross up by approximately 1.3 (flag as estimated).
  - Zero income (unemployed, retired with no pension stated) → 0.0

── investment_amount ────────────────────────────────────────────────────────
Amount the client wants to invest NOW, in EUR as a float.
  - Must be a positive float.
  - If expressed as a % of savings, compute the absolute amount.

── can_afford_total_loss ────────────────────────────────────────────────────
true  → ONLY if the client EXPLICITLY states they can lose the entire amount,
        OR clearly implies it (e.g. "this is purely speculative money I can
        write off", "I won't miss it if it's gone", "I am fine losing 100%").

false → ANY other case, including:
        • Silence on the topic
        • Vague language or uncertainty
        • Partial-loss tolerance only ("I can handle losing some")
        • Explicit statement that total loss WOULD harm them financially
          (e.g. "losing everything would seriously affect me", "I cannot
          afford to lose it all", "total loss would be devastating")

  CRITICAL: A statement that partial loss is acceptable does NOT imply total
  loss is affordable. Read the FULL sentence. If ANY part of the client's
  statement indicates total loss would hurt them → false.

  EXAMPLES:
    "I can lose part but losing all would seriously affect me" → false
    "losing everything would seriously affect me financially"  → false
    "I'm fine losing some but not everything"                 → false
    "this is my emergency fund so I need it back"             → false
    "this is play money, I can write it off entirely"         → true

  DEFAULT is false.

── age ───────────────────────────────────────────────────────────────────────
Client's age as an integer (years). Output null if the client does not state
their age. Do NOT infer age from retirement mentions or other hints.

── financial_vulnerability ──────────────────────────────────────────────────
Assign HIGH, MEDIUM, or LOW using this decision tree (apply top-down, stop at
the FIRST match):

  HIGH   → ANY of:
           • Client is over age 70 (i.e., age field > 70)
           • Client states they are heavily in debt / has significant loans
             relative to income
           • Client is unemployed (income = 0 and not retired with pension)
           • Client states this money is needed for essential expenses
             (rent, medical, education, living costs)
           • Client mentions serious health issues affecting finances

  MEDIUM → ANY of:
           • Client is within 5 years of stated retirement age
           • Client has irregular income (freelance, commission, seasonal)
           • Client mentions significant upcoming expenses (buying a home,
             tuition) but investment_amount does not deplete liquid_assets
             entirely

  LOW    → None of the above conditions are met.

  CRITICAL EXCLUSIONS — these do NOT trigger MEDIUM or HIGH:
    • Investing a large fraction (even >50%) of liquid_assets, unless the
      client explicitly states this depletes funds needed for essential costs
    • Expressing concern about loss ("losing all would affect me") — that is
      a loss-tolerance statement, NOT a vulnerability indicator
    • Having a stable salary without any HIGH/MEDIUM conditions above
    • Absence of mentions: if the client does NOT mention debt, health issues,
      irregular income, retirement proximity, or essential-expense reliance
      → assign LOW. Silence on vulnerability conditions = LOW, never MEDIUM.

════════════════════════════════════════
MISSING FIELDS: needs_clarification
════════════════════════════════════════
If ANY required field cannot be determined from the input, return:
{
    "status": "needs_clarification",
    "missing": ["<field_name_1>", "<field_name_2>"]
}

List ONLY the fields that are genuinely absent. Do not invent values.

════════════════════════════════════════
OUTPUT RULES (non-negotiable)
════════════════════════════════════════
1. Output ONLY the JSON object. Nothing before it, nothing after it.
2. All monetary values must be floats with one decimal: 8000.0 not 8000.
3. risk_tolerance_score must be an integer, never a string or float.
4. investment_horizon must be an integer in YEARS (already converted).
5. financial_knowledge must be lowercase: none | basic | moderate | advanced.
6. financial_vulnerability must be uppercase: LOW | MEDIUM | HIGH.
7. can_afford_total_loss must be a boolean: true or false (not a string).
8. age must be an integer if stated, or null if not mentioned. Never infer it.
9. Never add extra fields beyond the 9 listed above.
10. Never invent or assume data that is not stated or clearly implied.
"""


def parse_client_profile(raw_text: str) -> dict:
    """Extract and return a validated client_profile dict from agent text output."""
    from agents.parsing import extract_json_object
    from schemas.output_models import ClientProfileModel
    from pydantic import ValidationError
    data = extract_json_object(raw_text)
    try:
        return ClientProfileModel.model_validate(data).model_dump()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


async def run_client_profiler(raw_input: str, model_client) -> dict:
    """Run the A1 agent on raw_input. Returns a validated client_profile dict."""
    agent = AssistantAgent(
        name="ClientProfiler",
        model_client=model_client,
        system_message=CLIENT_PROFILER_SYSTEM_PROMPT,
    )

    response = await agent.on_messages(
        [TextMessage(content=raw_input, source="user")],
        cancellation_token=None,
    )

    raw_text = response.chat_message.content
    return parse_client_profile(raw_text)


async def run_client_profiler_with_feedback(
    raw_input: str,
    previous_output: dict,
    verification_result: dict,
    model_client,
    attempt_history: list | None = None,
) -> dict:
    """
    Retry A1 with surgical verifier feedback.

    Passing fields are locked — their current values are copied verbatim into
    the LOCKED_FIELDS section so the model cannot accidentally regress them.
    Only the failed fields are presented for re-reasoning.

    attempt_history: all prior (output, verifier_issues, failed_fields) tuples from
    earlier rounds; included in the prompt so the agent can't go in circles.
    """
    agent = AssistantAgent(
        name="ClientProfiler",
        model_client=model_client,
        system_message=CLIENT_PROFILER_SYSTEM_PROMPT,
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
        f"CLIENT INPUT:\n{raw_input}\n"
        + history_note
        + f"\nYOUR LATEST OUTPUT (flagged by the verifier):\n"
        f"{json.dumps(previous_output, indent=2)}\n\n"
        f"VERIFIER ISSUES:\n"
        + "\n".join(f"- {i}" for i in issues)
        + f"\n\nFAILED FIELDS (fix ONLY these):\n{json.dumps(failed_fields, indent=2)}"
        + locked_note
        + f"\n\nFocus your reasoning on the FAILED FIELDS. "
        f"Re-read the CLIENT INPUT and correct only the failed fields. "
        f"Output the complete JSON with all 9 fields."
    )

    response = await agent.on_messages(
        [TextMessage(content=feedback_prompt, source="user")],
        cancellation_token=None,
    )
    return parse_client_profile(response.chat_message.content)
