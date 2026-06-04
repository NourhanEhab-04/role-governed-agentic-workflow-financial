"""
evaluation/baseline_single_llm.py
===================================
Baseline: single LLM call (Architecture A).

This is the control condition for the thesis evaluation.
No multi-agent structure, no deterministic rule engine, no verifier.
The LLM receives raw client + product text and is prompted to apply
the same seven MiFID II rules entirely from its own reasoning.

Used by evaluation/evaluator.py to compare against the full pipeline
(Architecture B: multi-agent + rule engine).

Evaluation premise
------------------
The thesis research question is:
  "Does decomposing financial decision-making into structured agent roles
   improve consistency and rule compliance vs a single LLM call?"

This baseline intentionally uses the SAME model as the pipeline agents
(controlled experiment) so that any accuracy/consistency difference is
attributable to the architecture, not to a stronger model.
"""

import json
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from agents.parsing import extract_json_object


BASELINE_SYSTEM_PROMPT = """
You are a MiFID II suitability assessment system.

You receive a client description and a product description.
Your job: assess whether the product is suitable for the client under
MiFID II Article 25(2) by applying the following seven rules.

Apply ALL seven rules and return a single JSON object.

════════════════════════════════════════
MiFID II SUITABILITY RULES
════════════════════════════════════════

R1 — Knowledge check
  PASS if client financial knowledge level >= product required knowledge level.
  Levels (ascending): none < basic < moderate < advanced.

R2 — Risk alignment
  PASS if product risk class <= client risk tolerance score + 2.
  Risk class: 1 (lowest) to 7 (highest).
  Risk tolerance score: 1 (no risk) to 10 (maximum risk).

R3 — Investment horizon
  PASS if client investment horizon (years) >= product minimum horizon (years).

R4 — Affordability
  PASS unless: client cannot afford total loss AND product has total potential loss.
  If client can_afford_total_loss=false and product potential_loss="total" → FAIL.

R5 — Vulnerability
  PASS unless: client financial_vulnerability="HIGH" AND product risk_class >= 5.

R6 — Leverage
  PASS unless: product is leveraged AND client risk_tolerance_score < 7.

R7 — Complexity
  PASS unless: product complexity_tier="COMPLEX" AND client financial_knowledge
  is "none" or "basic" (i.e. below "moderate").

════════════════════════════════════════
DECISION LOGIC
════════════════════════════════════════
Step 1 — Apply hard-fail rules:
  If ANY of R1, R4, R5, R7 failed → decision = UNSUITABLE, escalated = false. STOP.

Step 2 — Compute score (soft-fail rules only, start at 100):
  R2 fail: -20
  R3 fail: -15
  R6 fail: -25

Step 3 — Score-based decision (no hard fails):
  score < 40  → UNSUITABLE, escalated = false
  score 40–69 → CONDITIONAL, then apply Step 4
  score >= 70 → SUITABLE, then apply Step 4

Step 4 — Escalation check (apply ONLY when score-based decision is SUITABLE or CONDITIONAL):
  Set escalated = true and decision = ESCALATED if ANY of the following hold:

  (a) Vulnerability override — ESMA GL §86-88:
      Client financial_vulnerability = "HIGH" AND all hard rules pass (R5 passed,
      meaning product risk_class < 5) AND score-based decision is SUITABLE or CONDITIONAL.
      Rationale: HIGH-vulnerability clients receiving a positive recommendation must
      be referred to a human compliance officer for review, even when all rules pass.

  (b) Profile contradiction:
      There is a significant internal contradiction in the client's profile that
      makes the assessment unreliable. Examples:
        - Client states a conservative or capital-preservation objective but has a
          high risk tolerance score (>= 7), or vice versa.
        - Client states they cannot afford any loss but requests a growth portfolio.
        - The client description contains explicit conflicting statements about
          risk appetite or objectives.
      Rationale: contradictory profiles cannot be reliably assessed; human review
      is required before a recommendation is made.

  If NEITHER (a) nor (b) applies, keep the score-based decision and set escalated = false.

════════════════════════════════════════
REQUIRED OUTPUT FORMAT — JSON only
════════════════════════════════════════
{
  "decision": "<SUITABLE | CONDITIONAL | UNSUITABLE | ESCALATED>",
  "escalated": <true | false>,
  "score": <integer>,
  "hard_failed_rules": [<list of rule IDs that are hard fails>],
  "rules": {
    "R1": "<PASS | FAIL>",
    "R2": "<PASS | FAIL>",
    "R3": "<PASS | FAIL>",
    "R4": "<PASS | FAIL>",
    "R5": "<PASS | FAIL>",
    "R6": "<PASS | FAIL>",
    "R7": "<PASS | FAIL>"
  },
  "client_facing_summary": "<2-3 sentence plain English explanation of the decision>"
}

Output ONLY the JSON object. No preamble. No markdown fences.
"""


def _parse_baseline_result(raw_text: str) -> dict:
    """Extract and lightly validate the baseline result dict."""
    data = extract_json_object(raw_text)
    required = {"decision", "score", "rules"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Baseline result missing keys: {missing}")
    valid_decisions = {"SUITABLE", "CONDITIONAL", "UNSUITABLE", "ESCALATED"}
    if data.get("decision") not in valid_decisions:
        raise ValueError(f"Invalid baseline decision: '{data.get('decision')}'")
    # Normalise escalated flag: true when decision is ESCALATED or field says true
    data["escalated"] = bool(data.get("escalated", False)) or data.get("decision") == "ESCALATED"
    return data


async def run_baseline(
    client_input: str,
    product_input: str,
    model_client,
) -> dict:
    """
    Run the baseline single-LLM assessment.

    Parameters
    ----------
    client_input   : raw client description text (same as pipeline input)
    product_input  : raw product description text (same as pipeline input)
    model_client   : AutoGen model client (same model as pipeline agents)

    Returns
    -------
    dict with keys: decision, score, hard_failed_rules, rules,
                    client_facing_summary, raw_text
    """
    agent = AssistantAgent(
        name="BaselineLLM",
        model_client=model_client,
        system_message=BASELINE_SYSTEM_PROMPT,
    )

    payload = (
        f"CLIENT DESCRIPTION:\n{client_input}\n\n"
        f"PRODUCT DESCRIPTION:\n{product_input}"
    )

    response = await agent.on_messages(
        [TextMessage(content=payload, source="user")],
        cancellation_token=CancellationToken(),
    )

    raw_text = response.chat_message.content
    try:
        result = _parse_baseline_result(raw_text)
    except (ValueError, Exception) as exc:
        result = {
            "decision":          "PARSE_ERROR",
            "score":             None,
            "hard_failed_rules": [],
            "rules":             {},
            "client_facing_summary": f"Parse error: {exc}",
        }

    result["raw_text"] = raw_text
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from config.llm_config import get_model_client

    if len(sys.argv) != 3:
        print("Usage: python -m evaluation.baseline_single_llm <client.json> <product.json>")
        sys.exit(1)

    client_text  = Path(sys.argv[1]).read_text()
    product_text = Path(sys.argv[2]).read_text()

    result = asyncio.run(
        run_baseline(client_text, product_text, get_model_client())
    )
    print(json.dumps(result, indent=2))
