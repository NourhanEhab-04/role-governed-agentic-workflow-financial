# agents/client_profiler.py

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from schemas.client_profile import REQUIRED_CLIENT_KEYS
import json
import re


CLIENT_PROFILER_SYSTEM_PROMPT = """
You are A1, the Client Profiler agent in a MiFID II suitability assessment pipeline.

YOUR ONLY JOB: Parse the client input you receive and return a single JSON object
containing exactly the fields listed below. Do not add commentary, explanations,
or any text outside the JSON object.

REQUIRED OUTPUT FORMAT — return exactly this structure:
{
    "financial_knowledge": "<one of: none | basic | moderate | advanced>",
    "risk_tolerance_score": <integer 1-10>,
    "investment_horizon": <integer, years>,
    "liquid_assets": <float, EUR>,
    "income": <float, EUR annual>,
    "investment_amount": <float, EUR>,
    "can_afford_total_loss": <true | false>,
    "financial_vulnerability": "<one of: LOW | MEDIUM | HIGH>"
}

FIELD DEFINITIONS:
- financial_knowledge: client's self-declared understanding of financial instruments.
  Map described experience to: none (no experience), basic (bank products only),
  moderate (stocks/funds experience), advanced (derivatives/complex instruments).
- risk_tolerance_score: integer 1-10. 1=no risk, 10=maximum risk. If described in
  words, map: conservative=2, cautious=3, balanced=5, growth=7, aggressive=9.
- investment_horizon: how many years the client intends to hold the investment.
  If a range is given, use the lower bound.
- liquid_assets: total immediately accessible cash/savings in EUR.
- income: gross annual income in EUR.
- investment_amount: amount the client wants to invest in EUR.
- can_afford_total_loss: true only if the client explicitly states or clearly implies
  they can absorb losing the full investment_amount without hardship.
- financial_vulnerability: HIGH if client is over 70, heavily indebted, unemployed,
  or states they need this money for essential expenses. MEDIUM if close to retirement
  or income is irregular. LOW otherwise.

IF A REQUIRED FIELD CANNOT BE DETERMINED from the input, output this instead:
{
    "status": "needs_clarification",
    "missing": ["<field_name_1>", "<field_name_2>"]
}

OUTPUT RULES:
- Output only the JSON object. No preamble, no explanation, no markdown fences.
- Never invent data. If a field is genuinely absent, use needs_clarification.
- All monetary values must be floats (e.g. 8000.0 not 8000).
- risk_tolerance_score must be an integer, not a string.
"""


def parse_client_profile(raw_text: str) -> dict:
    """Extract and return a client_profile dict from agent text output.
    Raises ValueError if JSON is missing, malformed, incomplete, or has invalid enum values."""
    # Step 1: try to find a JSON block (handles markdown fences)
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in agent output")

    # Step 2: parse it
    try:
        profile = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Agent output contained malformed JSON: {e}")

    # Step 3: validate all required keys are present
    missing = REQUIRED_CLIENT_KEYS - profile.keys()
    if missing:
        raise ValueError(f"client_profile missing required keys: {missing}")

    # Step 3b: validate that numeric/boolean fields are not null
    null_fields = [k for k in REQUIRED_CLIENT_KEYS if profile.get(k) is None]
    if null_fields:
        raise ValueError(
            f"client_profile has null values for required fields: {null_fields}. "
            "These must be extractable from the client description."
        )

    # Step 4: validate enum values
    valid_knowledge = {"none", "basic", "moderate", "advanced"}
    if profile["financial_knowledge"] not in valid_knowledge:
        raise ValueError(
            f"Invalid financial_knowledge: '{profile['financial_knowledge']}'. "
            f"Must be one of {valid_knowledge}"
        )

    valid_vulnerability = {"LOW", "MEDIUM", "HIGH"}
    if profile["financial_vulnerability"] not in valid_vulnerability:
        raise ValueError(
            f"Invalid financial_vulnerability: '{profile['financial_vulnerability']}'. "
            f"Must be one of {valid_vulnerability}"
        )

    return profile


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
