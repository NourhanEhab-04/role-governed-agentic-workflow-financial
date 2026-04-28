# schemas/verification_result.py
"""
Schema for the output produced by the VerifierAgent (AV).

The verifier runs after A1 and after A2.  It receives the original raw input
text and the parsed JSON output, then checks whether every field is faithfully
supported by the input.

Required fields
---------------
passed       : bool   — True if all field checks pass and confidence >= threshold
confidence   : float  — 0.0 – 1.0 overall confidence in the parsed output
issues       : list   — strings describing any detected mismatches or concerns
field_checks : dict   — per-field result: {"supported": bool, "reason": str}
"""

REQUIRED_VERIFICATION_KEYS = {"passed", "confidence", "issues", "field_checks"}

CONFIDENCE_THRESHOLD = 0.70   # below this → passed is forced False


def validate_verification_result(result: dict) -> tuple:
    """
    Structural validator for a verification_result dict.
    Returns (ok: bool, error_message: str).
    Used by orchestrator validators and unit tests.
    """
    if not isinstance(result, dict):
        return False, "verification_result is not a dict"

    missing = REQUIRED_VERIFICATION_KEYS - result.keys()
    if missing:
        return False, f"verification_result missing keys: {missing}"

    if not isinstance(result["passed"], bool):
        return False, f"'passed' must be a bool, got {type(result['passed'])}"

    if not isinstance(result["confidence"], (int, float)):
        return False, f"'confidence' must be a float, got {type(result['confidence'])}"

    if not (0.0 <= result["confidence"] <= 1.0):
        return False, f"'confidence' must be between 0.0 and 1.0, got {result['confidence']}"

    if not isinstance(result["issues"], list):
        return False, f"'issues' must be a list, got {type(result['issues'])}"

    if not isinstance(result["field_checks"], dict):
        return False, f"'field_checks' must be a dict, got {type(result['field_checks'])}"

    # Each field_check entry must have "supported" (bool) and "reason" (str)
    for field, check in result["field_checks"].items():
        if not isinstance(check, dict):
            return False, f"field_checks['{field}'] must be a dict"
        if "supported" not in check:
            return False, f"field_checks['{field}'] missing 'supported'"
        if not isinstance(check["supported"], bool):
            return False, f"field_checks['{field}']['supported'] must be a bool"
        if "reason" not in check:
            return False, f"field_checks['{field}'] missing 'reason'"

    return True, ""
