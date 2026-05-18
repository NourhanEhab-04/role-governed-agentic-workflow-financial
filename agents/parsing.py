# agents/parsing.py
# Shared JSON extraction utilities for agent output parsers.

import json


def _try_repair(text: str) -> dict | None:
    """Try json_repair on text; return dict or None if unavailable/failed."""
    try:
        from json_repair import repair_json
        recovered = repair_json(text, return_objects=True)
        if isinstance(recovered, dict):
            return recovered
    except Exception:
        pass
    return None


def extract_json_object(text: str) -> dict:
    """
    Extract the first well-formed JSON object from an LLM output string.

    Primary strategy: brace-balanced walk — correctly handles trailing text
    and nested objects, stops at the right closing brace even when the LLM
    appends extra content after the main object.

    Fallback: json_repair on the isolated candidate (malformed but bounded),
    then on the full suffix from '{' (truncated / unmatched braces).

    Raises ValueError if no complete object can be recovered.
    """
    start = text.find('{')
    if start == -1:
        raise ValueError(f"No JSON object found in agent output: {text!r}")

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # ast.literal_eval cannot parse JSON booleans/null; skip it.
                    # Try json_repair on the already-isolated candidate instead.
                    repaired = _try_repair(candidate)
                    if repaired is not None:
                        return repaired
                    raise ValueError(
                        f"Agent output contained malformed JSON that could not be repaired: "
                        f"{candidate!r}"
                    )

    # Brace walk found no matching '}' (truncated output, unclosed string, etc.).
    # Try json_repair on the full suffix starting from '{'.
    repaired = _try_repair(text[start:])
    if repaired is not None:
        return repaired

    raise ValueError(f"Unmatched braces in agent output: {text!r}")
