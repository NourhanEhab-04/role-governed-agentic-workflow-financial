# agents/parsing.py
# Shared JSON extraction utilities for agent output parsers.

import json
import ast


def extract_json_object(text: str) -> dict:
    """
    Extract the first well-formed JSON object from an LLM output string.

    Uses brace-balancing rather than a greedy regex so it stops at the
    correct closing '}' even when the LLM appends trailing text, notes,
    or additional JSON fragments after the main object.

    Raises ValueError if no complete object is found or the extracted
    string cannot be parsed as JSON / a Python literal.
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
                    try:
                        return ast.literal_eval(candidate)
                    except (ValueError, SyntaxError) as e:
                        raise ValueError(
                            f"Agent output contained malformed JSON: {e}"
                        )

    raise ValueError(f"Unmatched braces in agent output: {text!r}")
