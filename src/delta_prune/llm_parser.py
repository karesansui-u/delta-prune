"""JSON extraction utilities for LLM output parsing."""

from __future__ import annotations

import json
import re


def parse_json(raw_text: str) -> dict:
    """Extract a JSON object from LLM text output."""
    try:
        result = json.loads(raw_text)
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]
        return {}
    except json.JSONDecodeError:
        start = raw_text.find("{")
        if start == -1:
            return {}
        depth = 0
        for i in range(start, len(raw_text)):
            if raw_text[i] == "{":
                depth += 1
            elif raw_text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw_text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


def parse_json_array(raw_text: str) -> list[dict]:
    """Extract a JSON array of objects from LLM text output."""
    try:
        result = json.loads(raw_text)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if isinstance(result, dict):
            return [result]
        return []
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array
    start = raw_text.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(raw_text)):
            if raw_text[i] == "[":
                depth += 1
            elif raw_text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(raw_text[start : i + 1])
                        if isinstance(result, list):
                            return [item for item in result if isinstance(item, dict)]
                    except json.JSONDecodeError:
                        break
                    break

    # Fall back to extracting individual JSON objects
    objects: list[dict] = []
    pos = 0
    while pos < len(raw_text):
        obj_start = raw_text.find("{", pos)
        if obj_start == -1:
            break
        depth = 0
        for i in range(obj_start, len(raw_text)):
            if raw_text[i] == "{":
                depth += 1
            elif raw_text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw_text[obj_start : i + 1])
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    pos = i + 1
                    break
        else:
            break
    return objects


def validate_confidence(llm_output: dict) -> int:
    """Structural check + minimum of LLM-claimed confidence value."""
    claimed = llm_output.get("confidence", 0)
    rule_text = llm_output.get("integrated_rule", "") or llm_output.get("rule_candidate", "")

    has_if_then = bool(re.search(r"IF\s+.+\s+THEN\s+", rule_text, re.IGNORECASE))
    has_concrete = not any(w in rule_text for w in ["ケースバイケース", "柔軟に", "状況に応じて"])

    structural_score = 100 if (has_if_then and has_concrete) else 0
    return min(claimed, structural_score)
