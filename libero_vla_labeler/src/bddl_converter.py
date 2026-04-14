"""
BDDL → Canonical Operators (PDDL-style) via LLM.

Input : path to a LIBERO .bddl file
Output: list of dicts, each representing one canonical operator:
    {
        "operator": "place[moka_pot_1, stove_1]",
        "preconditions": ["On[stove_1]", "In[moka_pot_1, hand]"],
        "effects":       ["On[moka_pot_1, stove_1]"],
        "description":   "Place the moka pot on the stove."
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are converting a robot manipulation BDDL task into a canonical task definition.\n"
    "Infer the minimal sequence of robot-intent operators needed to satisfy the task.\n"
    "Each operator should be task-level and canonical, with explicit parameters, description, key_action, "
    "key_event, preconditions, effects, and source_predicates.\n"
    "Return strict JSON with keys: problem_name, language, initial_state, goal_state, operators.\n"
    "The operators field must be a list of objects with keys: name, parameters, description, key_action, "
    "key_event, preconditions, effects, source_predicates.\n"
    "Avoid mentioning internal representations such as regions or low-level simulator identifiers when a more "
    "human-meaningful referent is available.\n"
    "Remove non-observable predicates such as Reachable(...) and IKFeasible(...).\n"
    "Use the smallest number of operators required to achieve the goal.\n"
    "The description field should be a natural-language instruction that could be passed to a robot, "
    'for example: "open the drawer".\n'
    "Operator names must be short and generic verbs — use Close, Open, Place, PickUp, TurnOn, TurnOff, Push, Pull and so on. "
    "Never invent task-specific names like CloseCabinet or PlaceBowlOnCabinet.\n"
    "Object arguments must be minimal human-readable nouns. If only one instance of an object type is present, "
    "use just the type (e.g. 'cabinet', 'bowl', 'stove'). Only add a brief qualifier when disambiguation is "
    "necessary (e.g. 'left plate', 'top drawer'). Never use simulator identifiers like wooden_cabinet_1 or "
    "region suffixes like _top_region.\n"
    "Return ONLY valid JSON, no markdown fences, no extra text."
)

_USER_TEMPLATE = "Structured context: {context_json}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_bddl(bddl_path: str | Path) -> str:
    return Path(bddl_path).read_text(encoding="utf-8")


def convert_bddl_to_operators(bddl_content: str, config: dict) -> dict:
    """
    Call the configured LLM to convert a BDDL string into a canonical task definition.
    Returns a dict with keys: problem_name, language, initial_state, goal_state, operators.
    """
    context = {"bddl": bddl_content}
    provider = config["llm"]["provider"].lower()
    if provider == "gemini":
        return _gemini_convert(context, config)
    elif provider == "openai":
        return _openai_convert(context, config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

def _gemini_convert(context: dict, config: dict) -> dict:
    import google.generativeai as genai
    from src.utils import get_api_key

    genai.configure(api_key=get_api_key(config, "gemini"))
    model = genai.GenerativeModel(
        model_name=config["llm"]["model"],
        system_instruction=_SYSTEM_PROMPT,
    )
    prompt = _USER_TEMPLATE.format(context_json=json.dumps(context, indent=2))
    response = model.generate_content(prompt)
    return _parse_json_response(response.text)


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _openai_convert(context: dict, config: dict) -> dict:
    from openai import OpenAI
    from src.utils import get_api_key

    client = OpenAI(api_key=get_api_key(config, "openai"))
    prompt = _USER_TEMPLATE.format(context_json=json.dumps(context, indent=2))
    response = client.chat.completions.create(
        model=config["llm"]["model"],
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return _parse_json_response(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON:\n{text}\n\nError: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got: {type(data)}")
    required = {"problem_name", "language", "initial_state", "goal_state", "operators"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Response missing keys: {missing}")
    return data
