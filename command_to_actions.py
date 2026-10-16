#!/usr/bin/env python3
"""
Convert a natural-language robot command into a validated JSON action list.

Local LLM mode:
  python3 command_to_actions.py "go to the door and wave"
  python3 command_to_actions.py --interactive

Demo/test mode (does not require a downloaded model):
  python3 command_to_actions.py --demo "go to the door and wave"
  python3 command_to_actions.py --tests

Install for local LLM mode:
  python3 -m pip install llama-cpp-python
Place a GGUF instruct model at models/qwen2.5-1.5b-instruct-q4_k_m.gguf,
or set MODEL_PATH to your model file.
"""
import argparse
import json
import os
import re
import sys
from typing import Any

DEFAULT_MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# The robot/controller must implement these action names.
ALLOWED_ACTIONS = {
    "move_to", "wave", "stop", "turn", "move_forward",
    "move_backward", "pick_up", "put_down", "speak", "wait",
}
REQUIRED_FIELDS = {
    "move_to": {"action", "target"},
    "wave": {"action"},
    "stop": {"action"},
    "turn": {"action", "direction"},
    "move_forward": {"action", "distance_m"},
    "move_backward": {"action", "distance_m"},
    "pick_up": {"action", "object"},
    "put_down": {"action", "object", "target"},
    "speak": {"action", "text"},
    "wait": {"action", "seconds"},
}

SYSTEM_PROMPT = """You translate robot commands into JSON only.
Return exactly this shape: {"actions":[{"action":"..."}]}.
Allowed actions and fields:
- move_to: target (short location/object phrase)
- wave: no extra fields
- stop: no extra fields
- turn: direction ("left" or "right")
- move_forward: distance_m (number)
- move_backward: distance_m (number)
- pick_up: object (short object phrase)
- put_down: object and target (short phrases)
- speak: text (string)
- wait: seconds (number)
Use only allowed actions. Preserve the order of the user's instructions.
Do not invent actions. If a command is unclear or unsupported, return {"actions": []}.
No markdown fences, comments, or extra text."""


def parse_action_json(raw: str) -> dict[str, Any]:
    """Parse LLM output and validate the JSON action list."""
    # Allow accidental surrounding prose/fences, but extract the first JSON object.
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model output.")
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(data, dict) or set(data.keys()) != {"actions"}:
        raise ValueError('Top-level JSON must contain only the "actions" key.')
    if not isinstance(data["actions"], list):
        raise ValueError('"actions" must be a list.')

    validated = []
    for index, item in enumerate(data["actions"]):
        if not isinstance(item, dict) or "action" not in item:
            raise ValueError(f"Action {index} must be an object with an action field.")
        name = item["action"]
        if name not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported action at index {index}: {name!r}")
        required = REQUIRED_FIELDS[name]
        if set(item.keys()) != required:
            missing = required - set(item.keys())
            extra = set(item.keys()) - required
            raise ValueError(
                f"Action {index} fields mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        if name in {"move_forward", "move_backward", "wait"}:
            field = "seconds" if name == "wait" else "distance_m"
            if not isinstance(item[field], (int, float)) or isinstance(item[field], bool) or item[field] < 0:
                raise ValueError(f"Action {index}: {field} must be a non-negative number.")
        if name == "turn" and item["direction"] not in {"left", "right"}:
            raise ValueError(f"Action {index}: direction must be 'left' or 'right'.")
        for field in {"target", "object", "text"} & set(item):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Action {index}: {field} must be a non-empty string.")
        validated.append(item)
    return {"actions": validated}


def get_llm(model_path: str):
    """Load llama-cpp-python only when LLM mode is requested."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Run: python3 -m pip install llama-cpp-python"
        ) from exc
    if not os.path.isfile(model_path):
        raise RuntimeError(
            f"Model file not found: {model_path}\n"
            "Set MODEL_PATH or place a GGUF instruct model at that path."
        )
    return Llama(model_path=model_path, n_ctx=2048, verbose=False)


def llm_to_actions(command: str, llm) -> dict[str, Any]:
    if not command.strip():
        raise ValueError("Command cannot be empty.")
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": command.strip()},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    raw = response["choices"][0]["message"]["content"]
    return parse_action_json(raw)


def demo_to_actions(command: str) -> dict[str, Any]:
    """
    Deterministic demo parser for the included test cases.
    This is not a replacement for LLM mode; it makes validation/tests runnable
    without installing a model.
    """
    s = command.lower().strip()
    actions = []

    # Ordered, small command patterns for repeatable examples.
    if "go to" in s or "move to" in s or "navigate to" in s:
        match = re.search(r"(?:go to|move to|navigate to)\s+(.*?)(?=\s+and\s+|\s*,|$)", s)
        if match:
            actions.append({"action": "move_to", "target": match.group(1).strip(" .")})
    if "wave" in s:
        actions.append({"action": "wave"})
    if "turn left" in s:
        actions.append({"action": "turn", "direction": "left"})
    elif "turn right" in s:
        actions.append({"action": "turn", "direction": "right"})
    m = re.search(r"(?:move|go)\s+forward\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?", s)
    if m:
        actions.append({"action": "move_forward", "distance_m": float(m.group(1))})
    m = re.search(r"(?:move|go)\s+back(?:ward)?\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters)?", s)
    if m:
        actions.append({"action": "move_backward", "distance_m": float(m.group(1))})
    m = re.search(r"pick up\s+(.*?)(?=\s+and\s+|\s*,|$)", s)
    if m:
        actions.append({"action": "pick_up", "object": m.group(1).strip(" .")})
    m = re.search(r"put down\s+(.*?)\s+on\s+(.*?)(?:[.]|$)", s)
    if m:
        actions.append({"action": "put_down", "object": m.group(1).strip(), "target": m.group(2).strip()})
    m = re.search(r"(?:say|speak)\s+['\"]?(.*?)['\"]?$", s)
    if m:
        actions.append({"action": "speak", "text": m.group(1).strip(" .\"'")})
    m = re.search(r"wait\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)?", s)
    if m:
        actions.append({"action": "wait", "seconds": float(m.group(1))})

    if "stop" in s:
        actions.append({"action": "stop"})

    return parse_action_json(json.dumps({"actions": actions}))


TESTS = [
    ("go to the door and wave", {"actions": [{"action": "move_to", "target": "the door"}, {"action": "wave"}]}),
    ("move to the kitchen", {"actions": [{"action": "move_to", "target": "the kitchen"}]}),
    ("navigate to the charging station and stop", {"actions": [{"action": "move_to", "target": "the charging station"}, {"action": "stop"}]}),
    ("turn left", {"actions": [{"action": "turn", "direction": "left"}]}),
    ("turn right", {"actions": [{"action": "turn", "direction": "right"}]}),
    ("move forward 2 meters", {"actions": [{"action": "move_forward", "distance_m": 2.0}]}),
    ("go forward 0.5 m", {"actions": [{"action": "move_forward", "distance_m": 0.5}]}),
    ("move backward 1 meter", {"actions": [{"action": "move_backward", "distance_m": 1.0}]}),
    ("pick up the red box", {"actions": [{"action": "pick_up", "object": "the red box"}]}),
    ("put down the book on the table", {"actions": [{"action": "put_down", "object": "the book", "target": "the table"}]}),
    ('say "Hello there"', {"actions": [{"action": "speak", "text": "hello there"}]}),
    ("wait 5 seconds", {"actions": [{"action": "wait", "seconds": 5.0}]}),
    ("go to the window and turn left", {"actions": [{"action": "move_to", "target": "the window"}, {"action": "turn", "direction": "left"}]}),
    ("move to the desk and wave", {"actions": [{"action": "move_to", "target": "the desk"}, {"action": "wave"}]}),
    ("go forward 3 m and stop", {"actions": [{"action": "move_forward", "distance_m": 3.0}, {"action": "stop"}]}),
    ("pick up the bottle and go to the sink", {"actions": [{"action": "move_to", "target": "the sink"}, {"action": "pick_up", "object": "the bottle"}]}),
    ("wave", {"actions": [{"action": "wave"}]}),
    ("stop", {"actions": [{"action": "stop"}]}),
    ("wait 1 second and wave", {"actions": [{"action": "wave"}, {"action": "wait", "seconds": 1.0}]}),
    ("dance on the moon", {"actions": []}),
]


def run_tests() -> int:
    passed = 0
    for i, (command, expected) in enumerate(TESTS, 1):
        # Tests intentionally use the deterministic demo parser, not an LLM,
        # so expected output is reproducible on any machine.
        actual = demo_to_actions(command)
        ok = actual == expected
        print(f"{i:02d}. COMMAND: {command}")
        print("    OUTPUT:  " + json.dumps(actual, ensure_ascii=False))
        print(f"    RESULT:  {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("    EXPECTED: " + json.dumps(expected, ensure_ascii=False))
        passed += int(ok)
    print(f"\nSummary: {passed}/{len(TESTS)} tests passed.")
    return 0 if passed == len(TESTS) else 1


def main():
    parser = argparse.ArgumentParser(description="Convert natural-language commands to validated JSON actions.")
    parser.add_argument("command", nargs="?", help="Natural-language robot command")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH), help="Path to local GGUF model")
    parser.add_argument("--demo", action="store_true", help="Use deterministic demo parser, no model required")
    parser.add_argument("--tests", action="store_true", help="Run the 20 included test cases")
    parser.add_argument("--interactive", action="store_true", help="Read commands until 'quit'")
    args = parser.parse_args()

    if args.tests:
        raise SystemExit(run_tests())
    if args.interactive:
        llm = None if args.demo else get_llm(args.model)
        while True:
            try:
                command = input("Command (or 'quit'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if command.lower() in {"quit", "exit"}:
                break
            try:
                result = demo_to_actions(command) if args.demo else llm_to_actions(command, llm)
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except (ValueError, RuntimeError) as exc:
                print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return
    if not args.command:
        parser.print_help()
        return
    try:
        result = demo_to_actions(args.command) if args.demo else llm_to_actions(args.command, get_llm(args.model))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
