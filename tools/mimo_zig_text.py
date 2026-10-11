#!/usr/bin/env python3
"""Z5.1 text parity: the same /tokenize requests against two servers (the Python engine's and the Zig engine's); the
prompt ids must be equal, case by case. Without --against it records the first server's answers.

  python3 tools/mimo_zig_text.py http://gx10:8891 --out results/zig-z5/text-python.json
  python3 tools/mimo_zig_text.py http://gx10:8892 --against results/zig-z5/text-python.json
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                         "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "search", "description": "Web search",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]
CALL = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
        "name": "get_weather", "arguments": "{\"city\": \"Sydney\", \"unit\": \"c\"}"}}]}

CASES = {
    "plain": {"messages": [{"role": "user", "content": "Hello"}]},
    "system": {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Why is the sky blue?"}]},
    "multi_turn": {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"},
                                {"role": "user", "content": "Write a haiku about Sydney."}]},
    "chinese": {"messages": [{"role": "user", "content": "请用三句话解释量子纠缠。"}]},
    "emoji_unicode": {"messages": [{"role": "user", "content": "Translate: café, naïve, Zürich, 東京, 🙂🚀 — and ‘quotes’."}]},
    "code": {"messages": [{"role": "user", "content": "```python\ndef f(x):\n\treturn x**2\n```\nAdd type hints."}]},
    "tools_offered": {"messages": [{"role": "user", "content": "Weather in Sydney?"}], "tools": TOOLS},
    "tool_roundtrip": {"messages": [{"role": "user", "content": "Weather in Sydney?"}, CALL,
                                    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21, \"sky\": \"clear\"}"}],
                       "tools": TOOLS},
    "tool_choice_none": {"messages": [{"role": "user", "content": "Weather in Sydney?"}], "tools": TOOLS, "tool_choice": "none"},
    "thinking_on": {"messages": [{"role": "user", "content": "What is 17*23?"}], "chat_template_kwargs": {"enable_thinking": True}},
    "thinking_off": {"messages": [{"role": "user", "content": "What is 17*23?"}], "chat_template_kwargs": {"enable_thinking": False}},
    "effort_low": {"messages": [{"role": "user", "content": "Plan a trip."}], "reasoning_effort": "low"},
    "effort_high": {"messages": [{"role": "user", "content": "Plan a trip."}], "reasoning_effort": "high"},
    "no_generation_prompt": {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
                             "add_generation_prompt": False},
    "content_parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]}]},
    "empty_user": {"messages": [{"role": "user", "content": ""}]},
    "long": {"messages": [{"role": "user", "content": "Summarise: " + " ".join(f"Sentence {i} about lanes and graphs." for i in range(300))}]},
    "whitespace": {"messages": [{"role": "user", "content": "  leading\n\n\ntrailing   \t"}]},
    "completion_prompt": {"prompt": "The capital of Australia is"},
    "completion_no_special": {"prompt": "<|im_start|>user\nhi<|im_end|>", "add_special_tokens": False},
}


def post(base: str, body: dict) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/tokenize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode(errors="replace")[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--out", default="")
    ap.add_argument("--against", default="")
    a = ap.parse_args()
    got = {k: post(a.base, v) for k, v in CASES.items()}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(got, f, ensure_ascii=False, indent=1)
        ok = sum("tokens" in g for g in got.values())
        print(f"recorded {ok}/{len(got)} cases from {a.base} -> {a.out}")
        for k, g in got.items():
            if "tokens" not in g:
                print(f"  {k}: {g}")
    if a.against:
        want = json.load(open(a.against))
        bad = 0
        for k in CASES:
            w, g = want.get(k, {}), got[k]
            same = w.get("tokens") == g.get("tokens") and ("error" in w) == ("error" in g)
            if not same:
                bad += 1
                wt, gt = w.get("tokens") or [], g.get("tokens") or []
                i = next((j for j in range(min(len(wt), len(gt))) if wt[j] != gt[j]), min(len(wt), len(gt)))
                print(f"DIFF {k}: {len(gt)} ids vs {len(wt)}, first difference at {i}; {g.get('error', '')}")
        print(f"{'PASS' if bad == 0 else 'FAIL'} text: {len(CASES) - bad}/{len(CASES)} cases give the same prompt ids")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
