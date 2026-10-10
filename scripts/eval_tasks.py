#!/usr/bin/env python3
"""Task accuracy through the server (stdlib only): GSM8K and MMLU-Pro, thinking off, greedy, the items exported by
tools/mimo_eval_data.py. Settings follow benthecarman's pack evals (max tokens 1024 / 1536).

  python3 scripts/eval_tasks.py http://127.0.0.1:8891 MiMo-V2.6-Flash results/evals --tag 4.0 [--tasks gsm8k,mmlu_pro]
                                [--concurrency 4] [--limit N]

Writes results/evals/<tag>-<task>.jsonl (every reply) and <tag>-<task>.score.json (accuracy, a 95% interval,
unparsed and truncated counts, by category). Resumes: items already in the .jsonl are not asked again.
"""

import argparse
import concurrent.futures as cf
import json
import math
import re
import time
import urllib.request
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
MAX_TOKENS = {"gsm8k": 1024, "mmlu_pro": 1536}


def prompt(task: str, it: dict) -> str:
    if task == "gsm8k":
        return (f"{it['question']}\n\nSolve the problem step by step. On the last line, write the final answer as "
                f"'Answer: <number>'.")
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["options"]))
    return (f"{it['question']}\n\nOptions:\n{opts}\n\nThink step by step, then on the last line write the final "
            f"answer as 'Answer: <letter>'.")


def parse(task: str, text: str) -> str | None:
    if task == "gsm8k":
        m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
        nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        v = nums[-1].replace(",", "").rstrip(".")
        try:
            f = float(v)
        except ValueError:
            return None
        return str(int(f)) if f == int(f) else str(f)
    m = re.findall(r"Answer:\s*\(?([A-J])\)?", text)
    if not m:
        m = re.findall(r"\b(?:answer is|answer:)\s*\(?([A-J])\)?", text, re.I)
    return m[-1].upper() if m else None


def ask(url: str, model: str, task: str, it: dict) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt(task, it)}],
            "max_tokens": MAX_TOKENS[task], "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                j = json.load(r)
            break
        except Exception as exc:                          # noqa: BLE001  (retried, then recorded)
            if attempt == 2:
                return {"id": it["id"], "error": str(exc)}
            time.sleep(5)
    text = j["choices"][0]["message"].get("content") or ""
    got = parse(task, text)
    gold = it["gold"]
    ok = got is not None and (float(got) == float(gold) if task == "gsm8k" else got == gold)
    return {"id": it["id"], "got": got, "gold": gold, "correct": bool(ok), "finish": j["choices"][0].get("finish_reason"),
            "tokens": j["usage"]["completion_tokens"], "category": it.get("category"), "text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default="gsm8k,mmlu_pro")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.dir)
    for task in a.tasks.split(","):
        items = json.loads((d / f"items-{task}.json").read_text())
        if a.limit:
            items = items[:a.limit]
        raw = d / f"{a.tag}-{task}.jsonl"
        done = {}
        if raw.exists():
            for line in raw.read_text().splitlines():
                r = json.loads(line)
                if "error" not in r:
                    done[r["id"]] = r
        todo = [it for it in items if it["id"] not in done]
        t0 = time.time()
        with raw.open("a") as f, cf.ThreadPoolExecutor(a.concurrency) as ex:
            for r in ex.map(lambda it: ask(a.url.rstrip("/"), a.model, task, it), todo):
                f.write(json.dumps(r) + "\n")
                f.flush()
                if "error" not in r:
                    done[r["id"]] = r
        rs = [done[it["id"]] for it in items if it["id"] in done]
        n, c = len(rs), sum(r["correct"] for r in rs)
        acc = c / n if n else 0.0
        half = 1.96 * math.sqrt(acc * (1 - acc) / n) if n else 0.0
        cats: dict = {}
        for r in rs:
            if r.get("category"):
                cats.setdefault(r["category"], [0, 0])
                cats[r["category"]][0] += r["correct"]
                cats[r["category"]][1] += 1
        score = {"tag": a.tag, "task": task, "n": n, "correct": c, "acc": round(acc, 4), "ci95": round(half, 4),
                 "unparsed": sum(r["got"] is None for r in rs), "truncated": sum(r["finish"] == "length" for r in rs),
                 "gen_tokens": sum(r["tokens"] for r in rs), "wall_s": round(time.time() - t0, 1),
                 "by_category": cats or None, "settings": {"thinking": "off", "temperature": 0,
                                                            "max_tokens": MAX_TOKENS[task], "concurrency": a.concurrency}}
        (d / f"{a.tag}-{task}.score.json").write_text(json.dumps(score, indent=1))
        print(json.dumps({k: v for k, v in score.items() if k != "by_category"}), flush=True)


if __name__ == "__main__":
    main()
