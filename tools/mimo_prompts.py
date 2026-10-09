"""The task prompts shared by the engine-level DFlash check and the served benchmark (no dependencies)."""

PROMPTS = {
    "coding": "Write a Python function that parses an ISO-8601 duration string such as 'P3DT4H5M' into a "
              "datetime.timedelta. Handle years and months by raising ValueError. Include type hints and a short "
              "docstring, then three pytest tests.",
    "prose": "Write a short story of about 300 words about a lighthouse keeper who finds a message in a bottle "
             "that appears to be written in her own handwriting.",
    "reasoning": "A train leaves city A at 9:00 travelling at 80 km/h towards city B, 360 km away. Another train "
                 "leaves B at 10:00 towards A at 100 km/h. At what time do they meet, and how far from A? Reason "
                 "step by step, then state the answer.",
    "code_edit": "Rename the variable `cnt` to `count` everywhere and add a docstring to each function. Return the "
                 "full updated code only.\n\n```python\ndef tally(words):\n    cnt = {}\n    for w in words:\n"
                 "        cnt[w] = cnt.get(w, 0) + 1\n    return cnt\n\n\ndef top(words, n):\n    cnt = tally(words)\n"
                 "    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))\n    return items[:n]\n\n\n"
                 "def merge(a, b):\n    cnt = dict(a)\n    for k, v in b.items():\n        cnt[k] = cnt.get(k, 0) + v\n"
                 "    return cnt\n```",
}
