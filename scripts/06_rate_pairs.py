"""Step 6: rate every pair 0/1/2 for topical relevance with a language model.

One request per pair, fixed prompt, temperature 0, seed 42. Every raw response is appended to
responses_<run>.jsonl as it arrives, so an interrupted run resumes where it stopped and a finished
run can be re-parsed without calling the API. The prompt and its SHA-256 are saved in the manifest.
Reasoning models do not give identical answers on every call; rate again with --run r2 to measure
test-retest agreement.

Environment (OpenAI-compatible endpoints):
  RATER_API_KEY    API key (never written to disk)
  RATER_BASE_URL   endpoint base URL
  RATER_MODEL      exact model id
  RATER_API_STYLE  "chat" (POST {base}/chat/completions, default) or
                   "responses" (POST {base} itself, OpenAI Responses format)
  RATER_EFFORT     reasoning effort for the responses style (default low)

Input : data/validation/pairs_to_rate.csv
Output: data/validation/ratings/scores_<run>.csv, responses_<run>.jsonl, manifest_<run>.json

Usage: python scripts/06_rate_pairs.py [--run r1] [--workers 4] [--limit N]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrs_topics import config  # noqa: E402

PROMPT = """You are rating the topical relevance of two research abstracts that both use the Health and Retirement Study (HRS).
Judge only the research question/topic from title + abstract; ignore the data source (all are HRS), methods, authors and years.

0 = different research questions. Sharing only a broad domain (e.g. both about depression, retirement,
    wealth, cognition or functional limitations) is 0 when the main exposure and the main outcome both differ.
1 = partly related: the same main exposure OR the same main outcome, but a different main question
2 = strongly related: the same or a very close research question (same main exposure AND outcome)

Abstract A
Title: {a_title}
{a_abstract}

Abstract B
Title: {b_title}
{b_abstract}

Reply with JSON only: {{"score": 0|1|2, "reason": "<at most 15 words>"}}"""
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()
MAX_TOKENS = 4000                      # room for the hidden reasoning of thinking models
SAMPLING = {"temperature": 0, "seed": 42}


def request(prompt, cfg):
    if cfg["style"] == "responses":
        return cfg["base"], {"model": cfg["model"], "input": prompt, "max_output_tokens": MAX_TOKENS,
                             "reasoning": {"effort": cfg["effort"]}, "store": False, **SAMPLING}
    return cfg["base"] + "/chat/completions", {"model": cfg["model"], "max_tokens": MAX_TOKENS,
                                               "messages": [{"role": "user", "content": prompt}], **SAMPLING}


def call(prompt, cfg):
    url, body = request(prompt, cfg)
    headers = {"content-type": "application/json", "authorization": "Bearer " + cfg["key"]}
    for attempt in range(6):
        r = requests.post(url, headers=headers, json=body, timeout=300)
        if r.status_code in (429, 500, 502, 503, 529):        # rate limit / overload: back off
            time.sleep(2 ** attempt * 5)
            continue
        r.raise_for_status()
        j = r.json()
        if cfg["style"] == "responses":
            if j.get("error"):
                raise RuntimeError(j["error"])
            text = "".join(c["text"] for o in j["output"] if o["type"] == "message" for c in o["content"])
        else:
            text = j["choices"][0]["message"]["content"]
        return text, j.get("model", cfg["model"])
    raise RuntimeError(f"gave up after retries: {r.status_code} {r.text[:200]}")


def parse(text):
    m = re.search(r'"score"\s*:\s*([012])', text)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="r1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = {"key": os.environ.get("RATER_API_KEY", "none"), "base": os.environ["RATER_BASE_URL"].rstrip("/"),
           "model": os.environ["RATER_MODEL"], "style": os.environ.get("RATER_API_STYLE", "chat"),
           "effort": os.environ.get("RATER_EFFORT", "low")}
    out = config.VALIDATION / "ratings"
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"responses_{a.run}.jsonl"

    pairs = pd.read_csv(config.VALIDATION / "pairs_to_rate.csv")
    pairs = pairs.head(a.limit) if a.limit else pairs
    done = {json.loads(line)["pair_id"] for line in log.open()} if log.exists() else set()
    todo = pairs[~pairs.pair_id.isin(done)]
    print(f"{len(done)} already rated, {len(todo)} to rate", flush=True)

    def rate(r):
        text, served = call(PROMPT.format(a_title=r.anchor_title, a_abstract=r.anchor_abstract,
                                          b_title=r.candidate_title, b_abstract=r.candidate_abstract), cfg)
        return {"pair_id": r.pair_id, "raw": text, "served_model": served,
                "time": datetime.now(timezone.utc).isoformat()}

    with ThreadPoolExecutor(a.workers) as ex, log.open("a") as f:
        for i, rec in enumerate(ex.map(rate, todo.itertuples()), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, "rated", flush=True)

    recs = [json.loads(line) for line in log.open()]
    scores = pd.DataFrame({"pair_id": [r["pair_id"] for r in recs], "score": [parse(r["raw"]) for r in recs],
                           "raw": [r["raw"] for r in recs]})
    scores.to_csv(out / f"scores_{a.run}.csv", index=False)
    params = {k: v for k, v in request("", cfg)[1].items() if k not in ("input", "messages")}
    manifest = {"run": a.run, "model_requested": cfg["model"],
                "models_served": sorted({r["served_model"] for r in recs}), "api_style": cfg["style"],
                "request_params": params, "prompt_sha256": PROMPT_SHA, "prompt": PROMPT,
                "n_pairs": len(scores), "n_unparsed": int(scores.score.isna().sum()),
                "score_counts": scores.score.value_counts().sort_index().to_dict(),
                "finished": datetime.now(timezone.utc).isoformat()}
    (out / f"manifest_{a.run}.json").write_text(json.dumps(manifest, indent=1, default=str))
    print("scores", manifest["score_counts"], "unparsed", manifest["n_unparsed"])
    if manifest["n_unparsed"]:
        sys.exit("some responses did not parse; see the raw column of scores_*.csv")


if __name__ == "__main__":
    main()
