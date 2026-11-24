"""EXPERIMENT, step G: only ask the LLM when the ranker is unsure.

    python experiments/g_gate.py out/llm_results_alias_k5.json

The LLM fixes many ranker mistakes but also breaks a few of its correct
answers. When the ranker is very confident (a big gap between its 1st and 2nd
choice), it is usually right, so keep its answer and only let the LLM decide
the close calls.

The confidence threshold for each fold is picked using the other four folds
only, so the fold being scored never influences its own threshold.
"""
import json
import sys

import numpy as np

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    R = json.load(f)["results"]
tag = "_alias" if "alias" in path else ""
with open(path.rsplit("llm_results", 1)[0] + f"rank_lists{tag}.json", encoding="utf-8") as f:
    L = json.load(f)
margin = {}
for fold in L["folds"]:
    for r in fold["test"]:
        s = [c["score"] for c in r["cands"]]
        margin[r["sid"]] = s[0] - s[1] if len(s) > 1 else 99


def pick(r, thr):
    return r["ranker_top"] if margin[r["sid"]] >= thr else r["llm_top"]


def acc(rows, thr):
    return np.mean([pick(r, thr) == r["gold"] for r in rows])


grid = sorted(set(np.quantile(list(margin.values()), np.linspace(0, 1, 41))))
grid.append(99)
correct = 0
chosen = {}
for f in sorted({r["fold"] for r in R}):
    others = [r for r in R if r["fold"] != f]
    thr = max(grid, key=lambda t: acc(others, t))
    chosen[f] = round(float(thr), 2)
    correct += sum(pick(r, thr) == r["gold"] for r in R if r["fold"] == f)

n = len(R)
print(f"  ranker alone   {100 * np.mean([r['ranker_top'] == r['gold'] for r in R]):6.2f}")
print(f"  LLM alone      {100 * np.mean([r['llm_top'] == r['gold'] for r in R]):6.2f}")
print(f"  gated          {100 * correct / n:6.2f}   thresholds per fold {chosen}")
