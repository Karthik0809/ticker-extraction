"""EXPERIMENT, step E: score the language-model chooser against the ranker.

    python experiments/e_report.py experiments/out/llm_results_k5.json

Reports three systems on the same 687 out-of-fold segments:

  ranker   the current LambdaMART top-1 (80.06)
  llm      the fine-tuned model's pick from the ranker's top-K
  blend    llm score + beta x ranker score

For the blend, beta is chosen per fold using only the OTHER four folds' results,
so the fold being scored never influences its own setting.
"""
import json
import sys

import numpy as np

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    R = json.load(f)["results"]
tag = "_alias" if "alias" in path else ""
lists_path = path.rsplit("llm_results", 1)[0] + f"rank_lists{tag}.json"
with open(lists_path, encoding="utf-8") as f:
    L = json.load(f)
rank_score = {r["sid"]: {c["tic"]: c["score"] for c in r["cands"]}
              for fold in L["folds"] for r in fold["test"]}

n = len(R)
folds = sorted({r["fold"] for r in R})


def blend_pick(r, beta):
    rs = rank_score[r["sid"]]
    return max(r["llm_logp"], key=lambda t: r["llm_logp"][t] + beta * rs.get(t, -99))


def acc(rows, beta=None):
    if beta is None:
        return np.mean([r["llm_top"] == r["gold"] for r in rows])
    return np.mean([blend_pick(r, beta) == r["gold"] for r in rows])


betas = [0, 0.25, 0.5, 1, 1.5, 2, 3, 5]
blend_correct = 0
chosen = {}
for f in folds:
    others = [r for r in R if r["fold"] != f] or [r for r in R if r["fold"] == f]
    b = max(betas, key=lambda bb: acc(others, bb))
    chosen[f] = b
    blend_correct += sum(blend_pick(r, b) == r["gold"] for r in R if r["fold"] == f)

ranker = np.mean([r["ranker_top"] == r["gold"] for r in R])
llm = acc(R)
blend = blend_correct / n
ceiling = np.mean([r["in_topk"] for r in R])

print(f"{n} out-of-fold segments, folds {folds}")
print(f"  ranker alone       {100 * ranker:6.2f}")
print(f"  LLM chooser        {100 * llm:6.2f}   ({100 * (llm - ranker):+.2f})")
print(f"  blend              {100 * blend:6.2f}   ({100 * (blend - ranker):+.2f})"
      f"   beta per fold {chosen}")
print(f"  ceiling (in top-K) {100 * ceiling:6.2f}")

# where the changes came from
fixed = sum(r["ranker_top"] != r["gold"] and r["llm_top"] == r["gold"] for r in R)
broke = sum(r["ranker_top"] == r["gold"] and r["llm_top"] != r["gold"] for r in R)
print(f"\n  LLM fixed {fixed} ranker mistakes and broke {broke} ranker successes")

# abstention on the LLM's own margin
m = np.array([r["llm_margin"] for r in R])
c = np.array([r["llm_top"] == r["gold"] for r in R])
print("\n  precision if the LLM skips its least confident answers:")
for q in (0, 0.1, 0.2, 0.3):
    keep = m >= np.quantile(m, q)
    print(f"    skip {int(100 * q):2d}%: coverage {100 * keep.mean():5.1f}%  precision {100 * c[keep].mean():5.2f}")
