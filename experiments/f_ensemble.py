"""EXPERIMENT, step F: average several fine-tuned models.

    python experiments/f_ensemble.py out/llm_results_alias_k5.json out/llm_results_alias_screen_k5.json ...

Each run scored the same candidates for the same 687 out-of-fold segments. For
every candidate, average the log-probabilities across runs and pick the
highest. Models trained separately make different mistakes, so the average is
usually steadier than any one of them. Nothing here is tuned on test data.
"""
import json
import sys

import numpy as np

runs = []
for p in sys.argv[1:]:
    with open(p, encoding="utf-8") as f:
        runs.append({r["sid"]: r for r in json.load(f)["results"]})

sids = sorted(set.intersection(*[set(r) for r in runs]))
gold = {s: runs[0][s]["gold"] for s in sids}

for p, run in zip(sys.argv[1:], runs):
    print(f"  {np.mean([run[s]['llm_top'] == gold[s] for s in sids]) * 100:6.2f}  {p}")

correct, margins = [], []
for s in sids:
    tics = runs[0][s]["llm_logp"].keys()
    avg = {t: np.mean([run[s]["llm_logp"][t] for run in runs]) for t in tics}
    order = sorted(avg, key=avg.get, reverse=True)
    correct.append(order[0] == gold[s])
    margins.append(avg[order[0]] - avg[order[1]] if len(order) > 1 else 99)

correct, margins = np.array(correct), np.array(margins)
print(f"\n  ENSEMBLE of {len(runs)}: {100 * correct.mean():.2f}  over {len(sids)} segments")
for q in (0.1, 0.2, 0.3):
    keep = margins >= np.quantile(margins, q)
    print(f"    skip {int(100 * q)}%: coverage {100 * keep.mean():5.1f}%  precision {100 * correct[keep].mean():5.2f}")
