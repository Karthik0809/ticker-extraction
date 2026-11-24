"""EXPERIMENT, step A: produce the ranker's top-K list for every segment.

    python experiments/a_rank_lists.py

The next idea is to let a language model make the final pick, but only from
the ranker's top few candidates. To train and test that fairly, every list it
sees has to come from a ranker that never saw that segment's video.

So this is NESTED cross-validation:

  outer: the same 5 video-grouped folds as src/3_run.py
         test segments are ranked by a model trained on the other 4 folds
         (3 seeds, identical to the headline run, so top-1 reproduces 80.06)
  inner: inside the 4 training folds, another 4-way split, so the TRAINING
         segments also get lists from a ranker that never saw them

Without the inner split, the training lists would come from rankers that had
seen the outer test fold's answers, which leaks.

Writes experiments/out/rank_lists.json. No torch here: torch and LightGBM
crash when loaded in the same process on Windows.
"""
import importlib.util
import json
import os
import pickle
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from core import WordWeights, load_gazetteer, load_segments  # noqa: E402
from stage1 import CandidateGenerator, CheapRanker            # noqa: E402

spec = importlib.util.spec_from_file_location("run3", os.path.join(ROOT, "src", "3_run.py"))
run3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run3)

OUT = os.path.join(ROOT, "experiments", "out")
os.makedirs(OUT, exist_ok=True)
# --aliases adds Wikidata's alternative company names (step D) to the registry
ALIASES = "--aliases" in sys.argv
TAG = "_alias" if ALIASES else ""
CACHE = os.path.join(OUT, f"pairs{TAG}.pkl")
TOPK = 20


def add_aliases(gaz):
    """Append each ticker's other names to its registry name, so that both the
    name rule in stage 1 and the name features in stage 2 can match "Google"
    to GOOG. Only tickers already in the registry are touched."""
    with open(os.path.join(OUT, "wikidata_aliases.json"), encoding="utf-8") as f:
        alias = json.load(f)
    n = 0
    for t, names in alias.items():
        if t in gaz:
            extra = [a for a in names if a.lower() not in str(gaz[t]).lower()]
            if extra:
                gaz[t] = f"{gaz[t]} | " + " | ".join(extra)
                n += 1
    print(f"  aliases added to {n} registry tickers")
    return gaz


def ranker(seed):
    """Same settings as the headline run."""
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=900,
        learning_rate=0.03, num_leaves=63, min_child_samples=15,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        reg_lambda=1.0, random_state=seed, verbose=-1)


def fit_predict(X, y, qid, train_segs, test_segs, seeds):
    """Train on train_segs, return averaged scores for the rows of test_segs."""
    tr, te = np.isin(qid, train_segs), np.isin(qid, test_segs)
    sizes = pd.Series(qid[tr]).value_counts().sort_index().values
    s = np.zeros(int(te.sum()))
    for seed in seeds:
        m = ranker(seed)
        m.fit(X[tr], y[tr], group=sizes)
        s += m.predict(X[te])
    return te, s / len(seeds)


def main():
    df = load_segments()
    gaz = load_gazetteer()
    if ALIASES:
        gaz = add_aliases(gaz)

    # ---- build or load the (segment, candidate) feature table
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            X, y, qid, meta = pickle.load(f)
        print(f"loaded cached pairs: {len(X)} rows")
    else:
        corpus = list(df.segment_transcript) + list(df.videoDescription.fillna(""))
        w = WordWeights(gaz, corpus=corpus)
        X, y, qid, meta = run3.build(df, CandidateGenerator(gaz, w), CheapRanker(gaz, w), w,
                                     dense=run3.DenseScorer(), gaz=gaz, ocr=run3.load_ocr())
        with open(CACHE, "wb") as f:
            pickle.dump((X, y, qid, meta), f)
        print(f"built {len(X)} pairs")

    n_seg = len(np.unique(qid))
    groups = meta.groupby("sid").video_id.first().values
    gold = meta.groupby("sid").gold.first().values

    lists = {}
    outer = GroupKFold(n_splits=5)
    for f, (tr_seg, te_seg) in enumerate(outer.split(np.arange(n_seg), groups=groups)):
        # outer test: 3 seeds, the headline protocol
        te, s = fit_predict(X, y, qid, tr_seg, te_seg, seeds=(0, 1, 2))
        part = meta[te].assign(score=s)
        for sid, g in part.groupby("sid"):
            lists[int(sid)] = {"fold": f, "role": "test", "scores": g}

        # inner: lists for the training segments, from rankers blind to them
        inner = GroupKFold(n_splits=4)
        for itr, ite in inner.split(tr_seg, groups=groups[tr_seg]):
            te_i, s_i = fit_predict(X, y, qid, tr_seg[itr], tr_seg[ite], seeds=(0,))
            part = meta[te_i].assign(score=s_i)
            for sid, g in part.groupby("sid"):
                lists.setdefault(("train", f, int(sid)), g)
        print(f"  fold {f + 1}/5 done", flush=True)

    # ---- serialise: per outer fold, train lists and test lists
    def top(g):
        g = g.sort_values("score", ascending=False)
        return [{"tic": t, "score": round(float(sc), 4)} for t, sc in
                zip(g.tic.head(TOPK), g.score.head(TOPK))]

    out = {"topk": TOPK, "folds": []}
    for f, (tr_seg, te_seg) in enumerate(outer.split(np.arange(n_seg), groups=groups)):
        out["folds"].append({
            "train": [{"sid": int(s), "gold": gold[s], "cands": top(lists[("train", f, int(s))])}
                      for s in tr_seg],
            "test": [{"sid": int(s), "gold": gold[s], "cands": top(lists[int(s)]["scores"])}
                     for s in te_seg],
        })
    with open(os.path.join(OUT, f"rank_lists{TAG}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)

    in_pool = meta.groupby("sid").apply(lambda g: (g.tic == g.gold.iloc[0]).any())
    print(f"\nSTAGE 1: correct answer in the shortlist: {100 * in_pool.mean():.2f}% "
          f"({int(in_pool.sum())} of {n_seg})")

    # ---- recall@K on the outer test lists
    ranks = []
    for fold in out["folds"]:
        for r in fold["test"]:
            tics = [c["tic"] for c in r["cands"]]
            ranks.append(tics.index(r["gold"]) + 1 if r["gold"] in tics else 999)
    ranks = np.array(ranks)
    print("\nOUTER TEST, out of 687 segments")
    for k in (1, 2, 3, 5, 10, 20):
        print(f"  correct answer in ranker's top {k:>2}: {100 * (ranks <= k).mean():5.2f}%")


if __name__ == "__main__":
    main()
