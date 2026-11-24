"""STEP 3 of 3: run the system and score it.

    python src/3_run.py

WHAT HAPPENS, IN ORDER
----------------------
  1. Load the 687 segments, the 10,550-ticker list and the embeddings.
  2. Learn word weights from the transcripts (how much each word narrows down
     which company is meant).
  3. STAGE 1 - for each video, collect every ticker it could be about.
     10,550 -> about 93. The right answer survives this 92.6% of the time,
     and that number is the hard ceiling on everything below.
  4. Turn each (segment, candidate) pair into 37 numbers. ~100,000 rows.
  5. STAGE 2 - train a ranking model to put the right candidate first.
  6. Score it, and print the result against the paper's Table 6.

HOW IT IS SCORED HONESTLY
-------------------------
Five-fold cross-validation, split by VIDEO, not by segment. Segments from one
video share a candidate pool and most of their text, so a random split would
put near-copies in both training and test and the score would be inflated.

Each fold is trained three times with different random seeds and the results
averaged, because a single seed moves the score by about a point and the gaps
being claimed against the published table are about that size.

Takes roughly 8 minutes.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb

# LightGBM is fitted with feature names but predicts on a plain array, so
# sklearn prints the same notice once per fold per seed. Harmless, and it
# buries the result, so silence just that one.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DATA, RESULTS, WordWeights, load_gazetteer, load_segments
from stage1 import CandidateGenerator, CheapRanker
from stage2 import FEATURE_NAMES, pair_features

PAPER = {
    "Mistral-7B": 63.73, "Qwen2.5-7B": 65.57, "Claude 3.5 Haiku": 68.18,
    "Gemini 2.0 Flash": 70.05, "Mixtral-8x22B": 70.12, "DeepSeek-R1": 71.18,
    "Llama-3.1-8B": 71.86, "Gemini 1.5 Flash": 72.09, "Gemini 2.0 Pro": 74.66,
    "Claude 3.5 Sonnet": 75.90, "Llama-3.1-70B": 76.51, "Gemini 1.5 Pro": 76.56,
    "GPT-4o": 76.76, "DeepSeek-V3": 77.56, "Llama-3.1-405B": 78.60,
    "Qwen2.5-72B (best text-only)": 79.20, "Gemini 2.0 Flash (MM)": 82.24,
    "GPT-4o (MM)": 83.47, "Gemini 1.5 Pro (MM)": 86.01,
    "Gemini 2.0 Pro (MM)": 86.04, "Gemini 1.5 Flash (MM, best)": 86.23,
}
MAX_CAND = 180   # keep the top-N by the cheap score; recall cost is reported
DENSE_ADD = 60   # dense-retrieved companies added to each video's pool

class DenseScorer:
    """Cached bi-encoder similarities, loaded with numpy.

    Embeddings are produced by 11_dense_cache.py in a separate process. torch
    and LightGBM cannot share an interpreter on Windows (duplicate OpenMP
    runtimes, segfault at exit 139), so this side never imports torch.
    """

    def __init__(self):
        self.seg = np.load(os.path.join(DATA, "emb_segments.npy"))
        self.comp = np.load(os.path.join(DATA, "emb_companies.npy"))
        with open(os.path.join(DATA, "emb_company_index.json"), encoding="utf-8") as f:
            self.idx = json.load(f)
        print(f"  dense cache: {self.seg.shape[0]} segments x "
              f"{self.comp.shape[0]} companies", flush=True)

    def topk(self, si, k):
        """The k companies whose names are nearest this segment's embedding.

        Stage 1 is otherwise entirely string-based: a company enters the pool
        only if its symbol or some token of its name literally appears. That
        misses segments where the company is described rather than named, which
        is the same blind spot a lexical retriever has against paraphrase. This
        is the dense half, added at the candidate stage where recall is what
        matters.
        """
        sc = self.comp @ self.seg[si]
        idx = np.argpartition(-sc, k)[:k]
        inv = getattr(self, "_inv", None)
        if inv is None:
            inv = self._inv = {v: kk for kk, v in self.idx.items()}
        return [inv[int(i)] for i in idx if int(i) in inv]

    def sims(self, si, tickers):
        """{ticker: (cosine, rank within this candidate list)}"""
        rows = [(self.idx[t], t) for t in tickers if t in self.idx]
        if not rows:
            return {}
        ii = np.fromiter((i for i, _ in rows), dtype=np.int64, count=len(rows))
        sc = self.comp[ii] @ self.seg[si]
        order = np.argsort(-sc)
        return {rows[j][1]: (float(sc[j]), float(rk)) for rk, j in enumerate(order)}


def load_ocr():
    """OCR evidence per segment, keyed video_id_start. Empty if 04/05 not run."""
    p = os.path.join(DATA, "ocr_evidence.json")
    if not os.path.exists(p):
        print("  no OCR evidence (run 04_video_frames.py then 05_frame_ocr.py)")
        return {}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    print(f"  OCR evidence for {len(d)} segments")
    return d


def build(df, gen, cheap, weights, dense=None, gaz=None, ocr=None):
    """One row per (segment, candidate) pair."""
    X, y, qid, meta = [], [], [], []

    # ordinal position of each segment within its own video, by start time
    order = (df.sort_values(["video_id", "start"])
               .groupby("video_id").cumcount())
    counts = df.groupby("video_id").video_id.transform("size")
    seg_pos_map = (order / (counts - 1).clip(lower=1)).reindex(df.index).fillna(0.0)

    for si, (_, r) in enumerate(df.iterrows()):
        pool = gen.for_video(r)
        oc = (ocr or {}).get(f"{r.video_id}_{float(r.start):.2f}")
        if oc:
            # a symbol read off the screen is a candidate in its own right; this
            # is the only source that can see a ticker nobody ever says
            for sym in (oc.get("symbols") or {}):
                pool.setdefault(sym, set()).add("ocr/screen")
        if dense is not None and DENSE_ADD:
            for t in dense.topk(si, DENSE_ADD):
                pool.setdefault(t, set()).add("dense/sim")
        ranked = cheap.rank(r.segment_transcript, pool)[:MAX_CAND]
        if not ranked:
            continue

        sims = dense.sims(si, [t for _, t, _ in ranked]) if dense is not None else {}

        for rank, (score, tic, _) in enumerate(ranked):
            ds, dr = sims.get(tic, (0.0, -1.0))
            X.append(pair_features(r.segment_transcript, tic, pool.get(tic, set()),
                                   weights, len(pool), score, rank,
                                   desc=r.get("videoDescription", ""),
                                   title=r.get("video_title", ""),
                                   dense_sim=ds, dense_rank=dr,
                                   seg_pos=float(seg_pos_map.iloc[si]),
                                   full_text=r.get("transcript", ""),
                                   ocr=oc))
            y.append(1 if tic == r.ticker_name else 0)
            qid.append(si)
            meta.append((si, tic, r.video_id, r.ticker_name))
    return (np.asarray(X, dtype=np.float32), np.asarray(y),
            np.asarray(qid), pd.DataFrame(meta, columns=["sid", "tic", "video_id", "gold"]))


def main():
    """Train, evaluate and report, in five steps.

      1. Load the data and build the word weights.
      2. Turn every (segment, candidate) pair into 37 features.
      3. Train the ranker five times, holding out a fifth of the
         videos each time, so every segment is scored by a model that
         never saw its video.
      4. Print accuracy, the abstention curve and the scoreboard.
      5. Write the predictions and scores to results/.
    """
    # --- 1. data and weights ------------------------------------
    df = load_segments()
    gaz = load_gazetteer()

    # Corpus IDF is fitted over transcripts AND descriptions: a word that is
    # common in either is not distinctive in this domain.
    corpus = list(df.segment_transcript) + list(df.videoDescription.fillna(""))
    weights = WordWeights(gaz, corpus=corpus)

    gen = CandidateGenerator(gaz, weights)   # stage 1: which stocks could this video mean
    cheap = CheapRanker(gaz, weights)      # cheap pre-ordering; its rank is a feature

    # --- 2. features --------------------------------------------
    print(f"{len(df)} segments, building pair features ...", flush=True)
    dense = DenseScorer()
    ocr = load_ocr()
    X, y, qid, meta = build(df, gen, cheap, weights, dense=dense, gaz=gaz, ocr=ocr)
    n_seg = len(np.unique(qid))
    has_gold = meta.groupby("sid").apply(lambda g: (g.tic == g.gold.iloc[0]).any())
    print(f"  {len(X)} pairs over {n_seg} segments, {X.shape[1]} features")
    print(f"  correct answer in the top-{MAX_CAND} candidate list: "
          f"{100*has_gold.mean():.1f}%   <- ceiling for the ranker\n", flush=True)

    # One video id and one correct answer per segment, for grouping and scoring.
    groups = meta.groupby("sid").video_id.first().values
    seg_gold = meta.groupby("sid").gold.first().values

    pred_top = np.empty(n_seg, dtype=object)   # chosen ticker per segment
    pred_score = np.zeros(n_seg)               # margin over 2nd place, used for abstention
    importances = np.zeros(X.shape[1])

    # --- 3. train and predict, out of fold ----------------------
    # GroupKFold splits on video, never on segment. Segments from one video
    # share a candidate pool, a description and a full transcript, so a random
    # split would place near-duplicates in train and test and inflate the score.
    gkf = GroupKFold(n_splits=5)
    for fold, (tr_seg, te_seg) in enumerate(gkf.split(np.arange(n_seg), groups=groups), 1):
        # tr/te index PAIRS, tr_seg/te_seg index SEGMENTS; qid maps between them.
        tr = np.isin(qid, tr_seg)
        te = np.isin(qid, te_seg)
        # LightGBM's ranker takes candidate-list lengths in query order, so it
        # knows where one segment's list ends and the next begins.
        tr_sizes = pd.Series(qid[tr]).value_counts().sort_index().values
        # Average over seeds. LambdaMART on 687 queries is high variance and a
        # single seed moves the score by around a point, which is the same size
        # as the differences being claimed against the published table.
        s = np.zeros(int(te.sum()))
        for seed in (0, 1, 2):
            model = lgb.LGBMRanker(
                objective="lambdarank", metric="ndcg", n_estimators=900,
                learning_rate=0.03, num_leaves=63, min_child_samples=15,
                subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
                reg_lambda=1.0, random_state=seed, verbose=-1,
            )
            model.fit(X[tr], y[tr], group=tr_sizes)
            s += model.predict(X[te])
            importances += model.feature_importances_ / 3.0
        s /= 3.0

        te_df = meta[te].copy()
        te_df["score"] = s
        for sid, g in te_df.groupby("sid"):
            best = g.loc[g.score.idxmax()]
            pred_top[sid] = best.tic
            pred_score[sid] = best.score - g.score.nlargest(2).iloc[-1] if len(g) > 1 else 1.0
        print(f"  fold {fold}: {len(te_seg)} test segments", flush=True)

    # --- 4. report ----------------------------------------------
    pred = pd.Series(pred_top)
    gold = pd.Series(seg_gold)
    correct = (pred == gold)
    acc = correct.mean()

    print()
    print("=" * 72)
    print("OUT-OF-FOLD RESULTS  (every segment predicted by a model that never")
    print("saw any segment from its video)")
    print("=" * 72)
    print(f"  accuracy / recall / F1 at full coverage : {100*acc:5.2f}")
    print(f"  conditional on it being in the list     : "
          f"{100*correct[has_gold.values].mean():5.2f}")

    # abstention: margin between first and second
    print("\n  precision/coverage if we abstain on a thin margin:")
    for q in [0.0, 0.1, 0.2, 0.3]:
        thr = np.quantile(pred_score, q)
        keep = pred_score >= thr
        print(f"    drop bottom {int(100*q):2d}% by margin: coverage {100*keep.mean():5.1f}%  "
              f"precision {100*correct[keep].mean():5.2f}%")

    print()
    print("=" * 72)
    print("TASK T SCOREBOARD  (paper Table 6, segmented input, F1)")
    print("=" * 72)
    score = 100 * acc
    for k, v in sorted(PAPER.items(), key=lambda kv: kv[1]):
        print(f"  {'>>' if v < score else '  '} {k:30s} {v:5.2f}")
    print(f"  == {'THIS SYSTEM (learned ranker)':30s} {score:5.2f}")
    beaten = sorted([k for k, v in PAPER.items() if score > v], key=lambda k: -PAPER[k])
    print(f"\n  beats {len(beaten)}/{len(PAPER)} published results")
    if beaten:
        print(f"  best beaten: {beaten[0]} ({PAPER[beaten[0]]:.2f})")

    print()
    print("=" * 72)
    print("WHAT THE MODEL LEARNED TO USE")
    print("=" * 72)
    imp = sorted(zip(FEATURE_NAMES, importances / 5), key=lambda kv: -kv[1])
    for k, v in imp[:12]:
        print(f"  {k:18s} {v:7.1f}")

    # --- 5. save --------------------------------------------------
    out = pd.DataFrame({"gold": gold, "pred": pred, "margin": pred_score,
                        "correct": correct, "video_id": groups})
    out.to_csv(os.path.join(RESULTS, "task_t_predictions.csv"), index=False)
    with open(os.path.join(RESULTS, "task_t_scores.json"), "w", encoding="utf-8") as f:
        json.dump({"n_segments": int(n_seg), "n_pairs": int(len(X)),
                   "max_candidates": MAX_CAND,
                   "list_recall": float(has_gold.mean()),
                   "accuracy_f1": float(acc),
                   "cond_on_list": float(correct[has_gold.values].mean()),
                   "beats": beaten,
                   "feature_importance": {k: float(v) for k, v in imp}}, f, indent=2)
    print(f"\nwrote {RESULTS}/task_t_predictions.csv and task_t_scores.json")


if __name__ == "__main__":
    main()
