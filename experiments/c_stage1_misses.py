"""EXPERIMENT, step C: why does the right ticker miss the shortlist?

    python experiments/c_stage1_misses.py

Entity linking can only rescue a miss if the company is actually mentioned
somewhere. This sorts the 51 misses into reasons, so the possible gain from
improving stage 1 is known before building anything.
"""
import os
import pickle
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from core import load_gazetteer, load_segments, name_tokens  # noqa: E402

df = load_segments()
gaz = load_gazetteer()
with open(os.path.join(ROOT, "experiments", "out", "pairs.pkl"), "rb") as f:
    X, y, qid, meta = pickle.load(f)

in_pool = meta.groupby("sid").apply(lambda g: (g.tic == g.gold.iloc[0]).any())
misses = [int(s) for s, ok in in_pool.items() if not ok]
print(f"{len(misses)} segments where the correct ticker is not in the shortlist\n")

rows = []
for s in misses:
    r = df.loc[s]
    t = r.ticker_name
    text = " ".join(str(r.get(c, "")) for c in
                    ("segment_transcript", "transcript", "video_title", "videoDescription")).lower()
    seg = str(r.segment_transcript).lower()
    name = gaz.get(t)
    toks = name_tokens(name) if name else []
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", t):
        why = "label noise (not a ticker)"
    elif name is None:
        why = "no company name known"
    elif any(tok in seg for tok in toks if len(tok) >= 4):
        why = "name IS in the segment"
    elif any(tok in text for tok in toks if len(tok) >= 4):
        why = "name only elsewhere in video"
    elif re.search(r"(?<![a-z])" + re.escape(t.lower()) + r"(?![a-z])", text):
        why = "symbol in text, name not"
    else:
        why = "company never mentioned"
    rows.append((why, t, name, toks))

from collections import Counter
for why, n in Counter(w for w, *_ in rows).most_common():
    print(f"  {n:3d}  {why}")
print()
for why, t, name, toks in sorted(rows):
    print(f"  {why:30s} {t:8s} {str(name)[:40]}")
