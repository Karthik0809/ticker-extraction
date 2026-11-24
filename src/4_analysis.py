"""The two measurements the whole project rests on.

Everything in Part 1 of the write-up and slides 4 and 5 of the deck comes from
this file. It builds nothing and trains nothing; it only measures the data.

    python src/4_analysis.py

A. Is the correct ticker present in the transcript at all, in any form?
B. Does its absence correlate with how big the company is?

Run this when someone asks where those numbers came from.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DATA, RESULTS, load_segments

CAP_CACHE = os.path.join(DATA, "market_caps.json")
RNG = np.random.default_rng(0)


# ====================================================== A. is the answer there?
def surface_forms(text, ticker):
    """Which literal forms of the symbol occur in the text."""
    hits = set()
    # $VERI  - the social-media convention
    if re.search(r"\$" + re.escape(ticker) + r"(?![A-Za-z0-9])", text, re.I):
        hits.add("cashtag")
    # VERI  - written plainly
    if re.search(r"(?<![A-Za-z0-9])" + re.escape(ticker) + r"(?![A-Za-z0-9])", text, re.I):
        hits.add("plain")
    # V-E-R-I  - dictated letter by letter
    if len(ticker) >= 2:
        pat = r"[\s\-\.]{1,2}".join(re.escape(c) for c in ticker)
        if re.search(r"(?<![A-Za-z0-9])" + pat + r"(?![A-Za-z0-9])", text, re.I):
            hits.add("dictated")
    return hits


def part_a(df):
    """Count how often the correct symbol is written in the transcript,
    in any of the three forms. Returns one row per segment."""
    print("=" * 70)
    print("A.  IS THE CORRECT TICKER PRESENT IN THE TRANSCRIPT AT ALL?")
    print("=" * 70)
    rows = []
    for _, r in df.iterrows():
        h = surface_forms(r.segment_transcript, r.ticker_name)
        rows.append({"ticker": r.ticker_name,
                     "cashtag": "cashtag" in h,
                     "plain": "plain" in h,
                     "dictated": "dictated" in h})
    ev = pd.DataFrame(rows)
    n = len(ev)
    for col, label in [("cashtag", "Cashtag  $TICK"),
                       ("plain", "Bare token  TICK"),
                       ("dictated", "Dictated  T-I-C-K")]:
        c = int(ev[col].sum())
        print(f"  {label:24s} {c:4d}   {100*c/n:5.1f}%")
    any_form = ev[["cashtag", "plain", "dictated"]].any(axis=1)
    print(f"  {'-- any form at all':24s} {int(any_form.sum()):4d}   {100*any_form.mean():5.1f}%")
    print(f"  {'-- NONE':24s} {int((~any_form).sum()):4d}   {100*(~any_form).mean():5.1f}%  <<<")

    ev["visible"] = any_form.astype(int)

    # the headline examples
    print("\n  tickers appearing in 5+ segments that are NEVER spoken:")
    g = ev.groupby("ticker").agg(n=("visible", "size"), vis=("visible", "sum"))
    for t, row in g[(g.n >= 5) & (g.vis == 0)].sort_values("n", ascending=False).head(6).iterrows():
        print(f"    {t:6s} {int(row.n):3d} segments, spoken 0 times")
    return ev


# ============================================ B. does absence track company size?
def fetch_caps(tickers):
    """{ticker: market cap} from Yahoo, cached to disk.

    Cached because caps move daily, and a figure quoted in the write-up
    has to stay the same when the script is run again.
    """
    if os.path.exists(CAP_CACHE):
        with open(CAP_CACHE, encoding="utf-8") as f:
            return json.load(f)
    import yfinance as yf
    caps = {}
    for i, t in enumerate(tickers):
        try:
            mc = yf.Ticker(t).get_info().get("marketCap")
            if mc:
                caps[t] = float(mc)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"    yfinance {i+1}/{len(tickers)}", flush=True)
    with open(CAP_CACHE, "w", encoding="utf-8") as f:
        json.dump(caps, f)
    return caps


def bootstrap_diff(a, b, n=5000):
    """95% interval on mean(a) - mean(b), by resampling each group."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [RNG.choice(a, len(a), True).mean() - RNG.choice(b, len(b), True).mean()
         for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def part_b(ev):
    """Test whether the symbol goes missing more often for big companies.

    Three ways, because one test alone would not be convincing: a rank
    correlation over tickers, a bootstrap interval on the top-versus-
    bottom-third gap, and a Mann-Whitney U on the same split.
    """
    from scipy.stats import spearmanr, mannwhitneyu
    print()
    print("=" * 70)
    print("B.  DOES ABSENCE TRACK HOW BIG THE COMPANY IS?")
    print("=" * 70)
    caps = fetch_caps(sorted(ev.ticker.unique()))
    live = ev[ev.ticker.map(caps).notna()].copy()
    live["mcap"] = live.ticker.map(caps)
    print(f"  {len(live)}/{len(ev)} segments have a current market cap "
          f"({live.ticker.nunique()} tickers)")
    print("  delisted companies drop out here, and they skew small, so this is")
    print("  a conservative estimate of the effect.\n")

    for lo, hi, label in [(0, 3e8, "under $300M"), (3e8, 2e9, "$300M to $2B"),
                          (2e9, 1e10, "$2B to $10B"), (1e10, 2e11, "$10B to $200B"),
                          (2e11, 1e99, "over $200B")]:
        g = live[(live.mcap >= lo) & (live.mcap < hi)]
        if len(g) >= 5:
            print(f"  {label:16s} n={len(g):4d}   ticker spoken {100*g.visible.mean():5.1f}%")

    # per ticker, so one heavily-covered stock cannot drive the result
    per = live.groupby("ticker").agg(visible=("visible", "mean"),
                                     mcap=("mcap", "first")).reset_index()
    per["log_cap"] = np.log10(per.mcap)
    rho, p = spearmanr(per.log_cap, per.visible)
    print(f"\n  Spearman rho = {rho:+.3f}   p = {p:.2e}   over {len(per)} tickers")

    per = per.sort_values("log_cap")
    k = max(5, len(per) // 3)
    small, big = per.head(k).visible.values, per.tail(k).visible.values
    diff, lo, hi = bootstrap_diff(small, big)
    _, pu = mannwhitneyu(small, big, alternative="greater")
    print(f"  smallest third: {100*small.mean():5.1f}% spoken")
    print(f"  largest third:  {100*big.mean():5.1f}% spoken")
    print(f"  difference      {100*diff:+5.1f} points, 95% CI [{100*lo:+.1f}, {100*hi:+.1f}]")
    print(f"  Mann-Whitney one-sided p = {pu:.2e}")
    verdict = "SUPPORTED" if (lo > 0 and pu < 0.05) else "NOT SUPPORTED"
    print(f"\n  Hypothesis {verdict}: bigger company, less likely the ticker is spoken.")
    return {"spearman_rho": float(rho), "spearman_p": float(p),
            "tercile_diff": float(diff), "ci95": [float(lo), float(hi)],
            "mannwhitney_p": float(pu), "verdict": verdict}


def main():
    """Run both measurements and save them to results/analysis.json."""
    df = load_segments()
    print(f"{len(df)} annotated segments\n")
    ev = part_a(df)
    stats = part_b(ev)
    n = len(ev)
    none = int((~ev[["cashtag", "plain", "dictated"]].any(axis=1)).sum())
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_segments": n, "no_symbol_anywhere": none,
                   "no_symbol_share": none / n, **stats}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
