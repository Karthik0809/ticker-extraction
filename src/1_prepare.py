"""STEP 1 of 3: get everything the system needs before it can run.

    python src/1_prepare.py

Three jobs, each skipped if its output already exists:

  A. Download the dataset from HuggingFace          -> videoconviction.parquet
  B. Build the ticker -> company name list          -> ticker_names.json
  C. Turn text into number vectors (needs a GPU)    -> emb_*.npy

Takes about two minutes on a machine with a GPU. Run once.

Note on import order: torch has to be imported before pandas. Both ship their
own copy of OpenMP, and on Windows loading pandas first makes the process die
with a segmentation fault the moment the encoder runs. No error message, just
exit code 139. This cost me an afternoon.
"""
import torch  # MUST be first, see docstring
from sentence_transformers import SentenceTransformer

import json
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DATA, load_gazetteer, load_segments

PARQUET = os.path.join(DATA, "videoconviction.parquet")
TICKER_MAP = os.path.join(DATA, "ticker_names.json")
SEC_CACHE = os.path.join(DATA, "sec_tickers.json")
EMB_SEG = os.path.join(DATA, "emb_segments.npy")
EMB_COMP = os.path.join(DATA, "emb_companies.npy")
EMB_IDX = os.path.join(DATA, "emb_company_index.json")

SEC_URL = "https://www.sec.gov/files/company_tickers.json"
# The SEC requires a descriptive User-Agent. This is their access policy.
SEC_UA = {"User-Agent": "academic-research ticker-extraction contact@example.edu"}
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------- A. dataset
def download_dataset():
    """Fetch VideoConviction from HuggingFace and cache it as parquet."""
    if os.path.exists(PARQUET):
        print("[A] dataset already present")
        return
    from datasets import load_dataset
    os.makedirs(DATA, exist_ok=True)
    print("[A] downloading gtfintechlab/VideoConviction ...", flush=True)
    df = load_dataset("gtfintechlab/VideoConviction")["train"].to_pandas()
    df.to_parquet(PARQUET)
    print(f"[A] {len(df)} rows, {df.video_id.nunique()} videos")


# ------------------------------------------------------------- B. ticker list
def build_gazetteer():
    """Resolve every correct ticker to a company name.

    Needed because the dataset ships the symbol and nothing else, while half
    the segments never say the symbol out loud. Identifying those requires the
    company's name, so a symbol -> name mapping has to exist.

    The SEC file lists companies registered NOW; the videos are 2019-2023.
    Anything acquired or bankrupted since is missing, which is about a quarter
    of the correct answers. yfinance recovers some. The rest are hand-written
    in ticker_names_supplement.json.
    """
    if os.path.exists(TICKER_MAP) and os.path.exists(SEC_CACHE):
        print("[B] ticker list already present")
        return

    df = load_segments()
    wanted = sorted(df.ticker_name.unique())
    print(f"[B] resolving {len(wanted)} tickers", flush=True)

    out = {}
    try:
        r = requests.get(SEC_URL, headers=SEC_UA, timeout=30)
        r.raise_for_status()
        # arrives as {"0": {...}, "1": {...}}; the outer keys are row numbers
        sec = {v["ticker"].upper(): v["title"] for v in r.json().values()}
        with open(SEC_CACHE, "w", encoding="utf-8") as f:
            json.dump(sec, f, ensure_ascii=False)
        out = {t: sec[t] for t in wanted if t in sec}
        print(f"[B] SEC: {len(sec)} symbols, {len(out)} of ours resolved")
    except Exception as e:
        print("[B] SEC lookup failed, continuing:", e)

    missing = [t for t in wanted if t not in out]
    if missing:
        import yfinance as yf
        print(f"[B] yfinance for {len(missing)} missing ...", flush=True)
        for i, t in enumerate(missing):
            try:
                info = yf.Ticker(t).get_info()
                name = info.get("longName") or info.get("shortName")
                if name:
                    out[t] = name
            except Exception:
                pass   # fully delisted; handled by the hand-written supplement
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(missing)}", flush=True)

    with open(TICKER_MAP, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    still = [t for t in wanted if t not in out]
    print(f"[B] resolved {len(out)}/{len(wanted)}")
    if still:
        print(f"[B] unresolved, add to ticker_names_supplement.json: {still}")

# -------------------------------------------------------------- C. embeddings
def build_embeddings():
    """Turn segments and company names into vectors of 384 numbers.

    Trained so that similar meanings land near each other, which lets the
    system ask "which company name is closest in meaning to this segment?"
    with one matrix multiply instead of 10,550 string comparisons.

    This is what finds a company when the segment describes it without ever
    naming it.
    """
    if all(os.path.exists(p) for p in (EMB_SEG, EMB_COMP, EMB_IDX)):
        print("[C] embeddings already present")
        return

    df = load_segments()
    gaz = load_gazetteer()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[C] {MODEL} on {dev}", flush=True)
    m = SentenceTransformer(MODEL, device=dev)

    segs = [str(t)[:1200] for t in df.segment_transcript]
    se = m.encode(segs, batch_size=64, convert_to_numpy=True,
                  normalize_embeddings=True, show_progress_bar=False)
    np.save(EMB_SEG, se.astype(np.float32))

    # One vector per ticker, built from its company name. Sorted so the row
    # order is stable; emb_company_index.json records which row is which.
    tickers = sorted(gaz)
    names = [gaz[t] or t for t in tickers]
    ce = m.encode(names, batch_size=256, convert_to_numpy=True,
                  normalize_embeddings=True, show_progress_bar=False)
    np.save(EMB_COMP, ce.astype(np.float32))
    with open(EMB_IDX, "w", encoding="utf-8") as f:
        json.dump({t: i for i, t in enumerate(tickers)}, f)

    print(f"[C] segments {se.shape}, companies {ce.shape}")


def main():
    download_dataset()
    build_gazetteer()
    build_embeddings()
    print("\nready. next:  python src/2_video.py   (optional, adds 2 F1 points)")
    print("        then:  python src/3_run.py")


if __name__ == "__main__":
    main()
