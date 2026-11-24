"""STEP 2 of 3 (optional): read the ticker off the video itself.

    python src/2_video.py

WHY
---
47.9% of segments never say the ticker out loud. For those the answer simply
is not in the transcript.

But the benchmark paper reports that models given the video score 86.23 while
text-only models reach 79.20, and it says why:

    "stock charts in videos displaying ticker names"

So the symbol IS there, written on screen. Finfluencers screen-record a
finance site while they talk, and its header reads, literally:

    Veritone, Inc. (VERI)  ·  NasdaqGS  ·  8.15 +0.44 (+5.71%)

Reading that needs OCR, not a vision model. OCR is free and runs on a GPU.

WHAT THIS DOES
--------------
  A. Pull 6 frames per segment from YouTube        -> data/frames/*.jpg
  B. OCR them and record what was found            -> data/ocr_evidence.json

Nothing is downloaded whole: yt-dlp resolves a direct stream URL and ffmpeg
range-requests the bytes near each timestamp. About 40 minutes for A, 15 for B.

Resumable. Frames already on disk are skipped, so interrupting costs nothing.

Skip this entire script and the system scores 78.02 instead of 80.06. It is
separate because 27% of the videos have been deleted since 2019 and it depends
on what is still online.
"""
import torch  # MUST precede pandas, see 1_prepare.py
import easyocr

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DATA, WORD_TICKERS, load_gazetteer, load_segments

FRAMES = os.path.join(DATA, "frames")
OCR_OUT = os.path.join(DATA, "ocr_evidence.json")
DEAD = os.path.join(DATA, "dead_videos.json")

N_FRAMES = 6
VIDEO_FMT = "136/134/best[height<=720]"   # 720p then 360p; enough for OCR
MIN_CONF = 0.35                           # discard OCR reads below this

# Chart headers write the symbol two ways, and both are far stronger evidence
# than a capitalised word that happens to match. A bookshelf behind a presenter
# produces plenty of the latter.
PARENS = re.compile(r"\(([A-Z]{1,5})\)")                 # "(VERI)"
EXCHANGE = re.compile(r"(?:NASDAQ|NYSE|AMEX|OTC)\s*[:\-]\s*([A-Z]{1,5})\b", re.I)
BARE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,5})(?![A-Za-z0-9])")

# ============================================================ A. GRAB FRAMES
class UrlResolver:
    """yt-dlp stream URLs, one per video, cached on disk."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cache = {}           # not persisted: these URLs expire in hours
        self.dead = set()
        if os.path.exists(DEAD):
            with open(DEAD, encoding="utf-8") as f:
                self.dead = set(json.load(f))

    def save(self):
        with self.lock:
            with open(DEAD, "w", encoding="utf-8") as f:
                json.dump(sorted(self.dead), f, indent=1)

    def get(self, vid):
        """Stream URL, or None if the video is gone.

        Thread-safe. The subprocess runs outside the lock so workers do not
        serialise on the slowest part of the job.
        """
        with self.lock:
            if vid in self.dead:
                return None
            if vid in self.cache:
                return self.cache[vid]
        try:
            r = subprocess.run(
                ["python", "-m", "yt_dlp", "--js-runtimes", "node",
                 "-f", VIDEO_FMT, "-g", f"https://www.youtube.com/watch?v={vid}"],
                capture_output=True, text=True, timeout=180)
        except Exception:
            return None
        if r.returncode != 0 or not r.stdout.strip():
            # deleted, private, region-locked or age-gated. Recorded rather
            # than retried, since these videos are years old.
            with self.lock:
                self.dead.add(vid)
            return None
        url = r.stdout.strip().split("\n")[0]
        with self.lock:
            self.cache[vid] = url
        return url


def grab_frame(ff, url, ts, out):
    """Write one JPEG from `url` at time `ts`. True if it worked.

    `-ss` goes BEFORE `-i` so ffmpeg seeks before opening the stream, issuing
    an HTTP range request for the bytes near that timestamp instead of reading
    from the start. That is what makes this seconds per frame, not minutes per
    video.

    The 2000-byte floor rejects the near-empty files ffmpeg writes when a seek
    lands past the end of the stream.
    """
    if os.path.exists(out) and os.path.getsize(out) > 2000:
        return True                     # already have it; makes this resumable
    try:
        r = subprocess.run(
            [ff, "-y", "-loglevel", "error", "-ss", f"{ts:.2f}", "-i", url,
             "-frames:v", "1", "-q:v", "3", out],
            capture_output=True, timeout=180)
        return r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 2000
    except Exception:
        return False


def fetch_frames(df, workers, n_frames):
    """Save N still frames per segment, spread across its time range.

    One thread per video rather than per segment, because resolving the
    stream URL is the slow part and segments of one video share it.
    Frames already on disk are skipped, so this is safe to interrupt.
    """
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(FRAMES, exist_ok=True)
    res = UrlResolver()

    # group by video so each stream URL is resolved once, not once per segment
    byvid = defaultdict(list)
    for _, r in df.iterrows():
        byvid[r.video_id].append((float(r.start), float(r.end)))

    print(f"[A] {len(df)} segments over {len(byvid)} videos, "
          f"{n_frames} frames each, {workers} workers", flush=True)
    done = [0, 0, 0]    # videos, frames, unavailable
    lock = threading.Lock()

    def work(vid):
        """Fetch every frame for one video. Returns nothing; counts
        go into `done` under the lock."""
        url = res.get(vid)
        if url is None:
            with lock:
                done[2] += 1
            return
        got = 0
        for s, e in byvid[vid]:
            for i, ts in enumerate(np.linspace(s + 1.0, max(s + 1.5, e - 1.0), n_frames)):
                out = os.path.join(FRAMES, f"{vid}_{s:.2f}_{i}.jpg")
                if grab_frame(ff, url, float(ts), out):
                    got += 1
        with lock:
            done[0] += 1
            done[1] += got
            if done[0] % 10 == 0:
                res.save()
                print(f"    {done[0]}/{len(byvid)} videos, {done[1]} frames, "
                      f"{done[2]} unavailable", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, list(byvid)))
    res.save()
    print(f"[A] {done[0]} videos done, {done[2]} unavailable, {done[1]} frames")


# ================================================================== B. OCR
def read_frames(df, gaz):
    """OCR every frame and record which tickers appeared, and how."""
    # map frame filenames back to their segment
    want = {f"{r.video_id}_{float(r.start):.2f}" for _, r in df.iterrows()}
    files = defaultdict(list)
    for fn in os.listdir(FRAMES):
        if fn.endswith(".jpg"):
            key = fn.rsplit("_", 1)[0]
            if key in want:
                files[key].append(os.path.join(FRAMES, fn))

    gpu = torch.cuda.is_available()
    print(f"[B] {len(files)} segments, {sum(len(v) for v in files.values())} "
          f"frames, easyocr on {'gpu' if gpu else 'cpu'}", flush=True)
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    out = {}
    for i, (key, paths) in enumerate(sorted(files.items())):
        # For this segment: which symbols were seen, in how many
        # frames, at what confidence, and matched by which pattern.
        syms = defaultdict(lambda: {"frames": 0, "conf": 0.0, "how": set()})
        chunks = []
        for p in sorted(paths):
            try:
                res = reader.readtext(p, detail=1)
            except Exception:
                continue
            # Join the confident boxes into one line. Layout is lost,
            # which is fine: the patterns below only need adjacency.
            line = " ".join(t for _, t, c in res if c >= MIN_CONF)
            if not line:
                continue
            chunks.append(line)
            seen = defaultdict(set)
            for rx, how in ((PARENS, "parens"), (EXCHANGE, "exchange"), (BARE, "bare")):
                for m in rx.finditer(line):
                    s = m.group(1).upper()
                    if s not in gaz:
                        continue
                    if how == "bare" and s in WORD_TICKERS:
                        continue
                    seen[s].add(how)
            best = max((c for _, _, c in res), default=0.0)
            for s, hows in seen.items():
                syms[s]["frames"] += 1
                syms[s]["conf"] = max(syms[s]["conf"], float(best))
                syms[s]["how"] |= hows

        out[key] = {
            "n_frames": len(paths),
            "text": " ".join(chunks)[:20000],
            "symbols": {s: {"frames": v["frames"], "conf": round(v["conf"], 3),
                            "how": sorted(v["how"])} for s, v in syms.items()},
        }
        if (i + 1) % 50 == 0:
            with open(OCR_OUT, "w", encoding="utf-8") as f:
                json.dump(out, f)
            print(f"    {i+1}/{len(files)}", flush=True)

    with open(OCR_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)

    # how often does the screen alone show the right answer?
    hit = strong = 0
    for _, r in df.iterrows():
        d = out.get(f"{r.video_id}_{float(r.start):.2f}")
        if d and r.ticker_name in d["symbols"]:
            hit += 1
            if any(h in ("parens", "exchange") for h in d["symbols"][r.ticker_name]["how"]):
                strong += 1
    n = len(df)
    print(f"\n[B] correct ticker readable on screen: {hit}/{n} = {100*hit/n:.1f}%")
    print(f"    of those, a chart-header pattern: {strong}")
    print("    (the transcript contains it in 52.1% of segments)")


def main():
    """Parse the flags, then run fetch (A) and OCR (B)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N segments only")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--skip-fetch", action="store_true",
                    help="frames already on disk, just run OCR")
    args = ap.parse_args()

    df = load_segments()
    if args.limit:
        df = df.head(args.limit)

    if not args.skip_fetch:
        fetch_frames(df, args.workers, args.frames)
    if not os.path.isdir(FRAMES):
        print("no frames on disk, nothing to OCR")
        return
    read_frames(df, load_gazetteer())
    print("\nnext:  python src/3_run.py")


if __name__ == "__main__":
    main()
