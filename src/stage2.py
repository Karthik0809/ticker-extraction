"""STAGE 2: describe each (segment, candidate) pair as 37 numbers.

This is where every judgement about what counts as evidence is made. The
ranking model sees nothing except what this file emits, so if a signal is not
here the model cannot use it.

The correct ticker is in the stage-1 pool 92.6% of the time, so this is a ranking
problem over ~93 options, not a generation problem over 10,550.

My first attempt scored candidates with weights I picked by hand. It got the
answer first 58.6% of the time while having it in its top five 82.2% of the
time, so the evidence was fine and my ordering was wrong. There are 687
labelled segments; the weights are better learned from them than guessed.

Called roughly 100,000 times per run (687 segments x ~147 candidates).
"""
import os
import re
import sys
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import CUE

FEATURE_NAMES = [
    "sym_in_seg", "sym_count", "sym_first_pos",
    "spelled_in_seg", "cue_hit", "cashtag",
    "name_mass", "name_exact_toks", "name_frac", "name_fuzz", "name_first_pos",
    "src_seg", "src_desc", "src_title", "src_full", "src_dense", "src_ocr",
    "n_sources", "ticker_len", "pool_size", "cheap_score", "cheap_rank",
    "name_in_desc", "sym_in_title",
    # dense similarity between the segment and the company name, added as a
    # feature rather than used alone. Lexical evidence and semantic evidence
    # fail in different places: the symbol features are precise but silent on
    # the 47.9% of segments where the ticker is never spoken, and this one
    # still fires there.
    "dense_sim", "dense_rank",
    # Positional alignment. These videos are countdowns ("5 stocks to buy
    # now"), so the Nth segment of a video is usually about the Nth stock
    # mentioned in the transcript and in the description. A candidate whose
    # first mention sits at the same relative position as this segment is far
    # more likely to be the one being recommended here. Nothing in the content
    # of the segment carries this; it is structure.
    "seg_pos_in_video", "cand_pos_full", "cand_pos_desc",
    "align_full", "align_desc",
    # What is written on the screen during this segment. The paper's own
    # explanation for multimodal beating text-only on Task T is that stock
    # charts display the ticker, and that is exactly the evidence that survives
    # in the 47.9% of segments where nobody says the symbol out loud.
    # ocr_parens / ocr_exchange are chart-header patterns, "(VERI)" and
    # "NASDAQ: VERI", which are far stronger than a capitalised word that
    # happened to be on a bookshelf behind the presenter.
    "ocr_frames", "ocr_frac", "ocr_conf", "ocr_parens", "ocr_exchange",
    "ocr_name_in_text",
]

def _first_pos(m, n):
    """Where in the segment the evidence appears, normalised to [0,1].

    Finfluencer segments announce the stock early ("next up is..."), so a
    mention in the first fifth of the segment means something different from
    one in the last fifth, which is usually a comparison or an aside.
    """
    return (m.start() / n) if (m and n) else 1.0


def _mention_pos(text, tic, toks):
    """Relative position of the candidate's first mention, or -1 if absent."""
    text = str(text)
    n = len(text)
    if n < 20:
        return -1.0
    best = None
    m = re.search(r"(?<![A-Za-z0-9])" + re.escape(tic) + r"(?![A-Za-z0-9])", text)
    if m:
        best = m.start()
    low = text.lower()
    for t in toks:
        if len(t) < 4:
            continue
        mm = re.search(r"" + re.escape(t) + r"", low)
        if mm and (best is None or mm.start() < best):
            best = mm.start()
    return (best / n) if best is not None else -1.0


def pair_features(seg, tic, sources, weights, pool_size, cheap_score, cheap_rank,
                  desc="", title="", dense_sim=0.0, dense_rank=-1.0,
                  seg_pos=-1.0, full_text="", ocr=None):
    """Describe one (segment, candidate ticker) pair as a vector of 37 numbers.

    Called once per pair, so roughly 100k times per run. The ranker sees only
    what this function emits, which makes it the single place where "what
    counts as evidence" is decided.

    Args:
        seg:         this segment's transcript. The only text that speaks to
                     *which* candidate is meant; everything else is context.
        tic:         the candidate ticker being described.
        sources:     stage-1 tags saying how this candidate was found, e.g.
                     {"desc/cue", "ocr/screen"}. Becomes the provenance group.
        weights:         NameGroundingExtractor, supplies company-name tokens and
                     their two IDF weights.
        pool_size:   how many candidates this video produced. A video covering
                     three stocks is an easier problem than one covering thirty,
                     and the model learns to use that.
        cheap_score,
        cheap_rank:  the hand-written scorer's opinion. Kept as a feature rather
                     than discarded, because it encodes cheap signals the
                     learned model would otherwise have to rediscover.
        desc, title,
        full_text:   video-level context, used only for positional alignment.
        dense_sim,
        dense_rank:  embedding similarity between segment and company name.
        seg_pos:     where this segment sits in its video, 0 to 1. Pairs with
                     the candidate's mention position: these videos are
                     countdowns, so the nth segment is usually the nth stock.
        ocr:         this segment's OCR record, or None if no frames exist.

    Returns:
        A list of floats ordered exactly as FEATURE_NAMES. Order matters, since
        the trainer passes it positionally.
    """
    seg = str(seg)
    n = max(1, len(seg))
    low = seg.lower()
    f = {k: 0.0 for k in FEATURE_NAMES}

    # --- the symbol itself
    rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(tic) + r"(?![A-Za-z0-9])", re.I)
    hits = list(rx.finditer(seg))
    f["sym_in_seg"] = 1.0 if hits else 0.0
    f["sym_count"] = float(len(hits))
    f["sym_first_pos"] = _first_pos(hits[0] if hits else None, n)

    if len(tic) >= 3:
        sp = re.search(r"(?<![A-Za-z0-9])" + r"[\s\-\.]{1,2}".join(tic) + r"(?![A-Za-z0-9])",
                       seg, re.I)
        f["spelled_in_seg"] = 1.0 if sp else 0.0

    f["cashtag"] = 1.0 if re.search(r"\$" + re.escape(tic) + r"(?![A-Za-z0-9])", seg, re.I) else 0.0

    for cm in CUE.finditer(seg):
        tail = seg[cm.end():cm.end() + 40].upper()
        if re.sub(r"[^A-Z]", "", tail[: len(tic) * 3]).startswith(tic):
            f["cue_hit"] = 1.0
            break

    # --- the company name
    toks = weights.tok_of.get(tic, [])
    mass, exact, best_fz, first = 0.0, 0, 0, None
    for t in toks:
        if len(t) < 4:
            continue
        w = weights.full_weight(t)
        m = re.search(r"\b" + re.escape(t) + r"\b", low)
        if m:
            mass += w
            exact += 1
            if first is None or m.start() < first.start():
                first = m
        best_fz = max(best_fz, fuzz.partial_ratio(t, low))
    f["name_mass"] = mass
    f["name_exact_toks"] = float(exact)
    f["name_frac"] = exact / len(toks) if toks else 0.0
    f["name_fuzz"] = best_fz / 100.0
    f["name_first_pos"] = _first_pos(first, n)

    # --- where stage 1 found this candidate
    f["src_seg"] = 1.0 if any(s.startswith("seg/") for s in sources) else 0.0
    f["src_desc"] = 1.0 if any(s.startswith("desc/") for s in sources) else 0.0
    f["src_title"] = 1.0 if any(s.startswith("title/") for s in sources) else 0.0
    f["src_full"] = 1.0 if any(s.startswith("full/") for s in sources) else 0.0
    f["src_dense"] = 1.0 if any(s.startswith("dense/") for s in sources) else 0.0
    f["src_ocr"] = 1.0 if any(s.startswith("ocr/") for s in sources) else 0.0
    f["n_sources"] = float(len(sources))

    # --- context outside the segment
    if desc:
        dl = str(desc).lower()
        f["name_in_desc"] = 1.0 if any(t in dl for t in toks if len(t) >= 4) else 0.0
    if title:
        f["sym_in_title"] = 1.0 if re.search(
            r"(?<![A-Za-z0-9])" + re.escape(tic) + r"(?![A-Za-z0-9])", str(title), re.I) else 0.0

    f["ticker_len"] = float(len(tic))
    f["pool_size"] = float(pool_size)
    f["cheap_score"] = float(cheap_score)
    f["cheap_rank"] = float(cheap_rank)
    f["dense_sim"] = float(dense_sim)
    f["dense_rank"] = float(dense_rank)

    # --- what the screen showed
    if ocr:
        e = (ocr.get("symbols") or {}).get(tic)
        nf = max(1, int(ocr.get("n_frames") or 1))
        if e:
            f["ocr_frames"] = float(e.get("frames", 0))
            f["ocr_frac"] = float(e.get("frames", 0)) / nf
            f["ocr_conf"] = float(e.get("conf", 0.0))
            hows = e.get("how") or []
            f["ocr_parens"] = 1.0 if "parens" in hows else 0.0
            f["ocr_exchange"] = 1.0 if "exchange" in hows else 0.0
        otext = (ocr.get("text") or "").lower()
        if otext:
            f["ocr_name_in_text"] = 1.0 if any(
                t in otext for t in toks if len(t) >= 4) else 0.0

    # --- positional alignment
    f["seg_pos_in_video"] = float(seg_pos)
    f["cand_pos_full"] = _mention_pos(full_text, tic, toks)
    f["cand_pos_desc"] = _mention_pos(desc, tic, toks)
    f["align_full"] = (abs(seg_pos - f["cand_pos_full"])
                       if seg_pos >= 0 and f["cand_pos_full"] >= 0 else 1.0)
    f["align_desc"] = (abs(seg_pos - f["cand_pos_desc"])
                       if seg_pos >= 0 and f["cand_pos_desc"] >= 0 else 1.0)
    return [f[k] for k in FEATURE_NAMES]
