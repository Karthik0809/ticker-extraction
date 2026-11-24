"""STAGE 1: narrow 10,550 possible tickers down to about 93 for this video.

WHY THIS STAGE EXISTS
---------------------
The obvious approach is to search the segment for something ticker-shaped. That
caps out at F1 54.4, because 47.9% of segments never say the ticker at all.

The obvious fix is to also read the video's description, where the ticker
appears 35.1% of the time. I tried that. Recall moved 0.1 points and precision
fell 20 points, from 71.2% to 51.0%.

Looking at the failures explained it. A finfluencer lists EVERY stock of the
video in the description, usually as a timestamped index. So the description
tells you which stocks the VIDEO covers and says nothing about which one THIS
segment is about. It is not evidence. It is a candidate list.

That is the whole idea: anything that narrows the field without choosing
between the options belongs in a retrieval stage, not in a scoring function.

WHAT MATTERS HERE
-----------------
Recall, only. Being wrong is fine; stage 2 sorts that out. Missing the answer
is fatal, because nothing downstream can recover it. Current recall: 92.6%.
"""
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import CUE, WORD_TICKERS

class CandidateGenerator:
    """Collect every ticker a video could plausibly be about."""

    def __init__(self, gazetteer, weights):
        self.gaz = gazetteer
        self.w = weights

    def from_text(self, text, allow_bare=True):
        """Find ticker-shaped things in one block of text.

        Returns {TICKER: {how it was found}}. The "how" is kept because the
        ranker later treats a chart-header match differently from a capitalised
        word that happened to match.
        """
        found = defaultdict(set)
        t = str(text)

        # --- $VERI. Never occurs in speech, since the convention is written,
        # but it does appear in typed descriptions and titles.
        for m in re.finditer(r"\$([A-Za-z]{1,5})(?![A-Za-z])", t):
            s = m.group(1).upper()
            if s in self.gaz:
                found[s].add("cashtag")

        # --- "ticker VERI" / "trades under NVDA". Highest-precision rule at
        # 96.6% when it fires alongside a spelled-out symbol.
        for cm in CUE.finditer(t):
            tail = t[cm.end():cm.end() + 40]
            sm = re.match(r"([A-Za-z])(?:[\s\-\.]{1,2}([A-Za-z])){1,4}(?![A-Za-z])", tail)
            if sm:
                s = "".join(re.findall(r"[A-Za-z]", sm.group(0))).upper()
                if s in self.gaz:
                    found[s].add("cue")
            tm = re.match(r"([A-Za-z]{1,5})(?![A-Za-z])", tail)
            if tm and tm.group(1).upper() in self.gaz:
                found[tm.group(1).upper()].add("cue")

        # --- "V-E-R-I", the presenter dictating an unfamiliar symbol.
        for m in re.finditer(
                r"(?<![A-Za-z0-9])([A-Za-z](?:[\s\-\.]{1,2}[A-Za-z]){2,4})(?![A-Za-z0-9])", t):
            s = "".join(re.findall(r"[A-Za-z]", m.group(1))).upper()
            if s in self.gaz:
                found[s].add("spelled")

        # --- a bare capitalised word that is a valid symbol. Noisy, which is
        # why WORD_TICKERS filters out ALL, NOW, WELL and friends.
        if allow_bare:
            for tok in re.findall(r"(?<![A-Za-z0-9])([A-Z]{2,5})(?![A-Za-z0-9])", t):
                if tok in self.gaz and tok not in WORD_TICKERS:
                    found[tok].add("bare")

        # --- the company NAME, which is the only route for the 47.9% of
        # segments where the symbol is never spoken.
        capped = {m.group(1).lower() for m in
                  re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z']{2,})\b", t, re.M)}

        # ASR splits CamelCase names into ordinary words: CrowdStrike becomes
        # "Crowd Strike", SoFi becomes "So Fi", DraftKings becomes "Draft
        # Kings". Both halves match nothing and the company disappears. Rejoin
        # adjacent capitalised words and test that too. Worth +0.7 recall.
        for m in re.finditer(r"\b([A-Z][A-Za-z']{1,})\s+([A-Z][A-Za-z']{1,})\b", t):
            joined = (m.group(1) + m.group(2)).lower()
            if 4 <= len(joined) <= 24:
                capped.add(joined)

        for tok in capped:
            if len(tok) < 4:
                continue
            tics = self.w.by_token.get(tok)
            if not tics or len(tics) > 25:   # a word shared by 25+ companies says little
                continue
            # gaz_weight, NOT full_weight. This is the bug that cost 6 points:
            # full_weight penalises frequently-discussed words, and the most
            # frequently discussed companies are the mega caps, so using it
            # here deleted TSLA (17 segments), AAPL (7), NVDA (6) and AMZN (6)
            # from the pool entirely. Precision is stage 2's problem.
            if self.w.gaz_weight(tok) < 0.30 * self.w.max_idf:
                continue
            for tic in tics:
                found[tic].add(f"name:{tok}")
        return found

    def for_video(self, row):
        """Pool candidates from every text field of the video.

        Tagged by source, because where a candidate came from is itself a
        signal: one found in this segment outranks one found only in the
        description of a video covering ten stocks.
        """
        pool = defaultdict(set)
        for field, tag in [("videoDescription", "desc"), ("video_title", "title"),
                           ("transcript", "full"), ("segment_transcript", "seg")]:
            v = row.get(field)
            if not isinstance(v, str) or not v.strip():
                continue
            for tic, hows in self.from_text(v).items():
                pool[tic] |= {f"{tag}/{h}" for h in hows}
        return pool

class CheapRanker:
    """A quick first ordering of the candidates, using hand-picked weights.

    This is NOT the final answer. On its own it gets the right ticker first
    only 58.6% of the time, while having it somewhere in its top five 82.2% of
    the time: the evidence is there and the ordering is wrong. That gap is why
    stage 2 learns its weights instead.

    Kept anyway for two reasons. It trims the pool before the expensive feature
    step, and its output position turns out to be one of the most useful
    features the learned model consumes.
    """

    def __init__(self, gazetteer, weights):
        self.gaz = gazetteer
        self.w = weights

    def _score(self, seg, tic, sources):
        """How much does THIS segment look like it is about THIS ticker?"""
        s = 0.0
        why = []

        # the symbol, written plainly
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(tic) + r"(?![A-Za-z0-9])", seg, re.I):
            s += 3.0
            why.append("sym-in-seg")

        # the symbol, dictated letter by letter
        if re.search(r"(?<![A-Za-z0-9])" + r"[\s\-\.]{1,2}".join(tic) + r"(?![A-Za-z0-9])",
                     seg, re.I):
            s += 3.5
            why.append("spelled-in-seg")

        # a cue phrase right before it: the strongest single signal
        for cm in CUE.finditer(seg):
            tail = seg[cm.end():cm.end() + 40].upper()
            if re.sub(r"[^A-Z]", "", tail[:len(tic) * 3]).startswith(tic):
                s += 4.0
                why.append("cue-in-seg")
                break

        # the company name, weighted by how much each word narrows things down
        low = seg.lower()
        name_mass = sum(self.w.full_weight(tok)
                        for tok in self.w.tok_of.get(tic, [])
                        if len(tok) >= 4 and tok in low)
        if name_mass:
            s += min(4.0, name_mass)
            why.append(f"name({name_mass:.1f})")

        # small bonus for where stage 1 found it
        if any(x.startswith("seg/") for x in sources):
            s += 0.8
        if any(x.startswith("title/") for x in sources):
            s += 0.4
        if any(x.startswith("desc/") for x in sources):
            s += 0.25
        return s, ",".join(why)

    def rank(self, seg, pool):
        """Best candidate first.

        Returns [(score, ticker, why), ...]. `why` names which rules fired,
        which is what makes failures inspectable by hand. For example on the
        Veritone segment this returns VERI at 12.30 and Visa at 11.30, because
        spelling out "V-E-R-I" contains the letter V and V is Visa's ticker.
        """
        scored = []
        for tic, sources in pool.items():
            score, why = self._score(seg, tic, sources)
            scored.append((score, tic, why))
        scored.sort(reverse=True)
        return scored
