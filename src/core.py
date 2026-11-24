"""Shared building blocks: paths, data loading, the ticker list, word weights.

Nothing here runs on its own. The three numbered scripts import from this file.

Three things live here:

  1. Where files are and how to load the dataset.
  2. The GAZETTEER, which is the list of every ticker the system may answer
     with, and the phrases that announce a ticker in speech.
  3. WORD WEIGHTS, which decide how much a word tells you about which company
     is meant. "Veritone" tells you a lot. "Holdings" tells you nothing.
"""
import io
import json
import math
import os
import re
import sys
from collections import defaultdict
import pandas as pd

# Windows consoles default to cp1252 and crash on the characters in these
# transcripts, so force UTF-8 output.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

# ===========================================================================
# 1. LOADING THE DATA
# ===========================================================================

def load_segments():
    """The 687 segments that actually contain a stock recommendation.

    The file has 760 rows. The other 73 are marked is_rec_present = "No":
    a human watched them and decided the presenter mentions a stock but never
    recommends anything. Those have no correct answer, so they are dropped.
    """
    df = pd.read_parquet(os.path.join(DATA, "videoconviction.parquet"))
    df = df[df.is_rec_present == "Yes"]
    df["segment_transcript"] = df.segment_transcript.fillna("")
    df = df[df.segment_transcript.str.strip() != ""]
    df["ticker_name"] = df.ticker_name.astype(str).str.strip().str.upper()
    return df.reset_index(drop=True)

# ===========================================================================
# 2. THE GAZETTEER  (the list of answers the system is allowed to give)
# ===========================================================================

# Phrases that announce a symbol out loud: "ticker ZYXI", "trades under NVDA".
# When one of these sits right before a symbol it is the strongest single
# signal in the transcript, 96.6% precise on its own.
CUE = re.compile(
    r"(?:ticker(?:\s+symbol)?|symbol|trades?\s+(?:as|under)|listed\s+(?:as|under)"
    r"|under\s+the\s+(?:ticker|symbol)|the\s+ticker)\s*(?:is\s+)?[:,\-]?\s*",
    re.I,
)

# Tickers that are also ordinary English words. Without this filter, matching
# capitalised words against the gazetteer produces almost only false positives,
# because "ALL" and "NOW" are words long before they are companies.
WORD_TICKERS = {
    "A", "I", "IT", "IS", "ALL", "AN", "AM", "ARE", "AS", "AT", "BE", "BY", "DO",
    "FOR", "GO", "HAS", "HE", "IF", "IN", "NO", "ON", "OR", "SO", "TO", "UP", "US",
    "WE", "ALLY", "BIG", "CAN", "CAR", "CASH", "CEO", "COST", "DAY", "EAT", "EDIT",
    "END", "EVER", "FAST", "FIT", "FIVE", "FLOW", "FREE", "FUN", "GAIN", "GOOD",
    "HOME", "HOPE", "HUGE", "IDEA", "JOB", "KEY", "LIFE", "LIVE", "LOVE", "LOW",
    "MAN", "MAX", "MOVE", "NEW", "NEXT", "NICE", "NOW", "ONE", "OPEN", "OUT",
    "OWN", "PAY", "PLAN", "PLAY", "PLUS", "REAL", "RUN", "SAFE", "SEE", "SELF",
    "SHOP", "SMART", "STAY", "STEP", "SUN", "TEAM", "TELL", "TRUE", "TRY", "TWO",
    "VERY", "WANT", "WAY", "WELL", "WIN", "WORK", "YES", "BEST", "CARE", "DATA",
    "EDGE", "FUND", "GAS", "GOLD", "HIGH", "HOT", "LAND", "LINE", "MAIN", "MIND",
    "OIL", "PACE", "PEAK", "POST", "RISE", "ROAD", "ROOT", "SAVE", "SHOT", "SIDE",
    "SIGN", "SITE", "SKY", "SOLO", "SPOT", "TOP", "TRIP", "TURN", "VIEW", "WAVE",
    "WISH", "WORD", "ZERO",
}


def load_gazetteer():
    """{TICKER: company name} for all 10,550 symbols the system can answer with.

    Merged from three files, because no single source has them all:

      sec_tickers.json              every company registered with the SEC today
      ticker_names.json             names resolved for this dataset's tickers
      ticker_names_supplement.json  82 companies written in by hand

    The supplement exists because the SEC lists companies registered *now*,
    while these videos are from 2019-2023. Twitter, Slack, FireEye and Xilinx
    were all acquired since, so they are absent from the SEC file, and about a
    quarter of the correct answers are companies in that state.

    This merged dictionary is never saved to disk. It is cheap to rebuild and
    a cached copy would go stale.
    """
    g = {}
    for fn in ("ticker_names.json", "ticker_names_supplement.json", "sec_tickers.json"):
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if k.startswith("_"):      # the supplement has a "_comment" key
                    continue
                g.setdefault(k.upper(), v)
    return g


# ===========================================================================
# 3. WORD WEIGHTS  (how much does a word narrow down which company?)
# ===========================================================================

# Legal suffixes and generic nouns that appear in hundreds of company names and
# therefore identify nothing. Stripped before a name is indexed.
STOP_NAME_TOKENS = {
    "inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.",
    "ltd", "ltd.", "limited", "plc", "holdings", "holding", "group", "the",
    "technologies", "technology", "systems", "international", "class",
    "common", "stock", "shares", "sa", "nv", "ag", "trust", "etf", "fund",
    "index", "american", "depositary", "receipt", "adr",
}


def name_tokens(company_name):
    """The useful words of a company name. "Veritone, Inc." -> ["veritone"]."""
    toks = re.findall(r"[A-Za-z][A-Za-z'&]*", str(company_name).lower())
    return [t for t in toks if t not in STOP_NAME_TOKENS and len(t) > 2]


class WordWeights:
    """How much each word tells you about which company is being discussed.

    Two separate measures of rarity, multiplied, because either alone is wrong
    in a specific way.

    GAZETTEER RARITY: how few company names contain this word.
        "Veritone" appears in one name, so it identifies a company outright.
        "Holdings" appears in hundreds, so it scores zero.
        What it cannot see: "Today" is also rare in company names, so this
        measure rates it highly. An early version matched "today" to a company
        because of exactly that.

    CORPUS RARITY: how few of the 687 transcripts contain this word.
        This is what kills "today": rare in company names, but present in
        nearly every video. Learned from the data rather than from a stoplist
        I would otherwise have to write by hand, which would encode my
        assumptions instead of measuring them.

    IMPORTANT: the product is used for FEATURES, where precision matters.
    Candidate generation deliberately uses gazetteer rarity alone. Corpus
    rarity is a frequency penalty, and the most-discussed companies are the
    mega caps, so applying it at the recall stage silently dropped TSLA, AAPL,
    NVDA and AMZN out of the candidate pool entirely. See gaz_weight() below.
    """

    def __init__(self, gazetteer, corpus=None):
        self.gaz = gazetteer

        # --- gazetteer rarity: count how many company names use each word
        self.tok_of = {}
        name_freq = defaultdict(int)
        for tic, nm in gazetteer.items():
            toks = [t for t in sorted(set(name_tokens(nm))) if len(t) >= 3]
            if not toks:
                continue
            self.tok_of[tic] = toks
            for t in toks:
                name_freq[t] += 1

        n = max(1, len(self.tok_of))
        self.idf = {t: math.log(n / c) for t, c in name_freq.items()}
        self.max_idf = max(self.idf.values()) if self.idf else 1.0

        # reverse lookup: word -> which tickers use it
        self.by_token = defaultdict(set)
        for tic, toks in self.tok_of.items():
            for t in toks:
                self.by_token[t].add(tic)

        # --- corpus rarity
        self.corpus_idf = {}
        self.max_corpus_idf = 1.0
        if corpus is not None:
            self.fit_corpus(corpus)

    def fit_corpus(self, docs):
        """Count how many documents contain each word."""
        freq = defaultdict(int)
        n = 0
        for d in docs:
            n += 1
            for w in set(re.findall(r"[a-z][a-z']{2,}", str(d).lower())):
                freq[w] += 1
        n = max(1, n)
        self.corpus_idf = {w: math.log(n / c) for w, c in freq.items()}
        self.max_corpus_idf = math.log(n)

    def corpus_weight(self, tok):
        """Corpus rarity as 0 to 1. A word never seen is assumed maximally rare."""
        if not self.corpus_idf:
            return 1.0
        return min(1.0, self.corpus_idf.get(tok, self.max_corpus_idf) / self.max_corpus_idf)

    def full_weight(self, tok):
        """Both rarities combined. Used for features, where precision matters."""
        return self.idf.get(tok, 0.0) * self.corpus_weight(tok)

    def gaz_weight(self, tok):
        """Gazetteer rarity only. Used for finding candidates, where recall matters.

        Deliberately ignores corpus rarity. See the class docstring: including
        it here deletes the mega caps from the candidate pool.
        """
        return self.idf.get(tok, 0.0)
