"""EXPERIMENT, step D: other names for each company, from Wikidata.

    python experiments/d_wikidata_aliases.py

The registry knows each ticker by one name, usually the current legal one. But
speakers say "Google" not "Alphabet", "Raytheon" not "RTX", "Facebook" not
"Meta". And some symbols were reused: FB now points at a ProShares fund, FIT at
an unrelated company, so the registry's name for them is simply wrong for
2019-2023 videos.

Wikidata records, for listed companies, every ticker they have traded under
(including former ones) and their common alternative names. This pulls that for
companies listed on US exchanges and writes {ticker: [names...]}.

A general source on purpose: hand-adding aliases for the misses already looked
at would be fitting to the test set.
"""
import json
import os
import time
from collections import defaultdict

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "out", "wikidata_aliases.json")
URL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "ticker-extraction-research/0.1 (academic project)",
           "Accept": "application/sparql-results+json"}

# US exchanges: NYSE, Nasdaq, NYSE American, NYSE Arca, OTC Markets
EXCHANGES = ["wd:Q13677", "wd:Q82059", "wd:Q846626", "wd:Q4855598", "wd:Q1934683"]

QUERY = """
SELECT ?item ?ticker ?label ?alias WHERE {
  VALUES ?ex { %s }
  ?item p:P414 ?st .
  ?st ps:P414 ?ex ;
      pq:P249 ?ticker .
  OPTIONAL { ?item rdfs:label ?label FILTER(lang(?label) = "en") }
  OPTIONAL { ?item skos:altLabel ?alias FILTER(lang(?alias) = "en") }
}
"""


def main():
    names = defaultdict(set)
    for ex in EXCHANGES:
        r = requests.get(URL, params={"query": QUERY % ex}, headers=HEADERS, timeout=120)
        r.raise_for_status()
        rows = r.json()["results"]["bindings"]
        for b in rows:
            t = b["ticker"]["value"].strip().upper()
            for k in ("label", "alias"):
                if k in b:
                    names[t].add(b[k]["value"].strip())
        print(f"  {ex}: {len(rows)} rows", flush=True)
        time.sleep(2)
    out = {t: sorted(v) for t, v in names.items() if v}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0)
    print(f"\n{len(out)} tickers with names -> {OUT}")
    for t in ("GOOG", "RTX", "MARA", "FB", "FIT", "WULF", "XPEV", "META"):
        print(f"  {t:5s} {out.get(t, '-')}")


if __name__ == "__main__":
    main()
