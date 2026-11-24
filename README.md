# Ticker Extraction on VideoConviction (Task T)

Given one segment of a finfluencer video, name the stock being recommended.

Dataset: [gtfintechlab/VideoConviction](https://huggingface.co/datasets/gtfintechlab/VideoConviction) ·
Paper: [arXiv:2507.08104](https://arxiv.org/pdf/2507.08104), KDD 2025

Sri Veerabhadra Karthik Mulugu · October 2026

The full write-up is in [`paper/ticker_extraction.pdf`](paper/ticker_extraction.pdf).
This file is the practical guide: what is here, what each file does, and how the
result was reached.

> **Update:** a follow-up in [`experiments/`](experiments/README.md) adds a fine-tuned language model that picks from the ranker's top 5. It raises F1 from 80.06 to **85.88**. A further follow-up in [`large_model/`](large_model/README.md) trains Qwen2.5-3B and averages it with the 1.5B model for **86.03**. Everything below describes the main system.

---

## Result

**F1 80.06** on Task T, out of fold, over all 687 annotated segments.

That beats 16 of the 21 published results in the benchmark's Table 6,
**including every text-only model**, using no large language model, no API key
and nothing paid for. The five systems still ahead are all multimodal.

| System | Input | Task T F1 |
|---|---|---|
| Mistral-7B | text | 63.73 |
| DeepSeek-R1 | text | 71.18 |
| Claude 3.5 Sonnet | text | 75.90 |
| GPT-4o | text | 76.76 |
| DeepSeek-V3 | text | 77.56 |
| Llama-3.1-405B | text | 78.60 |
| Qwen2.5-72B (best text-only) | text | 79.20 |
| **This system** | **text + OCR on frames** | **80.06** |
| Gemini 2.0 Flash | text + video | 82.24 |
| GPT-4o | text + video | 83.47 |
| Gemini 1.5 Flash (best overall) | text + video | 86.23 |

The ranker's margin between its first and second choice works as a confidence
score at no extra cost. Declining the thinnest margins trades coverage for
precision, which matters because a wrong ticker is worse than no ticker in
anything downstream:

| Operating point | Coverage | Precision |
|---|---|---|
| answer everything | 100% | 80.06 |
| drop lowest 10% by margin | 90.0% | 86.25 |
| drop lowest 20% | 79.9% | 90.71 |
| drop lowest 30% | 70.0% | **93.76** |

**How it was evaluated.** 5-fold `GroupKFold` split by `video_id`. Segments from
one video share a candidate pool and most of their text, so a random split would
put near-duplicates on both sides and inflate the score. No video is ever in
both train and test. Scores are averaged over three seeds, because LambdaMART on
687 queries moves about a point between seeds, which is the same size as the
gaps being claimed against the published table.

---

## What I did, start to end

**1. I measured the data before building anything.**

The benchmark poses this as generation: read a transcript segment, emit a
symbol. Before writing an extractor I checked whether the symbol is there to be
read at all.

It is not, in **47.9% of segments**. No cashtag, no bare symbol, not even
spelled out letter by letter.

**2. I found the absence is not random.**

It concentrates on the largest companies:

| Market cap | Segments | Symbol actually spoken |
|---|---|---|
| under $300M | 38 | 60.5% |
| $300M to $2B | 40 | 55.0% |
| $2B to $10B | 76 | 64.5% |
| $10B to $200B | 211 | 60.7% |
| **over $200B** | **167** | **28.1%** |

The first four bands sit between 55% and 65%. Only the top band collapses, which
is what makes this a cliff rather than a gradient.

I did not want to report this as a trend read off a table, so I tested it three
ways. All three are computed in `src/4_analysis.py` and written to
`results/analysis.json`.

| Test | Result | What it answers |
|---|---|---|
| Spearman rank correlation | rho = -0.215, p = 1.5e-4, over 307 tickers | Does visibility fall as company size rises? Rank-based, so one huge outlier cannot drive it. Measured per ticker, not per segment, so a heavily covered stock counts once |
| Bootstrap confidence interval | gap = 19.4 points, 95% interval [7.0, 31.5] | How big is the effect, and how sure am I? 5,000 resamples of the smallest and largest third. The interval excludes zero, so the direction is not an artefact of which tickers happened to be sampled |
| Mann-Whitney U, one-sided | p = 5.1e-4 | Same comparison without assuming the distribution is normal, which it is not, since visibility is a bounded proportion |

Three tests rather than one, because a single test on a pattern this convenient
would not be convincing, and because each covers a different weakness: the
correlation could be driven by outliers, the point estimate alone says nothing
about uncertainty, and a t-test would assume a shape this data does not have.

TSLA is annotated in 26 segments and the string "TSLA" appears in none of them.
Meanwhile a $225M nano-cap gets read out letter by letter: *"Veritone Inc.,
ticker V-E-R-I."* You say the symbol when you cannot assume the audience knows
the company, and nobody spells out Tesla.

So symbol matching does not merely miss cases. It misses the economically
largest ones, which is a bias, not noise.

**3. That changed the task from generating a symbol to choosing one.**

A generative model has to have memorised that Zynex trades as `ZYXI`, and
nothing stops it emitting `APPL`. Selecting from a registry of real symbols
removes that failure by construction. So I built the registry (see `data/`
below) and made the system pick from 10,550 real symbols.

**4. First pass: text only. F1 78.02.**

Two stages, described below.

**5. Second pass: reading the screen. F1 80.06.**

The symbol is missing from the transcript half the time, so I looked for where
else it exists, and it is written on screen.

**6. Decomposed the result to see where the remaining error lives.**

`F1 = stage-1 recall x ranker accuracy = 92.6% x 86.5% = 80.1`

Both halves are independently improvable, and the factorisation says which to
attack. An aggregate score cannot distinguish failing to *consider* the answer
from failing to *select* it.

---

## Architecture

### Stage 1, candidate generation, per video

Pool the segment transcript, the full transcript, the title and the description.
Harvest every plausible ticker: cashtags, cue phrases (`"ticker ZYXI"`), letters
spelled aloud, bare capitals, company names through an IDF-weighted index, and
the nearest companies by embedding similarity.

**Recall is the only objective here.** Being wrong is fine, stage 2 sorts that
out. Missing the answer is fatal, because nothing downstream can recover it.

Result: 10,550 symbols narrowed to a median of 93 per segment, keeping the
correct answer **92.6%** of the time.

How it got there, including two bugs worth knowing about:

| Change | Stage-1 recall |
|---|---|
| segment transcript only | 52.1% (symbol present at all) |
| + description, title, full transcript | 81.8% |
| - corpus-IDF penalty at stage 1 | 88.2% |
| + rejoin ASR-split compound names | 88.9% |
| + dense retrieval over company names | 91.6% |
| + symbols read off video frames | **92.6%** |

**Bug 1, the corpus-IDF hole.** Name tokens are weighted by gazetteer IDF times
corpus IDF. Corpus IDF is what stops ordinary words like "today" matching
companies that contain them. But corpus IDF is a frequency penalty, and the most
frequently discussed companies are the mega caps. It silently dropped TSLA (17
segments), AAPL (7), NVDA (6) and AMZN (6) out of the candidate pool entirely,
which is exactly the population identified in step 2. A precision fix had opened
a recall hole, and the aggregate score never showed it. Removed from stage 1
only, where precision is not the objective. Worth 6.4 points.

**Bug 2, the compound-name split.** ASR breaks CamelCase names into ordinary
words: CrowdStrike becomes "Crowd Strike", SoFi becomes "So Fi", DraftKings
becomes "Draft Kings". Both halves match nothing and the company vanishes.
Rejoining adjacent capitalised words recovers them. Worth 0.7 points.

### Stage 2, learned ranking, per segment

The correct ticker is in the pool 92.6% of the time, so this is a ranking
problem, not a generation problem. Each segment is a query, its candidate pool
is the list, and **LambdaMART** (LightGBM `objective="lambdarank"`) learns the
ordering over **37 features**.

A ranker rather than a classifier, because only the order within one segment
matters, whether VERI ends up above V. The absolute scores mean nothing alone.

The first version scored candidates with weights I chose by hand (3.0 for the
symbol, 4.0 for a cue phrase). Those were guesses and it showed: recall@1 of
58.6% against recall@5 of 82.2%. The answer was in the shortlist and the
ordering was wrong. There are 687 labelled segments, so the weights should be
learned from them rather than invented. Replacing them was the single largest
gain in the whole project, 16.5 points.

What the model actually leans on:

| Feature | Importance | What it is |
|---|---|---|
| `dense_sim` | 6314 | embedding similarity, segment to company name |
| `dense_rank` | 5665 | that similarity's rank within the pool |
| `pool_size` | 5640 | how many candidates this segment had |
| `cheap_rank` | 4588 | position from the hand-weighted pre-ranker |
| `name_first_pos` | 3010 | where the company name first appears |
| `name_mass` | 2496 | IDF-weighted name evidence |
| `cand_pos_desc` | 2218 | position in the video description |
| `align_full` | 2148 | countdown-position alignment |

Two feature groups worth naming:

- **Position.** Where a mention falls in the segment matters. These segments
  announce the stock early ("next up is..."), so a late mention is usually a
  comparison or an aside.
- **Alignment.** These videos are countdowns, so the Nth segment is usually
  about the Nth stock listed in the description. Nothing in the segment's
  content carries this. It is structure, and the model uses it.

The dense features rank highest, but the lexical ones are not redundant: they
are decisive on the 52.1% of segments where the symbol is present, and silent
elsewhere.

### Reading the screen

The benchmark paper's stated reason for multimodal beating text-only on Task T
is "stock charts in videos displaying ticker names". That does not need a vision
language model. It needs OCR, which runs locally on a GPU for nothing.

Finfluencers screen-record a finance page while they talk, and the header reads
literally `Veritone, Inc. (VERI) NasdaqGS 8.15 +0.44`.

The dataset ships YouTube URLs, not frames. Nothing is downloaded whole: yt-dlp
resolves a direct stream URL per video, and ffmpeg range-requests only the bytes
near each timestamp for a single JPEG. Six frames per segment, about an hour
total instead of days.

| | |
|---|---|
| videos retrieved | 158 of 215 |
| **videos gone** (deleted or private since 2019) | **57, 27%** |
| frames extracted | 2,628 |
| segments with any frame | 438 of 687, 64% |
| **correct symbol readable on screen** | **288 of 687, 41.9%** |
| of those, matching a chart-header pattern | 97 |

41.9% is across all 687 segments including the 36% with no frames at all. Among
segments that do have frames it is 65.8%. The transcript contains the symbol in
52.1%, so the screen is comparably informative and, critically, informative in
**different** segments: it still shows `VERI` where the presenter only ever says
"Veritone".

Adding it moved the system from **78.02 to 80.06**, the step that crosses the
best published text-only result.

The frames also contain plenty of text with nothing to do with any stock:
bookshelves, posters, a "MAN CAVE BEER" sign behind one presenter. So the OCR
stage decides nothing on its own. It emits evidence with a confidence and a
pattern type, and the ranker weighs it against everything else.

---

## Models and libraries, and where each is used

Nothing here is a large language model. Four learned or pretrained components,
each doing one job.

| Component | Exact model | Where | What it does and why |
|---|---|---|---|
| **Sentence embeddings** | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim | `1_prepare.py` builds them; `3_run.py` retrieves with them (`DenseScorer`, adding the 60 nearest companies to each pool); `stage2.py` declares them as features | Embeds all 10,550 company names and all 687 segments. This is the only route to a company whose name is never written the same way twice, and whose symbol is never spoken. Chosen because it is small, free and runs locally in minutes; a larger encoder was not the bottleneck |
| **Ranker** | LightGBM `LGBMRanker`, `objective="lambdarank"`, which is **LambdaMART** | `3_run.py` | The core of the system. 900 trees, learning rate 0.03, 63 leaves, trained 5 times on grouped folds and averaged over 3 seeds. Ranks the candidate pool per segment. Gradient-boosted trees rather than a neural model because there are only 687 labelled segments and 37 dense features, which is squarely where boosted trees win |
| **OCR** | EasyOCR, English, on GPU | `2_video.py` | Reads text off the sampled video frames. Not a vision language model: the ticker is literally rendered as text in a chart header, so character recognition is sufficient and costs nothing |
| **Fuzzy string matching** | `rapidfuzz` | `stage2.py` | One feature, `name_fuzz`, for near-miss name matches that survive ASR errors |

Supporting tools, none of them learned:

| Tool | Where | Why |
|---|---|---|
| `yt-dlp` + `ffmpeg` | `2_video.py` | Resolves a direct stream URL and range-requests only the bytes near each timestamp, so no video is ever downloaded whole |
| `yfinance` | `1_prepare.py`, `4_analysis.py` | Resolves 34 company names the SEC file no longer lists, and fetches the market caps used in the size analysis |
| `scikit-learn` | `3_run.py` | `GroupKFold` only, to split by video |
| `scipy` | `4_analysis.py` | `spearmanr` and `mannwhitneyu` for the two tests above |
| SEC `company_tickers.json` | `1_prepare.py` | The 10,434-symbol base registry |

**Why no neural ranker.** With 687 queries, a learned-to-rank neural model would
have more parameters than training signal. LambdaMART optimises the ordering
directly, trains in seconds, and exposes feature importances, which is how the
two bugs above were found at all. A black-box ranker of the same accuracy would
have hidden them.

---

## Why no LLM

Three reasons, in order of weight.

**It would not have fixed the actual problem.** The failure measured in step 2
is that the symbol is absent from the text. An LLM reading the same transcript
has the same missing information. The benchmark's own results show this: 16
large models were evaluated and the best text-only one reached 79.20, so the
ceiling on reading the transcript alone is roughly where they all cluster. The
gain here came from adding a *source* (the screen) and from constraining the
output space (a registry), not from a stronger reader.

**The free tiers could not run it.** I intended to test an LLM reranker over the
stage-1 candidates. The Gemini free tier cut out after 22 of 687 segments, and
model names shifted underneath the work during the attempt. Putting the result
behind a paid key was not something I wanted, so I dropped it.

**Reproducibility.** Everything here runs free, offline after setup, and
deterministically given the same seeds. A published number that depends on a
hosted model is not re-checkable once that model is retired, and several of the
models in the benchmark's own table have already been superseded.

The honest cost is that there is no LLM comparison on this dataset. I would like
to run it properly with real quota.

---

## What is in each folder

### `src/` seven files: three you run, three they import, one that measures

| File | What it does | Why it exists |
|---|---|---|
| `1_prepare.py` | Downloads the dataset, builds the ticker registry, computes embeddings | One-time setup. Three jobs, each skipped if its output is already on disk |
| `2_video.py` | Pulls frames from YouTube and OCRs them | The optional visual pass. Worth +2.0 F1. Separate because 27% of videos are gone, so what it collects depends on what is still online |
| `3_run.py` | **The system.** Both stages, cross-validation, final score | Run this to reproduce the headline number |
| `4_analysis.py` | Measures the dataset: symbol presence and the market-cap effect | Produces the findings in steps 1 and 2 above. Builds and trains nothing |
| `core.py` | Data loading, the 10,550-ticker registry, word weights | Shared foundations. Nothing here runs on its own |
| `stage1.py` | Narrows 10,550 tickers to about 93 for this video | Seven detection rules plus the cheap pre-ranker |
| `stage2.py` | Turns each (segment, candidate) pair into 37 numbers | The feature definitions the ranker consumes |

### `data/` the registry and the cached evidence

| File | Size | What it is |
|---|---|---|
| `sec_tickers.json` | 355 KB | 10,434 symbols from the SEC's `company_tickers.json`. Every company registered **today**, authoritative and free |
| `ticker_names.json` | 14 KB | 417 resolved `{ticker: company name}` for this dataset's correct answers, merged from all three sources |
| `ticker_names_supplement.json` | 3 KB | **82 companies written in by hand.** The SEC lists current registrants, but the videos are 2019 to 2023, so acquired and delisted firms are absent: Twitter, Slack, Xilinx, FireEye, Activision Blizzard |
| `ocr_evidence.json` | 801 KB | What OCR read from 438 segments' frames: the raw text, and which symbols were found by which pattern at what confidence. **Keep this.** 27% of the videos are now gone, so it cannot be fully regenerated |
| `market_caps.json` | 7 KB | 307 market caps from Yahoo, cached. Cached deliberately: caps move daily, and a figure quoted in the write-up has to stay the same when the script is re-run |
| `dead_videos.json` | 1 KB | The 57 video IDs that no longer resolve, so later runs do not retry them |
| `emb_company_index.json` | 145 KB | Which row of `emb_companies.npy` is which ticker |
| `videoconviction.parquet` | 3.5 MB | The dataset itself. Not in the code zip, `1_prepare.py` downloads it |
| `emb_companies.npy`, `emb_segments.npy` | 17 MB | Sentence embeddings. Not in the code zip, rebuilt in minutes |

Registry coverage is where most of the data work went. Of the 441 distinct
correct answers, the SEC file covered **301**. yfinance resolved **34** of the
remaining 140. I wrote in the last **82** by hand. **24** could not be resolved
from any source and are left blank rather than guessed, since a wrong company
name actively steers the matcher toward the wrong company.

### `results/` what the scripts produce

| File | Produced by | Contents |
|---|---|---|
| `task_t_scores.json` | `3_run.py` | The headline numbers: F1, stage-1 recall, ranker accuracy, which published systems are beaten, and all 37 feature importances |
| `task_t_predictions.csv` | `3_run.py` | One row per segment: correct answer, prediction, confidence margin, right or wrong, video ID. This is the file to open to see *which* segments failed |
| `analysis.json` | `4_analysis.py` | The dataset measurements: 47.9% absence, the Spearman correlation, the bootstrap interval, the Mann-Whitney result |

### `paper/`

`ticker_extraction.pdf`, the write-up, 7 pages in IEEE conference format,
and `ticker_extraction.tex`, its LaTeX source.

---

## Running it

```bash
pip install datasets pandas numpy scipy scikit-learn lightgbm sentence-transformers torch easyocr rapidfuzz yfinance requests yt-dlp imageio-ffmpeg
```

```bash
python src/1_prepare.py    # ~2 min, one time
python src/2_video.py      # ~55 min, optional, worth +2 F1
python src/3_run.py        # ~8 min, prints the result
python src/4_analysis.py   # ~1 min, prints the dataset measurements
```

Without step 2 the system scores 78.02; with it, 80.06.

**One thing that will bite you if you edit the imports.** `1_prepare.py` and
`2_video.py` both import `torch` before `pandas`. That is not style. Both
libraries ship their own OpenMP runtime, and on Windows the wrong order kills
the process with a segmentation fault and no error message at all.

---

## Limitations

- **Comparability.** The published systems are zero-shot; this one is trained on
  VideoConviction under grouped cross-validation. That is a valid benchmark
  number on this dataset and is not a claim about transfer to another corpus.
  Stated plainly because the results table would otherwise invite that reading.
- **24% of the correct answers no longer trade** (TWTR, SQ, WORK, FEYE, XLNX,
  ATVI). Any system validated against a current registry loses them silently.
- **Label noise.** The answer key contains `"NYSE: AMZN"`, `"NYSE: WMT"` and
  `"ADR"`, which are not symbols at all.
- **27% of the source videos no longer exist.** 57 of 215, so 36% of segments
  get no visual evidence and fall back to text features. The benchmark's authors
  ran their experiments when more of the corpus was live, so their multimodal
  numbers reflect fuller coverage than is obtainable now. Splitting the system's
  own predictions by whether a frame was retrieved gives F1 84.47 on the 438
  segments that still have video against 72.29 on the 249 that do not, so most
  of the distance to 86.23 is attrition rather than method. Not a clean
  experiment: surviving segments also carry transcripts 46% longer on average,
  which independently helps the text features, so treat it as an upper bound.
- **OCR is not the full visual signal.** The multimodal models see motion,
  layout and charts as images. This reads text off still frames.
- **One corpus.** The method cannot be separated from what this dataset rewards
  without testing on another.
