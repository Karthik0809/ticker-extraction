# Experiments: LLM chooser and entity linking

Follow-up to the main system in `src/`. The ranker still builds each
segment's shortlist; a small fine-tuned language model now makes the final
pick from the ranker's top 5.

## Result

| System | F1 |
|---|---|
| Main system (`src/3_run.py`) | 80.06 |
| + Wikidata aliases in the ticker list | 80.64 |
| **+ fine-tuned LLM choosing from the top 5** | **85.88** |
| Best text-only model in the paper (Qwen2.5-72B) | 79.20 |
| Best model in the paper (Gemini 1.5 Flash, sees the video) | 86.23 |

- Above 18 of the 21 published systems, including all 16 text-only models.
- On the 438 segments whose video is still online: **89.04**. On the 249 whose
  video is deleted: 80.32. (Surviving segments also have longer transcripts,
  so not all of that gap is the video.)
- The LLM fixed 43 of the ranker's mistakes and broke 7 of its correct answers.
- Skipping its least confident 30% of answers: 96.88% precision.

All numbers are out of fold, over the same five video-grouped folds as the
main system.

## Steps, in order

| Script | What it does | Runtime |
|---|---|---|
| `a_rank_lists.py` | Nested cross-validation: every segment's ranker shortlist comes from a ranker that never saw its video, including the training segments. Reproduces 80.06 at top-1. `--aliases` adds step D's names | ~25 min |
| `b_llm_chooser.py` | Fine-tunes Qwen2.5-1.5B-Instruct with LoRA on each fold's training segments, then scores each test segment's top-K candidates and picks the most likely | ~45 min, GPU |
| `c_stage1_misses.py` | Sorts the segments where the right ticker misses the shortlist by reason | seconds |
| `d_wikidata_aliases.py` | Pulls alternative and former company names from Wikidata ("Google" for GOOG) | ~1 min |
| `e_report.py` | Scores an LLM run: accuracy, fixes vs breaks, abstention curve, and a ranker blend | seconds |
| `f_ensemble.py` | Averages several LLM runs | seconds |
| `g_gate.py` | Uses the ranker when it is confident, the LLM otherwise | seconds |

To reproduce the headline 85.88:

```
python experiments/d_wikidata_aliases.py
python experiments/a_rank_lists.py --aliases
python experiments/b_llm_chooser.py --k 5 --aliases
python experiments/e_report.py experiments/out/llm_results_alias_k5.json
```

Steps D and A are already done: their outputs are in `out/`. Step B needs a
CUDA GPU with 8 GB; it downloads the model (about 3 GB) on first run.

## How the LLM step works

- Prompt: video title, "segment 3 of 5", the segment transcript, and the
  ranker's top 5 candidates with company names.
- Training: LoRA trains 18.5M numbers (1.2% of the model) rather than all 1.5
  billion, which fits an 8 GB laptop GPU. 2 epochs, learning rate 2e-4.
- Prediction: the model never writes free text. It scores how likely each of
  the 5 candidates is as the full answer and takes the highest, so the answer
  is always a real ticker.
- Only predictions for the answer tokens are computed. Computing them over the
  whole 152k-word vocabulary at every prompt position needs about 11 GB.

## What did not help

| Variant | F1 | Flag |
|---|---|---|
| Tickers read on screen added to the prompt | 85.59 | `--screen` |
| Top 10 candidates instead of 5 | 85.59 | `--k 10` |
| Order candidates appear in the description | 85.88 | `--desc` |
| Average of several trained models | 85.74 | `f_ensemble.py` |
| LLM only when the ranker is unsure | 85.59 | `g_gate.py` |
| Qwen2.5-3B, bf16 and 4-bit | did not fit | `--model`, `--4bit` |

Every variant lands between 85.3 and 85.9, which suggests the 1.5B model is
at its limit. The ceiling inside the top 5 is 90.54, so a larger model on a
larger GPU is the obvious next step.

## Why the shortlist misses

Of the 51 segments where the right ticker never reaches the shortlist, 24
cannot be fixed by any linking: the ticker has no known company name, or the
answer key itself is wrong ("NYSE: AMZN"). Most of the rest are companies
named differently from the registry ("Google" for Alphabet) or symbols that
were reused (FB now points at an unrelated fund). Wikidata aliases recovered 3.

## Files in `out/`

| File | What it is |
|---|---|
| `wikidata_aliases.json` | {ticker: [other names]} from step D |
| `rank_lists.json`, `rank_lists_alias.json` | Each segment's ranker top 20, per fold, from step A |
| `llm_results_*.json` | Each LLM run's per-segment picks and candidate scores |
