# Larger model: Qwen2.5-3B

Follow-up to `experiments/`. Same task, same shortlists, same 5 video-grouped
folds: the ranker's top 5 go to a language model that picks one.

```
python large_model/train_large.py
python experiments/f_ensemble.py experiments/out/llm_results_alias_k5.json large_model/out/results_3B_k5.json
```

About 66 minutes on an RTX 4060 (8 GB).

## Result

| System | F1 |
|---|---|
| Ranker (with Wikidata aliases) | 80.64 |
| Qwen2.5-1.5B picks from top 5 (`experiments/`) | 85.88 |
| Qwen2.5-3B picks from top 5 (this folder) | 85.59 |
| **Average of the 1.5B and 3B models** | **86.03** |
| Best in the paper (Gemini 1.5 Flash, sees the video) | 86.23 |

Averaging the two models' scores for each candidate gives 86.03. They agree on
582 segments and disagree on 14, so they make slightly different mistakes.
Skipping the least confident 30% of answers gives 97.30% precision.

## Fitting a 3B model on 8 GB

- **4-bit weights (QLoRA):** 2.2 GB instead of 6.3 GB.
- **No `prepare_model_for_kbit_training`:** that helper converts the 152k-word
  vocabulary layer back to 32-bit. Skipping it took training from 8.6 GB and
  39 s/step down to 7.1 GB and 0.7 s/step.
- **Prompts capped at 540 tokens** (the 1.5B run used 700). Above about 600 the
  model spills into system RAM. This is the likely reason 3B alone did not beat
  1.5B: it sees a shorter transcript.

## What did not fit

Qwen2.5-7B in 4-bit: 5.7 GB of weights, 10.5 GB peak during training, 15 s per
step. About 20 hours per run on this GPU. It needs a GPU with 16 GB or more.
