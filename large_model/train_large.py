"""Train a larger language model as the final picker, on an 8 GB laptop GPU.

    python large_model/train_large.py                              (Qwen2.5-3B)
    python large_model/train_large.py --model Qwen/Qwen2.5-7B-Instruct

Same task and same evaluation as experiments/b_llm_chooser.py (85.88 with the
1.5B model): the ranker's top 5 candidates go to the model, which scores each
one and picks the most likely. Same 5 video-grouped folds, same shortlists.

What is different, and why
--------------------------
1. A bigger model, stored in 4 bits (QLoRA). Qwen2.5-3B takes 2.2 GB this way
   instead of 6.3 GB, so it fits next to the training memory.
2. No `prepare_model_for_kbit_training`. That standard helper converts the
   152,000-word vocabulary layer back to 32-bit precision, which costs over a
   gigabyte and pushed training to 8.6 GB and 39 seconds per step. Without it:
   7.1 GB and 0.7 seconds per step.
3. Prompts capped at 540 tokens (the 1.5B run used 700). Above about 600 the
   3B model spills out of GPU memory into system RAM and slows down 6x. The
   transcript is shortened to fit; the candidates are never cut.

Needs experiments/out/rank_lists_alias.json (from experiments/a_rank_lists.py
--aliases). Writes large_model/out/.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch  # must load before pandas on this machine
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from core import load_gazetteer, load_segments                               # noqa: E402
from b_llm_chooser import encode, make_prompt, score_candidates, seg_context  # noqa: E402

from peft import LoraConfig, get_peft_model                                   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

OUT = os.path.join(ROOT, "large_model", "out")
LISTS = os.path.join(ROOT, "experiments", "out", "rank_lists_alias.json")


def load_model(name):
    """4-bit weights, bf16 compute, and nothing upcast to 32-bit."""
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(name, quantization_config=q,
                                             dtype=torch.bfloat16, device_map={"": 0})
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    return get_peft_model(m, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))


def fit_prompt(tok, row, pos, cands, gaz, max_tokens):
    """Shorten the transcript until the prompt fits under max_tokens."""
    chars = 1800
    while True:
        p_ids = encode(tok, make_prompt(row, pos, cands, gaz, chars))
        if len(p_ids) <= max_tokens or chars <= 300:
            return p_ids
        chars -= 150


def train(model, examples, args, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = math.ceil(len(examples) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / total))
    model.train()
    for ep in range(args.epochs):
        random.shuffle(examples)
        run = 0.0
        for i, (p_ids, a_ids) in enumerate(examples):
            ids = torch.tensor([p_ids + a_ids]).cuda()
            # predictions only for the answer tokens, see experiments/b_llm_chooser.py
            logits = model(input_ids=ids, logits_to_keep=len(a_ids) + 1).logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits[0, :-1], torch.tensor(a_ids).cuda()) / args.accum
            loss.backward()
            run += loss.item() * args.accum
            if (i + 1) % args.accum == 0 or i == len(examples) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        print(f"    epoch {ep + 1}: mean loss {run / len(examples):.4f}", flush=True)
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=540)
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    size = args.model.split("-")[1]
    os.makedirs(OUT, exist_ok=True)

    df, gaz = load_segments(), load_gazetteer()
    pos = seg_context(df)
    with open(LISTS, encoding="utf-8") as f:
        lists = json.load(f)
    tok = AutoTokenizer.from_pretrained(args.model)

    results, t0 = [], time.time()
    for f in [int(x) for x in args.folds.split(",")]:
        fold = lists["folds"][f]
        ex = []
        for r in fold["train"]:
            cands = r["cands"][:args.k]
            if r["gold"] in [c["tic"] for c in cands]:
                p = fit_prompt(tok, df.loc[r["sid"]], pos[r["sid"]], cands, gaz, args.max_tokens)
                a = tok(r["gold"] + "<|im_end|>", add_special_tokens=False).input_ids
                ex.append((p, a))
        print(f"\nFOLD {f + 1}/5: training {size} on {len(ex)} segments", flush=True)
        model = load_model(args.model)
        train(model, ex, args, seed=f)

        for r in fold["test"]:
            cands = r["cands"][:args.k]
            p = fit_prompt(tok, df.loc[r["sid"]], pos[r["sid"]], cands, gaz, args.max_tokens)
            lp = score_candidates(model, tok, p, cands)
            order = np.argsort(lp)[::-1]
            results.append({
                "sid": r["sid"], "fold": f, "gold": r["gold"],
                "ranker_top": r["cands"][0]["tic"], "llm_top": cands[order[0]]["tic"],
                "llm_margin": float(lp[order[0]] - lp[order[1]]) if len(lp) > 1 else 99.0,
                "in_topk": r["gold"] in [c["tic"] for c in cands],
                "llm_logp": {c["tic"]: v for c, v in zip(cands, lp)},
            })
        fr = [x for x in results if x["fold"] == f]
        print(f"  fold {f + 1}: ranker {100 * np.mean([x['ranker_top'] == x['gold'] for x in fr]):.2f}"
              f"  ->  {size} {100 * np.mean([x['llm_top'] == x['gold'] for x in fr]):.2f}"
              f"   ({(time.time() - t0) / 60:.0f} min so far)", flush=True)
        del model
        torch.cuda.empty_cache()

    path = os.path.join(OUT, f"results_{size}_k{args.k}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "results": results}, fh)

    c = np.array([x["llm_top"] == x["gold"] for x in results])
    m = np.array([x["llm_margin"] for x in results])
    print(f"\nOVER {len(results)} TEST SEGMENTS")
    print(f"  ranker alone       {100 * np.mean([x['ranker_top'] == x['gold'] for x in results]):.2f}")
    print("  1.5B model         85.88  (experiments/)")
    print(f"  {size} model       {100 * c.mean():.2f}")
    print(f"  ceiling (top-{args.k})    {100 * np.mean([x['in_topk'] for x in results]):.2f}")
    for qq in (0.1, 0.2, 0.3):
        keep = m >= np.quantile(m, qq)
        print(f"  skip {int(100 * qq)}%: precision {100 * c[keep].mean():.2f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
