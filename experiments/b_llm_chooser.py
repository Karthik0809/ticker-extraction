"""EXPERIMENT, step B: fine-tune a small language model to pick from the shortlist.

    python experiments/b_llm_chooser.py --k 5 --aliases     (the 85.88 result)

Reads experiments/out/rank_lists.json from step A. For each of the same 5
video-grouped folds:

  1. Build a prompt per segment: the transcript, the video title, where the
     segment sits in the video, and the ranker's top-K candidates with their
     company names.
  2. Fine-tune Qwen2.5-1.5B-Instruct with LoRA on the training folds, teaching
     it to answer with the correct ticker.
  3. On the test fold, score every candidate by how likely the model thinks it
     is the answer, and pick the highest. The model never writes free text, so
     the answer is always a real ticker from the list.

LoRA trains a small add-on (a few million numbers) instead of all 1.5 billion,
which is what makes this fit on an 8 GB laptop GPU.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch  # must load before pandas on this machine, see src/1_prepare.py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from core import load_gazetteer, load_segments  # noqa: E402

from peft import LoraConfig, get_peft_model                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

OUT = os.path.join(ROOT, "experiments", "out")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = ("You identify which stock a finance YouTube segment is recommending. "
          "Answer with exactly one ticker from the candidate list.")


def seg_context(df):
    """For each segment: 'segment i of n' within its own video."""
    order = df.sort_values(["video_id", "start"]).groupby("video_id").cumcount() + 1
    count = df.groupby("video_id").video_id.transform("size")
    return {i: (int(order[i]), int(count[i])) for i in df.index}


OCR = {}


def screen_line(row):
    """What OCR read off this segment's video frames, as one short line.

    The ticker is often printed in a chart header on screen even when nobody
    says it, so this is the strongest clue the transcript cannot give.
    """
    if not OCR:
        return ""
    ev = OCR.get(f"{row.video_id}_{float(row.start):.2f}")
    if not ev:
        return "Screen: video no longer available.\n"
    syms = ev.get("symbols") or {}
    if not syms:
        return "Screen: no tickers read.\n"
    parts = []
    for s, v in sorted(syms.items(), key=lambda kv: -kv[1]["frames"])[:6]:
        how = "chart header" if set(v["how"]) & {"parens", "exchange"} else "text"
        parts.append(f"{s} ({how}, {v['frames']} of {ev['n_frames']} frames)")
    return "Tickers read on screen: " + "; ".join(parts) + "\n"


USE_DESC = []


def desc_line(row, cands, gaz):
    """Where each candidate first appears in the video description.

    Finfluencer videos are countdowns and the description usually lists the
    stocks in order, so "segment 3 of 5" plus "listed 3rd" is a strong hint.
    """
    if not USE_DESC:
        return ""
    import re
    from core import name_tokens
    desc = str(row.get("videoDescription") or "")
    low = desc.lower()
    first = {}
    for c in cands:
        t = c["tic"]
        hits = [m.start() for m in re.finditer(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", desc)]
        for tok in name_tokens(gaz.get(t, "")):
            if len(tok) >= 4 and tok in low:
                hits.append(low.index(tok))
        if hits:
            first[t] = min(hits)
    if not first:
        return "Description: none of the candidates are listed.\n"
    order = sorted(first, key=first.get)
    ords = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, 21)]
    return ("Order the candidates appear in the description: "
            + ", ".join(f"{t} ({ords[i]})" for i, t in enumerate(order)) + "\n")


def make_prompt(row, pos, cands, gaz, max_chars):
    text = " ".join(str(row.segment_transcript).split())
    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    lines = [f"- {c['tic']}: {gaz.get(c['tic'], 'unknown company')}" for c in cands]
    return (f"Video title: {row.video_title}\n"
            f"This is segment {pos[0]} of {pos[1]} in the video.\n"
            f"{screen_line(row)}{desc_line(row, cands, gaz)}\n"
            f"Segment transcript:\n{text}\n\n"
            f"Candidates:\n" + "\n".join(lines) + "\n\n"
            "Which ticker is being recommended in this segment?")


def encode(tok, prompt, answer=None):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(p, add_special_tokens=False).input_ids
    if answer is None:
        return p_ids
    a_ids = tok(answer + "<|im_end|>", add_special_tokens=False).input_ids
    return p_ids, a_ids


def load_model(args):
    """bf16 for the 1.5B model. For bigger models, --4bit stores the frozen
    weights in 4 bits (QLoRA), so a 3B model takes about 2 GB instead of 6.3."""
    if not args.four_bit:
        return AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=q,
                                             dtype=torch.bfloat16, device_map={"": 0})
    return prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)


def train_fold(examples, tok, args, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = load_model(args)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    steps_total = math.ceil(len(examples) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps_total))

    model.train()
    step = 0
    for ep in range(args.epochs):
        random.shuffle(examples)
        run = 0.0
        for i, (p_ids, a_ids) in enumerate(examples):
            ids = torch.tensor([p_ids + a_ids]).cuda()
            # Only compute predictions for the answer tokens. Computing them
            # over the full 152k-word vocabulary for every prompt position
            # needs about 11 GB and spills out of the 8 GB card.
            logits = model(input_ids=ids, logits_to_keep=len(a_ids) + 1).logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits[0, :-1], torch.tensor(a_ids).cuda()) / args.accum
            loss.backward()
            run += loss.item() * args.accum
            if (i + 1) % args.accum == 0 or i == len(examples) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
        print(f"    epoch {ep + 1}: mean loss {run / len(examples):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def score_candidates(model, tok, p_ids, cands):
    """Log-probability of each candidate being the complete answer.

    The end-of-turn token is included, so a short ticker cannot win just by
    being a prefix of a longer one.
    """
    out = []
    for c in cands:
        a = tok(c["tic"] + "<|im_end|>", add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + a]).cuda()
        logp = torch.log_softmax(
            model(input_ids=ids, logits_to_keep=len(a) + 1).logits.float()[0, :-1], -1)
        out.append(float(logp.gather(1, torch.tensor(a).cuda()[:, None]).sum()))
    return out


def fit_prompt(tok, row, pos, cands, gaz, max_chars, max_tokens=700):
    """Shorten the transcript until the prompt fits, so it stays on the GPU."""
    while True:
        prompt = make_prompt(row, pos, cands, gaz, max_chars)
        p_ids = encode(tok, prompt)
        if len(p_ids) <= max_tokens or max_chars <= 300:
            return p_ids
        max_chars -= 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="candidates shown to the model")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_chars", type=int, default=1800)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--aliases", action="store_true",
                    help="use the shortlists built with Wikidata aliases")
    ap.add_argument("--screen", action="store_true",
                    help="tell the model which tickers OCR read on screen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--desc", action="store_true",
                    help="tell the model the order candidates appear in the description")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    args = ap.parse_args()
    size = args.model.split("-")[1] if args.model != MODEL else ""
    tag = ("_alias" if args.aliases else "") + ("_screen" if args.screen else "") \
        + ("_desc" if args.desc else "") \
        + (f"_s{args.seed}" if args.seed else "") + (f"_{size}" if size else "")
    if args.desc:
        USE_DESC.append(True)
    if args.screen:
        with open(os.path.join(ROOT, "data", "ocr_evidence.json"), encoding="utf-8") as f:
            OCR.update(json.load(f))

    df = load_segments()
    gaz = load_gazetteer()
    pos = seg_context(df)
    lists_tag = "_alias" if args.aliases else ""
    with open(os.path.join(OUT, f"rank_lists{lists_tag}.json"), encoding="utf-8") as f:
        lists = json.load(f)

    tok = AutoTokenizer.from_pretrained(args.model)
    results = []
    t_start = time.time()
    for f in [int(x) for x in args.folds.split(",")]:
        fold = lists["folds"][f]
        print(f"\nFOLD {f + 1}/5: {len(fold['train'])} train, {len(fold['test'])} test",
              flush=True)

        # training: only segments whose answer is in their top-K, otherwise
        # there is no correct option to learn to pick
        ex = []
        for r in fold["train"]:
            cands = r["cands"][:args.k]
            if r["gold"] not in [c["tic"] for c in cands]:
                continue
            p_ids = fit_prompt(tok, df.loc[r["sid"]], pos[r["sid"]], cands, gaz, args.max_chars)
            a_ids = tok(r["gold"] + "<|im_end|>", add_special_tokens=False).input_ids
            ex.append((p_ids, a_ids))
        print(f"  training on {len(ex)} segments", flush=True)
        model = train_fold(ex, tok, args, seed=f + 100 * args.seed)

        for r in fold["test"]:
            cands = r["cands"][:args.k]
            p_ids = fit_prompt(tok, df.loc[r["sid"]], pos[r["sid"]], cands, gaz, args.max_chars)
            lp = score_candidates(model, tok, p_ids, cands)
            order = np.argsort(lp)[::-1]
            results.append({
                "sid": r["sid"], "fold": f, "gold": r["gold"],
                "ranker_top": r["cands"][0]["tic"],
                "llm_top": cands[order[0]]["tic"],
                "llm_margin": float(lp[order[0]] - lp[order[1]]) if len(lp) > 1 else 99.0,
                "in_topk": r["gold"] in [c["tic"] for c in cands],
                "llm_logp": {c["tic"]: v for c, v in zip(cands, lp)},
            })
        acc_r = np.mean([x["ranker_top"] == x["gold"] for x in results if x["fold"] == f])
        acc_l = np.mean([x["llm_top"] == x["gold"] for x in results if x["fold"] == f])
        print(f"  fold {f + 1}: ranker {100 * acc_r:.2f}  ->  LLM {100 * acc_l:.2f}"
              f"   ({(time.time() - t_start) / 60:.1f} min so far)", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, f"llm_results{tag}_k{args.k}.json"), "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "results": results}, fh)

    n = len(results)
    r_acc = np.mean([x["ranker_top"] == x["gold"] for x in results])
    l_acc = np.mean([x["llm_top"] == x["gold"] for x in results])
    print(f"\nOVER {n} TEST SEGMENTS")
    print(f"  ranker alone : {100 * r_acc:.2f}")
    print(f"  LLM chooser  : {100 * l_acc:.2f}")
    print(f"  answer in top-{args.k}: {100 * np.mean([x['in_topk'] for x in results]):.2f}  (ceiling)")


if __name__ == "__main__":
    main()
