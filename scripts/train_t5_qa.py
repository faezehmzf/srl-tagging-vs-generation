"""SRL as question answering with a pre-trained text-to-text model (default: FLAN-T5-base).

Each (sentence, predicate) becomes five questions, one per role, e.g.
    input : "role: ARG0 | predicate: accept | sentence: He would n't accept anything of value ."
    target: "He"                       (or "no answer" if the role is absent)
Answers are scored with the same token-level metrics as the RNN encoder-decoder
(srl/metrics.py), so the two generators are directly comparable.

    python scripts/train_t5_qa.py --out results/flan_t5_base_qa
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srl import get_device, set_seed  # noqa: E402
from srl.data import ROLES, load_split, role_of  # noqa: E402
from srl.metrics import answer_scores, role_answer_table  # noqa: E402

EMPTY = "no answer"


def make_pairs(split):
    src, tgt, gold, meta = [], [], [], []
    for i, (words, v, tags) in enumerate(zip(split["text"], split["verb_index"], split["srl_frames"])):
        sent = " ".join(words)
        for role in ROLES:
            ans = [w for w, t in zip(words, tags) if role_of(t) == role]
            src.append(f"role: {role} | predicate: {words[v]} | sentence: {sent}")
            tgt.append(" ".join(ans) if ans else EMPTY)
            gold.append(ans)
            meta.append((i, role))
    return src, tgt, gold, meta


def to_tokens(text):
    text = text.strip()
    return [] if text == EMPTY or not text else text.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch-size x grad-accum")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-train", type=int, default=0, help="debug: use only the first N training pairs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    tr_src, tr_tgt, _, _ = make_pairs(load_split(f"{args.data_dir}/train.json"))
    if args.limit_train:
        tr_src, tr_tgt = tr_src[: args.limit_train], tr_tgt[: args.limit_train]
    va_src, va_tgt, va_gold, va_meta = make_pairs(load_split(f"{args.data_dir}/valid.json"))

    def collate(batch):
        s, t = zip(*batch)
        enc = tok(list(s), padding=True, truncation=True, max_length=192, return_tensors="pt")
        lab = tok(list(t), padding=True, truncation=True, max_length=64, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        enc["labels"] = lab
        return enc

    loader = DataLoader(list(zip(tr_src, tr_tgt)), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = args.epochs * math.ceil(len(loader) / args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total))

    history, t0 = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        opt.zero_grad()
        for step, b in enumerate(loader):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            loss_sum += loss.item()
            if step % 800 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(loader)})
    train_time = time.time() - t0

    model.eval()
    preds, t1 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(va_src), 32):
            enc = tok(va_src[i:i + 32], padding=True, truncation=True, max_length=192, return_tensors="pt").to(device)
            out = model.generate(**enc, max_new_tokens=64, num_beams=args.num_beams)
            preds.extend(tok.batch_decode(out, skip_special_tokens=True))
    gen_time = time.time() - t1

    pred_tokens = [to_tokens(p) for p in preds]
    with open(os.path.join(args.out, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for (idx, role), g, p, raw in zip(va_meta, va_gold, pred_tokens, preds):
            f.write(json.dumps({"idx": idx, "role": role, "gold": g, "pred_beam": p, "raw": raw}) + "\n")
    table = role_answer_table(zip([r for _, r in va_meta], va_gold, pred_tokens))
    summary = {"config": vars(args) | {"device": str(device),
                                       "trainable_params": sum(p.numel() for p in model.parameters())},
               "train_time_sec": round(train_time, 1), "generation_time_sec": round(gen_time, 1),
               "history": history, "final_valid": {"answers": table}}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    a = table["ALL"]
    print(f"done: train {train_time:.0f}s; EM {a['exact_match']:.4f}, token-F1 when present {a['token_f1_when_present']:.4f}")


if __name__ == "__main__":
    main()
