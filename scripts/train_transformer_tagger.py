"""Fine-tune a pre-trained transformer encoder (default: RoBERTa-base) as a
predicate-aware SRL tagger.

* Words are split into sub-tokens; each word is represented by its first sub-token.
* The predicate is marked with a learned indicator embedding added to the input
  embeddings (initialised at zero, so training starts from the pre-trained model).
* Each word's state is concatenated with the predicate's state and classified
  (softmax or the BIO-constrained CRF from srl/crf.py).

    python scripts/train_transformer_tagger.py --out results/roberta_base
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srl import get_device, set_seed  # noqa: E402
from srl.crf import CRF, bio_constraints  # noqa: E402
from srl.data import ID2LABEL, IGNORE, LABEL2ID, ROLES, load_split  # noqa: E402
from srl.metrics import role_answer_table, span_f1, tokens_for_role  # noqa: E402


class TransformerTagger(nn.Module):
    def __init__(self, name, num_labels, crf=False, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.predicate = nn.Embedding(2, h)
        nn.init.zeros_(self.predicate.weight)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * h, num_labels)
        self.crf = CRF(num_labels, *bio_constraints(ID2LABEL)) if crf else None

    def forward(self, input_ids, attention_mask, pred_flag, word_index, word_mask, verb_word):
        emb = self.encoder.get_input_embeddings()(input_ids) + self.predicate(pred_flag)
        h = self.encoder(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state
        rows = torch.arange(h.size(0), device=h.device).unsqueeze(1)
        words = h[rows, word_index]                                  # [B, n_words, H] first sub-tokens
        verb = words[torch.arange(h.size(0), device=h.device), verb_word].unsqueeze(1).expand_as(words)
        return self.out(self.dropout(torch.cat([words, verb], dim=-1)))


def encode(split, tok, max_len=256):
    items = []
    tags = split.get("srl_frames")
    for i, (words, v) in enumerate(zip(split["text"], split["verb_index"])):
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=max_len)
        wid = enc.word_ids()
        first = {}
        for pos, w in enumerate(wid):
            if w is not None and w not in first:
                first[w] = pos
        items.append({
            "input_ids": enc["input_ids"],
            "pred_flag": [1 if w == v else 0 for w in wid],
            "word_index": [first[w] for w in range(len(words))],
            "verb_word": v,
            "labels": [LABEL2ID[t] for t in tags[i]] if tags else [0] * len(words),
        })
    return items


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    W = max(len(b["word_index"]) for b in batch)
    out = {k: [] for k in ("input_ids", "attention_mask", "pred_flag", "word_index", "word_mask", "labels")}
    for b in batch:
        n, m = len(b["input_ids"]), len(b["word_index"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * (L - n))
        out["attention_mask"].append([1] * n + [0] * (L - n))
        out["pred_flag"].append(b["pred_flag"] + [0] * (L - n))
        out["word_index"].append(b["word_index"] + [0] * (W - m))
        out["word_mask"].append([True] * m + [False] * (W - m))
        out["labels"].append(b["labels"] + [IGNORE] * (W - m))
    t = {k: torch.tensor(v) for k, v in out.items()}
    t["verb_word"] = torch.tensor([b["verb_word"] for b in batch])
    return t


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(b["input_ids"], b["attention_mask"], b["pred_flag"], b["word_index"], b["word_mask"], b["verb_word"])
            if model.crf is not None:
                paths = model.crf.decode(logits, b["word_mask"])
            else:
                paths = [row[: int(m.sum())] for row, m in zip(logits.argmax(-1).tolist(), b["word_mask"])]
            preds.extend([[ID2LABEL[x] for x in p] for p in paths])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--crf", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    train, valid = load_split(f"{args.data_dir}/train.json"), load_split(f"{args.data_dir}/valid.json")
    tr_items, va_items = encode(train, tok), encode(valid, tok)
    col = lambda b: collate(b, tok.pad_token_id)  # noqa: E731
    tr_loader = DataLoader(tr_items, batch_size=args.batch_size, shuffle=True, collate_fn=col)
    va_loader = DataLoader(va_items, batch_size=64, shuffle=False, collate_fn=col)

    model = TransformerTagger(args.model, len(LABEL2ID), crf=args.crf).to(device)
    enc_params = list(model.encoder.parameters())
    other = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": other, "lr": args.head_lr}], weight_decay=0.01)
    total = args.epochs * len(tr_loader)
    warm = int(args.warmup * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    history, t0 = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for step, b in enumerate(tr_loader):
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(b["input_ids"], b["attention_mask"], b["pred_flag"], b["word_index"], b["word_mask"], b["verb_word"])
            loss = model.crf.nll(logits, b["labels"], b["word_mask"]) if model.crf is not None else \
                ce(logits.reshape(-1, logits.size(-1)), b["labels"].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            loss_sum += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(tr_loader)} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        preds = predict(model, va_loader, device)
        f1 = span_f1(valid["srl_frames"], preds)["micro_f1"]
        history.append({"epoch": epoch, "train_loss": loss_sum / len(tr_loader), "valid_span_f1": f1})
        print(f"epoch {epoch}: train loss {loss_sum/len(tr_loader):.4f}, valid span-F1 {f1:.4f}", flush=True)
    train_time = time.time() - t0

    spans = span_f1(valid["srl_frames"], preds)
    records = []
    with open(os.path.join(args.out, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for i, (toks, verb, gold, pred) in enumerate(zip(valid["text"], valid["verb_index"], valid["srl_frames"], preds)):
            f.write(json.dumps({"idx": i, "tokens": toks, "verb_index": verb, "gold": gold, "pred": pred}) + "\n")
            for role in ROLES:
                records.append((role, tokens_for_role(toks, gold, role), tokens_for_role(toks, pred, role)))
    gold_flat = [t for s in valid["srl_frames"] for t in s]
    pred_flat = [t for s in preds for t in s]
    summary = {
        "config": vars(args) | {"device": str(device),
                                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "train_time_sec": round(train_time, 1),
        "history": history,
        "final_valid": {"token_acc": sum(a == b for a, b in zip(gold_flat, pred_flat)) / len(gold_flat),
                        "span_micro_f1": spans["micro_f1"], "span_report": spans["report"],
                        "role_answers": role_answer_table(records)},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"done in {train_time:.0f}s; span micro-F1 = {spans['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
