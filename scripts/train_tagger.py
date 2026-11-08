"""Train and evaluate a predicate-aware RNN tagger for semantic role labeling.

Examples:
    # baseline: one unidirectional LSTM layer, 64 units
    python scripts/train_tagger.py --cell lstm --out results/lstm_baseline
    # BiLSTM-CRF with GloVe and a predicate indicator
    python scripts/train_tagger.py --bidirectional --verb-flag-dim 16 --glove data/glove/glove.6B.300d.txt \\
        --hidden-dim 128 --num-layers 2 --dropout 0.3 --crf --epochs 15 --out results/bilstm_crf_glove
    # common mistake: score padding positions as 'O' (inflates accuracy)
    python scripts/train_tagger.py --combine add --count-padding --out results/pitfall_padding
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srl import get_device, set_seed  # noqa: E402
from srl.data import (ID2LABEL, IGNORE, LABEL2ID, ROLES, TaggingDataset, Vocab, load_glove_matrix,  # noqa: E402
                      load_split, make_tagging_collate)
from srl.metrics import role_answer_table, span_f1, tokens_for_role  # noqa: E402
from srl.models import RNNTagger  # noqa: E402


def run_epoch(model, loader, device, criterion, count_padding, optimizer=None, clip=5.0):
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0
    correct_real = n_real = correct_all = n_all = 0
    preds_out = []
    with torch.set_grad_enabled(training):
        for ids, verbs, labels, lengths in loader:
            ids, verbs, labels = ids.to(device), verbs.to(device), labels.to(device)
            logits = model(ids, verbs, lengths)
            real = labels != IGNORE
            if model.crf is not None:
                loss = model.crf.nll(logits, labels, real)
            else:
                # --count-padding: padding positions labelled 'O' and trained/scored like real tokens
                target = labels.masked_fill(~real, LABEL2ID["O"]) if count_padding else labels
                loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item(); n_batches += 1
            if model.crf is not None:  # Viterbi path (also during training, for comparable accuracy)
                paths = model.crf.decode(logits, real)
                pred = torch.zeros_like(labels)
                for b, p in enumerate(paths):
                    pred[b, : len(p)] = torch.tensor(p, device=device)
            else:
                pred = logits.argmax(-1)
            correct_real += (pred[real] == labels[real]).sum().item(); n_real += real.sum().item()
            correct_all += (pred == labels.masked_fill(~real, LABEL2ID["O"])).sum().item(); n_all += labels.numel()
            if not training:
                for p, n in zip(pred.cpu().tolist(), lengths.tolist()):
                    preds_out.append([ID2LABEL[x] for x in p[:n]])
    return {"loss": total_loss / n_batches, "token_acc": correct_real / n_real,
            "token_acc_incl_padding": correct_all / n_all}, preds_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    ap.add_argument("--combine", choices=["concat", "add"], default="concat")
    ap.add_argument("--count-padding", action="store_true", help="score padding as 'O' (inflates token accuracy)")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--verb-flag-dim", type=int, default=0)
    ap.add_argument("--glove", default=None, help="initialise word embeddings from this GloVe file")
    ap.add_argument("--crf", action="store_true")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    train = load_split(os.path.join(args.data_dir, "train.json"))
    valid = load_split(os.path.join(args.data_dir, "valid.json"))
    vocab = Vocab.from_corpus(train["text"], size=20000, remove_frac=0.3, freq_cutoff=1)
    collate = make_tagging_collate(vocab)
    train_loader = DataLoader(TaggingDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(TaggingDataset(valid), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    emb = load_glove_matrix(args.glove, vocab)[0] if args.glove else None

    model = RNNTagger(len(vocab), len(LABEL2ID), args.emb_dim, args.hidden_dim, args.cell, args.combine,
                      args.bidirectional, args.verb_flag_dim, args.num_layers, args.dropout, emb,
                      args.crf, ID2LABEL).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr, _ = run_epoch(model, train_loader, device, criterion, args.count_padding, optimizer)
        va, preds = run_epoch(model, valid_loader, device, criterion, args.count_padding)
        va["span_f1"] = span_f1(valid["srl_frames"], preds)["micro_f1"]
        history.append({"epoch": epoch, "train": tr, "valid": va})
        print(f"epoch {epoch:2d} | train loss {tr['loss']:.4f} acc {tr['token_acc']:.4f} | "
              f"valid acc {va['token_acc']:.4f} (incl. padding {va['token_acc_incl_padding']:.4f}) "
              f"span-F1 {va['span_f1']:.4f}", flush=True)
    train_time = time.time() - start

    # Reported numbers = last epoch (no model selection on the validation set).
    spans = span_f1(valid["srl_frames"], preds)
    records = []
    with open(os.path.join(args.out, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for i, (toks, verb, gold, pred) in enumerate(zip(valid["text"], valid["verb_index"], valid["srl_frames"], preds)):
            f.write(json.dumps({"idx": i, "tokens": toks, "verb_index": verb, "gold": gold, "pred": pred}) + "\n")
            for role in ROLES:
                records.append((role, tokens_for_role(toks, gold, role), tokens_for_role(toks, pred, role)))
    summary = {
        "config": vars(args) | {"device": str(device), "vocab_size": len(vocab),
                                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "train_time_sec": round(train_time, 1),
        "history": history,
        "final_valid": {
            "token_acc": history[-1]["valid"]["token_acc"],
            "token_acc_incl_padding": history[-1]["valid"]["token_acc_incl_padding"],
            "span_micro_f1": spans["micro_f1"],
            "span_report": spans["report"],
            "role_answers": role_answer_table(records),
        },
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    plot_history(history, os.path.join(args.out, "curves.png"), os.path.basename(args.out.rstrip("/")))
    print(f"done in {train_time:.0f}s; span micro-F1 = {spans['micro_f1']:.4f}")


def plot_history(history, path, title):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(ep, [h["train"]["loss"] for h in history], "o-", label="train")
    ax[0].plot(ep, [h["valid"]["loss"] for h in history], "o-", label="valid")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(ep, [h["train"]["token_acc"] for h in history], "o-", label="train")
    ax[1].plot(ep, [h["valid"]["token_acc"] for h in history], "o-", label="valid")
    ax[1].plot(ep, [h["valid"]["token_acc_incl_padding"] for h in history], "--", label="valid incl. padding")
    ax[1].set_title("token accuracy"); ax[1].legend()
    ax[2].plot(ep, [h["valid"]["span_f1"] for h in history], "o-", color="C3")
    ax[2].set_title("valid span micro-F1")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
