"""Train and evaluate an RNN encoder-decoder that solves SRL as question answering.

BiLSTM encoder + attention LSTM decoder, GloVe-300d (frozen) embeddings,
teacher forcing 0.5, beam search (size 16) for the final evaluation.

Example:
    python scripts/train_seq2seq.py --glove data/glove/glove.6B.300d.txt --out results/rnn_seq2seq
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srl import get_device, set_seed  # noqa: E402
from srl.data import BOS, EOS, PAD, QADataset, Vocab, create_qa_pairs, load_glove_matrix, load_split, make_qa_collate  # noqa: E402
from srl.metrics import answer_scores, role_answer_table  # noqa: E402
from srl.models import Seq2Seq  # noqa: E402

SPECIAL = {PAD, BOS, EOS}


def strip(ids, vocab, eos_id):
    out = []
    for i in ids:
        if i == eos_id:
            break
        w = vocab.id2word[i]
        if w not in SPECIAL:
            out.append(w)
    return out


def run_epoch(model, loader, device, criterion, pad_id, optimizer=None, teacher_forcing=0.5, clip=1.0):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    correct_real = n_real = 0
    with torch.set_grad_enabled(training):
        for src, src_len, trg in tqdm(loader, leave=False, disable=not training):
            src, trg = src.to(device), trg.to(device)
            logits = model(src, src_len, trg, teacher_forcing if training else 1.0)
            gold = trg[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total += loss.item(); n += 1
            pred = logits.argmax(-1)
            real = gold != pad_id
            correct_real += (pred[real] == gold[real]).sum().item(); n_real += real.sum().item()
    return {"loss": total / n, "token_acc": correct_real / n_real}


@torch.no_grad()
def decode_all(model, questions, vocab, device, bos_id, eos_id, beam_size=0, batch_size=256):
    """Greedy (beam_size=0, batched) or beam search (one question at a time)."""
    model.eval()
    preds = []
    if beam_size:
        for q in tqdm(questions, desc=f"beam={beam_size}", leave=False):
            src, src_len = vocab.to_input_tensor([q])
            ids = model.beam_search(src.to(device), src_len, bos_id, eos_id, beam_size, max_len=len(q))
            preds.append(strip(ids, vocab, eos_id))
        return preds
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        src, src_len = vocab.to_input_tensor(batch)
        out = model.greedy(src.to(device), src_len, bos_id, eos_id, max_len=max(len(q) for q in batch))
        preds.extend(strip(ids, vocab, eos_id) for ids in out)
    return preds


def gold_answers(answers):
    return [[w for w in a if w not in SPECIAL] for a in answers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--glove", default="data/glove/glove.6B.300d.txt")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--teacher-forcing", type=float, default=0.5)
    ap.add_argument("--beam-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--beam-device", default="cpu", help="beam search runs one example at a time; CPU is faster")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    train = load_split(os.path.join(args.data_dir, "train.json"))
    valid = load_split(os.path.join(args.data_dir, "valid.json"))
    tq, ta, _ = create_qa_pairs(train)
    vq, va, vmeta = create_qa_pairs(valid)
    vocab = Vocab.from_corpus(tq + ta, size=20000, remove_frac=0.3, freq_cutoff=1)
    pad_id, bos_id, eos_id = vocab[PAD], vocab[BOS], vocab[EOS]
    emb, found = load_glove_matrix(args.glove, vocab)
    print(f"QA pairs: train {len(tq)}, valid {len(vq)}; vocab {len(vocab)}; GloVe vectors found for {found}")

    collate = make_qa_collate(vocab)
    train_loader = DataLoader(QADataset(tq, ta), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(QADataset(vq, va), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = Seq2Seq(emb, len(vocab), args.hidden_dim, pad_id).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gold = gold_answers(va)

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, criterion, pad_id, optimizer, args.teacher_forcing)
        ev = run_epoch(model, valid_loader, device, criterion, pad_id)
        greedy = decode_all(model, vq, vocab, device, bos_id, eos_id)
        scores = [answer_scores(g, p) for g, p in zip(gold, greedy)]
        ev["greedy_exact_match"] = sum(s[0] for s in scores) / len(scores)
        ev["greedy_token_f1"] = sum(s[1] for s in scores) / len(scores)
        history.append({"epoch": epoch, "train": tr, "valid": ev})
        print(f"epoch {epoch:2d} | train loss {tr['loss']:.4f} acc {tr['token_acc']:.4f} | valid loss {ev['loss']:.4f} "
              f"acc {ev['token_acc']:.4f} | greedy EM "
              f"{ev['greedy_exact_match']:.4f} F1 {ev['greedy_token_f1']:.4f}", flush=True)
    train_time = time.time() - start
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))

    model.to(args.beam_device)
    t0 = time.time()
    beam = decode_all(model, vq, vocab, args.beam_device, bos_id, eos_id, beam_size=args.beam_size)
    beam_time = time.time() - t0
    greedy = decode_all(model, vq, vocab, args.beam_device, bos_id, eos_id)

    with open(os.path.join(args.out, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for (idx, role), g, pb, pg in zip(vmeta, gold, beam, greedy):
            f.write(json.dumps({"idx": idx, "role": role, "gold": g, "pred_beam": pb, "pred_greedy": pg}) + "\n")

    oov = sum(1 for g in gold for w in g if w not in vocab.word2id)
    summary = {
        "config": vars(args) | {"device": str(device), "vocab_size": len(vocab), "glove_found": found,
                                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "train_time_sec": round(train_time, 1),
        "beam_time_sec": round(beam_time, 1),
        "history": history,
        "final_valid": {
            "token_acc": history[-1]["valid"]["token_acc"],
            "beam": role_answer_table(zip([r for _, r in vmeta], gold, beam)),
            "greedy": role_answer_table(zip([r for _, r in vmeta], gold, greedy)),
            "gold_answer_tokens_outside_vocab": oov,
            "gold_answer_tokens_total": sum(len(g) for g in gold),
        },
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_history(history, os.path.join(args.out, "curves.png"))
    b = summary["final_valid"]["beam"]["ALL"]
    print(f"done: train {train_time:.0f}s, beam {beam_time:.0f}s; beam EM {b['exact_match']:.4f} token-F1 {b['token_f1']:.4f}")


def plot_history(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(ep, [h["train"]["loss"] for h in history], "o-", label="train")
    ax[0].plot(ep, [h["valid"]["loss"] for h in history], "o-", label="valid")
    ax[0].set_title("loss (padding ignored)"); ax[0].legend()
    ax[1].plot(ep, [h["train"]["token_acc"] for h in history], "o-", label="train")
    ax[1].plot(ep, [h["valid"]["token_acc"] for h in history], "o-", label="valid")
    ax[1].set_title("next-token accuracy (teacher forced)"); ax[1].legend()
    ax[2].plot(ep, [h["valid"]["greedy_exact_match"] for h in history], "o-", label="exact match")
    ax[2].plot(ep, [h["valid"]["greedy_token_f1"] for h in history], "o-", label="token F1")
    ax[2].set_title("valid answers (greedy)"); ax[2].legend()
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.suptitle("Encoder-decoder (BiLSTM + attention) SRL-as-QA")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
