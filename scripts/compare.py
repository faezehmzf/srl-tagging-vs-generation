"""Compare all systems on the validation set (800 sentences x 5 roles).

* Span-level micro F1 (CoNLL style) for the taggers.
* A role-answer metric shared by taggers and generators: for every
  (sentence, role) the tokens a system assigns to the role are compared with
  the gold tokens (exact match, token F1, and token F1 restricted to roles that
  are present). An "always answer nothing" baseline shows how much of the exact
  match comes from absent roles.
* Diagnostics: invalid BIO transitions (taggers) and generated words that do
  not occur in the sentence (generators).
* A common evaluation mistake: token accuracy with padding counted.

Outputs: results/comparison.md, results/comparison.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srl.data import ROLES, load_split  # noqa: E402
from srl.metrics import role_answer_table, tokens_for_role  # noqa: E402

RES = "results"
TAGGERS = {
    "LSTM (baseline)": "lstm_baseline",
    "GRU (baseline)": "gru_baseline",
    "BiLSTM + predicate flag": "bilstm_flag",
    "BiLSTM + GloVe": "bilstm_glove",
    "BiLSTM-CRF + GloVe": "bilstm_glove_crf",
    "RoBERTa-base": "roberta_base",
    "RoBERTa-base + CRF": "roberta_base_crf",
}
GENERATORS = {
    "RNN encoder-decoder + attention (beam 16)": "rnn_seq2seq",
    "FLAN-T5-base (beam 4)": "flan_t5_base_qa",
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def invalid_transitions(tags):
    """Number of I-X tags not preceded by B-X or I-X (an ill-formed span)."""
    bad, prev = 0, "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            bad += 1
        prev = t
    return bad


def is_sublist(a, s):
    return any(s[j:j + len(a)] == a for j in range(len(s) - len(a) + 1))


def main():
    valid = load_split("data/valid.json")
    gold = {(i, r): tokens_for_role(t, tags, r)
            for i, (t, tags) in enumerate(zip(valid["text"], valid["srl_frames"])) for r in ROLES}
    systems, spans, params, times = {"always answer nothing": {k: [] for k in gold}}, {}, {}, {}
    invalid = {}
    for name, d in TAGGERS.items():
        p = os.path.join(RES, d, "predictions.jsonl")
        if not os.path.exists(p):
            continue
        rows = read_jsonl(p)
        systems[name] = {(r["idx"], role): tokens_for_role(r["tokens"], r["pred"], role) for r in rows for role in ROLES}
        invalid[name] = (sum(invalid_transitions(r["pred"]) for r in rows),
                         sum(invalid_transitions(r["pred"]) > 0 for r in rows))
        m = json.load(open(os.path.join(RES, d, "metrics.json")))
        spans[name] = m["final_valid"]["span_micro_f1"]
        params[name], times[name] = m["config"]["trainable_params"], m["train_time_sec"]
    for name, d in GENERATORS.items():
        p = os.path.join(RES, d, "predictions.jsonl")
        if not os.path.exists(p):
            continue
        systems[name] = {(r["idx"], r["role"]): r["pred_beam"] for r in read_jsonl(p)}
        m = json.load(open(os.path.join(RES, d, "metrics.json")))
        params[name], times[name] = m["config"]["trainable_params"], m["train_time_sec"]

    tables = {n: role_answer_table((k[1], gold[k], pred[k]) for k in gold) for n, pred in systems.items()}
    lines = ["# Comparison on the validation set (800 sentences)", "",
             "## Overview", "",
             "| system | span F1 | role exact match | token F1 (role present) | parameters | training time |",
             "|---|---|---|---|---|---|"]
    for n, t in tables.items():
        sp = f"{spans[n]:.3f}" if n in spans else "—"
        pr = f"{params[n]/1e6:.1f}M" if n in params else "—"
        tm = f"{times[n]/60:.1f} min" if n in times else "—"
        lines.append(f"| {n} | {sp} | {t['ALL']['exact_match']:.3f} | {t['ALL']['token_f1_when_present']:.3f} | {pr} | {tm} |")
    lines += ["", "## Token F1 per role (only (sentence, role) pairs where the role is present)", "",
              "| system | " + " | ".join(f"{r} (n={tables['always answer nothing'][r]['n_present']})" for r in ROLES) + " |",
              "|---" * (len(ROLES) + 1) + "|"]
    for n, t in tables.items():
        if n == "always answer nothing":
            continue
        lines.append(f"| {n} | " + " | ".join(f"{t[r]['token_f1_when_present']:.3f}" for r in ROLES) + " |")

    pit = os.path.join(RES, "pitfall_padding", "metrics.json")
    if os.path.exists(pit):
        m = json.load(open(pit))["final_valid"]
        lines += ["", "## Counting padding (a common evaluation mistake)", "",
                  "The same LSTM baseline evaluated three ways (padding scored as `O` during training and evaluation):", "",
                  f"- token accuracy including padding: **{m['token_acc_incl_padding']:.3f}**",
                  f"- token accuracy on real tokens: {m['token_acc']:.3f} (always predicting `O` gives 0.842)",
                  f"- span F1: **{m['span_micro_f1']:.3f}**"]

    gold_invalid = sum(invalid_transitions(t) for t in valid["srl_frames"])
    lines += ["", "## Diagnostics", "",
              f"Taggers: ill-formed spans (`I-X` not preceded by `B-X`/`I-X`; gold has {gold_invalid}).", "",
              "| tagger | invalid `I-` tags | sentences affected |", "|---|---|---|"]
    lines += [f"| {n} | {a} | {b} |" for n, (a, b) in invalid.items()]
    novel = {}
    for n in GENERATORS:
        if n not in systems:
            continue
        toks = [(i, w) for (i, _), ans in systems[n].items() for w in ans]
        miss = sum(w not in set(valid["text"][i]) for i, w in toks)
        nonempty = [a for a in systems[n].values() if a]
        sub = sum(is_sublist(a, valid["text"][i]) for (i, _), a in systems[n].items() if a)
        novel[n] = {"answer_tokens": len(toks), "not_in_sentence": miss,
                    "nonempty_answers": len(nonempty), "contiguous_substring": sub}
    if novel:
        lines += ["", "Generators: is the answer copied from the sentence?", "",
                  "| generator | answer tokens not in the sentence | non-empty answers that are a contiguous span of the sentence |",
                  "|---|---|---|"]
        lines += [f"| {n} | {v['not_in_sentence']}/{v['answer_tokens']} ({v['not_in_sentence']/max(1, v['answer_tokens']):.1%}) "
                  f"| {v['contiguous_substring']}/{v['nonempty_answers']} ({v['contiguous_substring']/max(1, v['nonempty_answers']):.1%}) |"
                  for n, v in novel.items()]

    names = [n for n in ("LSTM (baseline)", "BiLSTM-CRF + GloVe", "RoBERTa-base",
                         "RNN encoder-decoder + attention (beam 16)", "FLAN-T5-base (beam 4)") if n in systems]
    lines += ["", "## Examples (first short validation sentences with at least two roles)", ""]
    shown = 0
    for i, toks in enumerate(valid["text"]):
        if len(toks) > 14 or sum(bool(gold[(i, r)]) for r in ROLES) < 2:
            continue
        t = list(toks); v = valid["verb_index"][i]; t[v] = f"**[{t[v]}]**"
        lines += [" ".join(t), "", "| role | gold | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
        for r in ROLES:
            cells = [" ".join(systems[n][(i, r)]) or "∅" for n in names]
            if gold[(i, r)] or any(c != "∅" for c in cells):
                lines.append(f"| {r} | {' '.join(gold[(i, r)]) or '∅'} | " + " | ".join(cells) + " |")
        lines.append("")
        shown += 1
        if shown == 4:
            break

    with open(os.path.join(RES, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "comparison.json"), "w") as f:
        json.dump({"role_answers": tables, "span_micro_f1": spans, "params": params, "train_time_sec": times,
                   "invalid_transitions": invalid, "generator_copying": novel}, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
