"""Evaluation helpers.

* Span-level F1 (CoNLL style, via seqeval) is the standard SRL metric.
* Role-answer metrics compare, for each (sentence, role), the set of tokens a
  model assigns to that role with the gold tokens. This is the only metric that
  can be computed identically for the tagger and the seq2seq (QA) model.
"""
from collections import Counter, defaultdict

from seqeval.metrics import classification_report, f1_score

from .data import ROLES, role_of


def span_f1(gold_tags, pred_tags):
    """gold_tags / pred_tags: list of tag-string lists (real tokens only)."""
    return {
        "micro_f1": f1_score(gold_tags, pred_tags),
        "report": classification_report(gold_tags, pred_tags, output_dict=True, zero_division=0),
    }


def tokens_for_role(tokens, tags, role):
    return [w for w, t in zip(tokens, tags) if role_of(t) == role]


def answer_scores(gold, pred):
    """Exact match and SQuAD-style token F1 between two token lists.
    Two empty answers (role absent and predicted absent) count as correct."""
    if not gold and not pred:
        return 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0
    overlap = sum((Counter(gold) & Counter(pred)).values())
    if overlap == 0:
        return float(gold == pred), 0.0
    p, r = overlap / len(pred), overlap / len(gold)
    return float(gold == pred), 2 * p * r / (p + r)


def role_answer_table(records):
    """records: iterable of (role, gold_tokens, pred_tokens). Returns per-role and
    overall exact match / token F1, plus F1 restricted to roles that are present."""
    acc = defaultdict(lambda: {"n": 0, "em": 0.0, "f1": 0.0, "n_present": 0, "f1_present": 0.0})
    for role, gold, pred in records:
        em, f1 = answer_scores(gold, pred)
        for key in (role, "ALL"):
            a = acc[key]
            a["n"] += 1; a["em"] += em; a["f1"] += f1
            if gold:
                a["n_present"] += 1; a["f1_present"] += f1
    out = {}
    for key in ROLES + ["ALL"]:
        a = acc[key]
        if a["n"]:
            out[key] = {
                "n": a["n"],
                "exact_match": a["em"] / a["n"],
                "token_f1": a["f1"] / a["n"],
                "n_present": a["n_present"],
                "token_f1_when_present": a["f1_present"] / a["n_present"] if a["n_present"] else None,
            }
    return out
