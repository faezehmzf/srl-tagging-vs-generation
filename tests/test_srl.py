import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from srl.data import (BOS, EOS, IGNORE, LABEL2ID, Vocab, create_qa_pairs,  # noqa: E402
                      make_tagging_collate)
from srl.metrics import answer_scores, role_answer_table, span_f1  # noqa: E402
from srl.models import RNNTagger, Seq2Seq  # noqa: E402

SPLIT = {
    "text": [["He", "would", "n't", "accept", "anything", "of", "value", "."], ["A", "dog", "barked", "today"]],
    "verb_index": [3, 2],
    "srl_frames": [["B-ARG0", "O", "O", "O", "B-ARG1", "I-ARG1", "I-ARG1", "O"], ["B-ARG0", "I-ARG0", "O", "B-ARGM-TMP"]],
}


def test_vocab_specials_cap_and_unknown():
    v = Vocab.from_corpus([["a", "a", "b", "b", "c"]], size=10, remove_frac=0.0, freq_cutoff=1)
    assert [v.id2word[i] for i in range(4)] == ["Pad", "Start", "End", "Unknown"]
    assert "a" in v.word2id and "b" in v.word2id and "c" not in v.word2id  # c appears once
    assert v["zzz"] == v["Unknown"]
    capped = Vocab.from_corpus([["a", "a", "b", "b"]], size=2, remove_frac=0.5, freq_cutoff=1)
    assert len(capped) == 5  # 4 specials + int(2 * 0.5) words


def test_tagging_collate_pads_labels_with_ignore():
    v = Vocab.from_corpus(SPLIT["text"], 100, 0.0, 0)
    batch = list(zip(SPLIT["text"], SPLIT["verb_index"], SPLIT["srl_frames"]))
    ids, verbs, labels, lengths = make_tagging_collate(v)(batch)
    assert ids.shape == labels.shape == (2, 8)
    assert lengths.tolist() == [8, 4]
    assert (labels[1, 4:] == IGNORE).all() and labels[1, 3] == LABEL2ID["B-ARGM-TMP"]


def test_qa_pairs_format():
    q, a, meta = create_qa_pairs(SPLIT)
    assert len(q) == 10 and meta[0] == (0, "ARG0")
    assert q[0][:2] == ["accept", "[SEP]"] and q[0][-1] == "ARG0"
    assert a[0] == [BOS, "He", EOS]
    assert a[1] == [BOS, "anything", "of", "value", EOS]
    assert a[2] == [BOS, EOS]  # ARG2 absent


def test_answer_scores_and_tables():
    assert answer_scores([], []) == (1.0, 1.0)
    assert answer_scores(["a"], []) == (0.0, 0.0)
    em, f1 = answer_scores(["a", "b"], ["a"])
    assert em == 0.0 and abs(f1 - 2 / 3) < 1e-9
    t = role_answer_table([("ARG0", ["x"], ["x"]), ("ARG0", ["y"], [])])
    assert t["ARG0"]["exact_match"] == 0.5 and t["ARG0"]["token_f1_when_present"] == 0.5
    assert span_f1([["B-ARG0", "O"]], [["B-ARG0", "O"]])["micro_f1"] == 1.0


def test_padding_does_not_change_tagger_outputs():
    torch.manual_seed(0)
    for kwargs in ({}, {"combine": "add"}, {"bidirectional": True, "verb_flag_dim": 4}, {"cell": "gru"}):
        m = RNNTagger(20, 11, 8, 8, **kwargs).eval()
        x = torch.tensor([[5, 6, 7, 0, 0]])
        alone = m(x[:, :3], torch.tensor([1]), torch.tensor([3]))
        padded = m(x, torch.tensor([1]), torch.tensor([3]))[:, :3]
        assert torch.allclose(alone, padded, atol=1e-6), kwargs
        assert m(x, torch.tensor([1]), torch.tensor([3])).shape == (1, 5, 11)


def test_seq2seq_shapes_attention_mask_and_beam1_equals_greedy():
    torch.manual_seed(0)
    emb = torch.randn(30, 6)
    m = Seq2Seq(emb, 30, hidden_dim=8).eval()
    src = torch.tensor([[5, 6, 7, 8], [9, 10, 0, 0]])
    src_len = torch.tensor([4, 2])
    trg = torch.tensor([[1, 11, 12, 2], [1, 13, 2, 0]])
    assert m(src, src_len, trg, 1.0).shape == (2, 3, 30)
    enc, h, c = m.encoder(src, src_len)
    w = m.decoder.attention(h[-1], enc, src != 0)
    assert torch.allclose(w[1, 2:], torch.zeros(2)) and torch.allclose(w.sum(-1), torch.ones(2))
    one = src[1:, :2]
    greedy = m.greedy(one, torch.tensor([2]), 1, 2, max_len=6)[0]
    if 2 in greedy:
        greedy = greedy[: greedy.index(2)]
    assert m.beam_search(one, torch.tensor([2]), 1, 2, beam_size=1, max_len=6) == greedy
    assert isinstance(m.beam_search(one, torch.tensor([2]), 1, 2, beam_size=16, max_len=6), list)
