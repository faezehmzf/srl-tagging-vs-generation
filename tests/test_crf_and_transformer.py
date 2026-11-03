import itertools
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from srl.crf import CRF, bio_constraints  # noqa: E402
from srl.data import ID2LABEL, LABEL2ID  # noqa: E402
from srl.models import RNNTagger  # noqa: E402


def _brute_force(crf, em, n):
    scores = {}
    for path in itertools.product(range(em.size(-1)), repeat=n):
        s = crf._start()[path[0]] + em[0, path[0]] + crf.end[path[-1]]
        s = s + sum(crf._trans()[path[i - 1], path[i]] + em[i, path[i]] for i in range(1, n))
        scores[path] = s.item()
    return scores


def test_crf_partition_and_viterbi_match_brute_force():
    torch.manual_seed(0)
    crf = CRF(4)
    for p in (crf.trans, crf.start, crf.end):
        p.data.normal_()
    em = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    logz, paths = crf.log_partition(em, mask), crf.decode(em, mask)
    for b in range(2):
        n = int(mask[b].sum())
        scores = _brute_force(crf, em[b], n)
        assert math.isclose(logz[b].item(), math.log(sum(math.exp(v) for v in scores.values())), rel_tol=1e-5)
        assert tuple(paths[b]) == max(scores, key=scores.get)


def test_crf_nll_is_positive_and_decreases_with_training():
    torch.manual_seed(0)
    crf = CRF(3)
    em = torch.randn(4, 5, 3, requires_grad=True)
    tags = torch.randint(0, 3, (4, 5))
    mask = torch.ones(4, 5, dtype=torch.bool)
    opt = torch.optim.SGD([em, *crf.parameters()], lr=0.5)
    first = crf.nll(em, tags, mask)
    assert first.item() > 0
    for _ in range(50):
        opt.zero_grad(); loss = crf.nll(em, tags, mask); loss.backward(); opt.step()
    assert loss.item() < first.item()
    assert crf.decode(em, mask) == tags.tolist()


def test_bio_constraints_forbid_invalid_spans():
    allowed, start_ok = bio_constraints(ID2LABEL)
    L = LABEL2ID
    assert not allowed[L["O"], L["I-ARG1"]] and not allowed[L["B-ARG0"], L["I-ARG1"]]
    assert allowed[L["B-ARG1"], L["I-ARG1"]] and allowed[L["I-ARG1"], L["I-ARG1"]] and allowed[L["I-ARG1"], L["O"]]
    assert not start_ok[L["I-ARG0"]] and start_ok[L["B-ARG0"]]
    crf = CRF(len(L), allowed, start_ok)
    em = torch.zeros(1, 2, len(L))
    em[0, 0, L["O"]] = 5.0
    em[0, 1, L["I-ARG1"]] = 5.0  # emissions prefer the invalid "O I-ARG1"
    assert crf.decode(em, torch.ones(1, 2, dtype=torch.bool))[0][1] != L["I-ARG1"]


def test_rnn_tagger_with_crf_and_pretrained_embeddings():
    emb = torch.randn(30, 12)
    m = RNNTagger(30, len(LABEL2ID), bidirectional=True, verb_flag_dim=4, num_layers=2, dropout=0.1,
                  embeddings=emb, crf=True, id2label=ID2LABEL).eval()
    x = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0]])
    out = m(x, torch.tensor([1, 0]), torch.tensor([3, 2]))
    assert out.shape == (2, 4, len(LABEL2ID)) and m.crf is not None
    assert torch.allclose(m.embedding.weight[5], emb[5])  # initialised from the given vectors


def test_transformer_encoding_maps_words_to_first_subtoken():
    from train_transformer_tagger import collate, encode

    class FakeTok:  # splits "steles" into two pieces, everything else one piece
        pad_token_id = 1

        def __call__(self, words, is_split_into_words, truncation, max_length):
            ids, wid = [0], [None]
            for i, w in enumerate(words):
                for _ in range(2 if w == "steles" else 1):
                    ids.append(100 + i); wid.append(i)
            ids.append(2); wid.append(None)

            class Enc(dict):
                def word_ids(self_inner):
                    return wid
            return Enc(input_ids=ids)

    split = {"text": [["two", "steles", "stand"]], "verb_index": [2], "srl_frames": [["B-ARG1", "I-ARG1", "O"]]}
    item = encode(split, FakeTok())[0]
    assert item["word_index"] == [1, 2, 4]
    assert item["pred_flag"] == [0, 0, 0, 0, 1, 0]
    b = collate([item], pad_id=1)
    assert b["labels"].tolist() == [[LABEL2ID["B-ARG1"], LABEL2ID["I-ARG1"], LABEL2ID["O"]]]
