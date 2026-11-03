"""Data loading, vocabulary and batching for SRL (tagging and QA formats)."""
import json
from collections import Counter

import torch
from torch.utils.data import Dataset

# BIO tags for five PropBank roles. No I-ARGM-* tag occurs in this dataset
# (location and time arguments are single tokens here).
LABEL2ID = {
    "O": 0, "B-ARG0": 1, "I-ARG0": 2, "B-ARG1": 3, "I-ARG1": 4, "B-ARG2": 5,
    "I-ARG2": 6, "B-ARGM-LOC": 7, "I-ARGM-LOC": 8, "B-ARGM-TMP": 9, "I-ARGM-TMP": 10,
}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
ROLES = ["ARG0", "ARG1", "ARG2", "ARGM-TMP", "ARGM-LOC"]
IGNORE = -100  # label used for padding positions; ignored by the loss

PAD, START, END, UNK = "Pad", "Start", "End", "Unknown"
SEP, BOS, EOS = "[SEP]", "<s>", "</s>"


def load_split(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def role_of(tag):
    return None if tag == "O" else tag[2:]


class Vocab:
    """Word <-> id mapping with special tokens Pad, Start, End, Unknown."""

    def __init__(self, word2id=None):
        if word2id is None:
            self.word2id = {PAD: 0, START: 1, END: 2, UNK: 3}
        else:
            self.word2id = dict(word2id)
        self.id2word = {i: w for w, i in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.word2id[UNK])

    def __len__(self):
        return len(self.word2id)

    def add(self, word):
        if word not in self.word2id:
            idx = len(self.word2id)
            self.word2id[word] = idx
            self.id2word[idx] = word
        return self.word2id[word]

    def words2indices(self, sents):
        return [[self[w] for w in s] for s in sents]

    def indices2words(self, word_ids):
        return [[self.id2word[i] for i in ids] for ids in word_ids]

    def to_input_tensor(self, sents):
        """Right-pad a batch of token lists with Pad and return (ids, lengths)."""
        lengths = [len(s) for s in sents]
        max_len = max(lengths)
        ids = [self.words2indices([s])[0] + [self.word2id[PAD]] * (max_len - len(s)) for s in sents]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    @staticmethod
    def from_corpus(corpus, size, remove_frac, freq_cutoff):
        """Keep words seen more than `freq_cutoff` times, most frequent first,
        capped at int(size * (1 - remove_frac)) words."""
        freq = Counter(w for sent in corpus for w in sent)
        kept = [w for w, c in sorted(freq.items(), key=lambda x: x[1], reverse=True) if c > freq_cutoff]
        vocab = Vocab()
        for w in kept[: int(size * (1 - remove_frac))]:
            vocab.add(w)
        return vocab


# ---------------------------------------------------------------- tagging format
class TaggingDataset(Dataset):
    def __init__(self, split):
        self.texts = split["text"]
        self.verbs = split["verb_index"]
        self.tags = split.get("srl_frames")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.verbs[i], (self.tags[i] if self.tags else None)


def make_tagging_collate(vocab):
    def collate(batch):
        texts, verbs, tags = zip(*batch)
        ids, lengths = vocab.to_input_tensor(list(texts))
        labels = torch.full(ids.shape, IGNORE, dtype=torch.long)
        if tags[0] is not None:
            for i, t in enumerate(tags):
                labels[i, : len(t)] = torch.tensor([LABEL2ID[x] for x in t])
        return ids, torch.tensor(verbs, dtype=torch.long), labels, lengths
    return collate


# ---------------------------------------------------------------- QA format
def create_qa_pairs(split):
    """Turn every (sentence, verb) into 5 question/answer pairs, one per role:
    question = verb [SEP] sentence ROLE ; answer = <s> tokens-with-ROLE </s>."""
    questions, answers, meta = [], [], []
    for idx, (sent, verb, tags) in enumerate(zip(split["text"], split["verb_index"], split["srl_frames"])):
        roles = [role_of(t) for t in tags]
        for role in ROLES:
            questions.append([sent[verb], SEP] + list(sent) + [role])
            answers.append([BOS] + [w for w, r in zip(sent, roles) if r == role] + [EOS])
            meta.append((idx, role))
    return questions, answers, meta


class QADataset(Dataset):
    def __init__(self, questions, answers):
        self.q, self.a = questions, answers

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        return self.q[i], self.a[i]


def make_qa_collate(vocab):
    def collate(batch):
        qs, ans = zip(*batch)
        src, src_len = vocab.to_input_tensor(list(qs))
        trg, _ = vocab.to_input_tensor(list(ans))
        return src, src_len, trg
    return collate


def load_glove_matrix(path, vocab, dim=300, seed=0):
    """GloVe vectors for the vocabulary. GloVe is lower-cased, so words are
    looked up in lower case.
    Words without a vector get small random vectors; Pad is all zeros."""
    g = torch.Generator().manual_seed(seed)
    matrix = torch.randn(len(vocab), dim, generator=g) * 0.1
    wanted = {}
    for w, i in vocab.word2id.items():
        wanted.setdefault(w.lower(), []).append(i)
    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, rest = line.split(" ", 1)
            if word in wanted:
                vec = torch.tensor([float(x) for x in rest.split()])
                for i in wanted[word]:
                    matrix[i] = vec
                    found += 1
    matrix[vocab.word2id[PAD]] = 0.0
    return matrix, found
