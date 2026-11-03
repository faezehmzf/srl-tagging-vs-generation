"""Linear-chain conditional random field with BIO constraints.

score(x, y) = Σ_t emission[t, y_t] + Σ_t transition[y_{t-1}, y_t] + start[y_1] + end[y_n]
The loss is log Z(x) - score(x, y_gold), with Z computed by the forward
algorithm; decoding uses Viterbi. Transitions that break the BIO scheme
(e.g. O -> I-ARG1, B-ARG0 -> I-ARG1, start -> I-X) get a large negative score,
so only well-formed spans can be predicted.
"""
import torch
import torch.nn as nn

NEG = -1e4


def bio_constraints(id2label):
    n = len(id2label)
    allowed = torch.ones(n, n, dtype=torch.bool)
    start_ok = torch.ones(n, dtype=torch.bool)
    for j in range(n):
        lj = id2label[j]
        if lj.startswith("I-"):
            start_ok[j] = False
            for i in range(n):
                li = id2label[i]
                allowed[i, j] = li[2:] == lj[2:] and li[:2] in ("B-", "I-")
    return allowed, start_ok


class CRF(nn.Module):
    def __init__(self, num_tags, allowed=None, start_ok=None):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        self.register_buffer("trans_mask", torch.zeros(num_tags, num_tags) if allowed is None
                             else (~allowed).float() * NEG)
        self.register_buffer("start_mask", torch.zeros(num_tags) if start_ok is None else (~start_ok).float() * NEG)

    def _trans(self):
        return self.trans + self.trans_mask

    def _start(self):
        return self.start + self.start_mask

    def log_partition(self, em, mask):
        """em: [B, L, T]; mask: [B, L] bool, True for real tokens (a prefix of each row)."""
        alpha = self._start() + em[:, 0]
        trans = self._trans()
        for t in range(1, em.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + em[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def gold_score(self, em, tags, mask):
        B = em.size(0)
        rows = torch.arange(B, device=em.device)
        trans = self._trans()
        score = self._start()[tags[:, 0]] + em[rows, 0, tags[:, 0]]
        for t in range(1, em.size(1)):
            step = trans[tags[:, t - 1], tags[:, t]] + em[rows, t, tags[:, t]]
            score = score + step * mask[:, t]
        last = tags[rows, mask.sum(1) - 1]
        return score + self.end[last]

    def nll(self, em, tags, mask):
        """Mean negative log-likelihood; tags at masked positions are ignored."""
        tags = tags.masked_fill(~mask, 0)
        return (self.log_partition(em, mask) - self.gold_score(em, tags, mask)).mean()

    @torch.no_grad()
    def decode(self, em, mask):
        """Viterbi; returns a list of tag-id lists (one per sequence, real tokens only)."""
        trans = self._trans()
        score = self._start() + em[:, 0]
        back = []
        for t in range(1, em.size(1)):
            best, idx = (score.unsqueeze(2) + trans.unsqueeze(0)).max(dim=1)
            nxt = best + em[:, t]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, nxt, score)
            back.append(torch.where(m, idx, torch.arange(em.size(2), device=em.device).expand_as(idx)))
        score = score + self.end
        out = []
        lengths = mask.sum(1).tolist()
        best_last = score.argmax(1).tolist()
        back = [b.tolist() for b in back]
        for b, (n, tag) in enumerate(zip(lengths, best_last)):
            path = [tag]
            for t in range(n - 1, 0, -1):
                tag = back[t - 1][b][tag]
                path.append(tag)
            out.append(path[::-1])
        return out
