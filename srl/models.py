"""Models: RNN tagger (LSTM/GRU) and BiLSTM encoder + attention LSTM decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNTagger(nn.Module):
    """Predicate-aware sequence tagger.

    Embedding (random or GloVe) [+ predicate-indicator embedding] -> LSTM/GRU
    (uni- or bidirectional, 1+ layers) -> [token state ; predicate state] -> Linear
    -> softmax or CRF.

    The default configuration (one unidirectional layer, 64 units, no indicator,
    no CRF) is the plain baseline; `combine='add'` sums the two states instead of
    concatenating them.
    """

    def __init__(self, vocab_size, num_labels, emb_dim=64, hidden_dim=64, cell="lstm", combine="concat",
                 bidirectional=False, verb_flag_dim=0, num_layers=1, dropout=0.0, embeddings=None,
                 crf=False, id2label=None):
        super().__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
            emb_dim = embeddings.size(1)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.verb_flag = nn.Embedding(2, verb_flag_dim) if verb_flag_dim else None
        self.bidirectional = bidirectional
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(emb_dim + verb_flag_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.combine = combine
        state_dim = hidden_dim * (2 if bidirectional else 1)
        self.out = nn.Linear(2 * state_dim if combine == "concat" else state_dim, num_labels)
        self.crf = None
        if crf:
            from .crf import CRF, bio_constraints

            self.crf = CRF(num_labels, *bio_constraints(id2label))

    def forward(self, x, verb_idx, lengths=None):
        rows = torch.arange(x.size(0), device=x.device)
        emb = self.dropout(self.embedding(x))
        if self.verb_flag is not None:
            flag = torch.zeros_like(x)
            flag[rows, verb_idx] = 1
            emb = torch.cat([emb, self.verb_flag(flag)], dim=-1)
        if self.bidirectional:  # pack so the backward pass does not start on padding
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.rnn(packed)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        else:
            h, _ = self.rnn(emb)  # right padding: real tokens never see pads
        verb = h[rows, verb_idx].unsqueeze(1).expand_as(h)
        z = torch.cat([h, verb], dim=-1) if self.combine == "concat" else h + verb
        return self.out(self.dropout(z))


class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_dim=256, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze, padding_idx=0)
        self.lstm = nn.LSTM(embeddings.size(1), hidden_dim, batch_first=True, bidirectional=True)
        self.bridge_h = nn.Linear(2 * hidden_dim, hidden_dim)
        self.bridge_c = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, src, src_len):
        packed = pack_padded_sequence(self.embedding(src), src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        h = torch.tanh(self.bridge_h(torch.cat([h[0], h[1]], dim=-1))).unsqueeze(0)
        c = self.bridge_c(torch.cat([c[0], c[1]], dim=-1)).unsqueeze(0)
        return out, h, c


class Attention(nn.Module):
    """Additive attention, score = v^T tanh(W [dec_state ; enc_state])."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(3 * hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_h, enc_out, src_mask):
        h = dec_h.unsqueeze(1).expand(-1, enc_out.size(1), -1)
        scores = self.v(torch.tanh(self.attn(torch.cat([h, enc_out], dim=-1)))).squeeze(-1)
        scores = scores.masked_fill(~src_mask, float("-inf"))  # never attend to padding
        return F.softmax(scores, dim=-1)


class Decoder(nn.Module):
    def __init__(self, embeddings, vocab_size, hidden_dim=256, dropout=0.1, freeze=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embeddings.size(1) + 2 * hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, h, c, enc_out, src_mask):
        emb = self.dropout(self.embedding(token)).unsqueeze(1)
        weights = self.attention(h[-1], enc_out, src_mask)
        context = torch.bmm(weights.unsqueeze(1), enc_out)
        out, (h, c) = self.lstm(torch.cat([emb, context], dim=-1), (h, c))
        return self.out(out.squeeze(1)), h, c, weights


class Seq2Seq(nn.Module):
    def __init__(self, embeddings, vocab_size, hidden_dim=256, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Encoder(embeddings, hidden_dim)
        self.decoder = Decoder(embeddings, vocab_size, hidden_dim)

    def forward(self, src, src_len, trg, teacher_forcing=0.5):
        """Returns logits for trg[:, 1:] (the token after <s> onwards).
        The decoder reads trg[:, t-1] and predicts trg[:, t]."""
        enc_out, h, c = self.encoder(src, src_len)
        mask = src != self.pad_id
        token = trg[:, 0]
        logits = []
        for t in range(1, trg.size(1)):
            step, h, c, _ = self.decoder(token, h, c, enc_out, mask)
            logits.append(step)
            use_gold = torch.rand(1).item() < teacher_forcing
            token = trg[:, t] if use_gold else step.argmax(-1)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy(self, src, src_len, bos_id, eos_id, max_len):
        enc_out, h, c = self.encoder(src, src_len)
        mask = src != self.pad_id
        token = torch.full((src.size(0),), bos_id, dtype=torch.long, device=src.device)
        out = []
        for _ in range(max_len):
            step, h, c, _ = self.decoder(token, h, c, enc_out, mask)
            token = step.argmax(-1)
            out.append(token)
        return torch.stack(out, dim=1).tolist()

    @torch.no_grad()
    def beam_search(self, src, src_len, bos_id, eos_id, beam_size=16, max_len=60, length_penalty=1.0):
        """Beam search for one example (src: [1, S]). Hypotheses that emit </s>
        are scored by log-prob / len**length_penalty; the best one is returned."""
        enc_out, h, c = self.encoder(src, src_len)
        mask = src != self.pad_id
        tokens = torch.tensor([bos_id], device=src.device)
        scores = torch.zeros(1, device=src.device)
        seqs, finished = [[]], []
        for _ in range(max_len):
            k = tokens.size(0)
            step, h, c, _ = self.decoder(tokens, h, c, enc_out.expand(k, -1, -1), mask.expand(k, -1))
            cand = (scores.unsqueeze(1) + F.log_softmax(step, dim=-1)).view(-1)
            top, idx = cand.topk(min(beam_size, cand.numel()))
            vocab_size = step.size(1)
            keep, new_tokens, new_scores, new_seqs = [], [], [], []
            for s, i in zip(top.tolist(), idx.tolist()):
                b, t = divmod(i, vocab_size)
                if t == eos_id:
                    finished.append((s / (len(seqs[b]) + 1) ** length_penalty, seqs[b]))
                else:
                    keep.append(b); new_tokens.append(t); new_scores.append(s); new_seqs.append(seqs[b] + [t])
            if len(finished) >= beam_size or not keep:
                break
            sel = torch.tensor(keep, device=src.device)
            h, c = h[:, sel].contiguous(), c[:, sel].contiguous()
            tokens = torch.tensor(new_tokens, device=src.device)
            scores = torch.tensor(new_scores, device=src.device)
            seqs = new_seqs
        if not finished:
            finished = [(s / (len(q) + 1) ** length_penalty, q) for s, q in zip(scores.tolist(), seqs)]
        return max(finished, key=lambda x: x[0])[1]
