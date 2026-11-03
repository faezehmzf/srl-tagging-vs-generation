# Comparison on the validation set (800 sentences)

## Overview

| system | span F1 | role exact match | token F1 (role present) | parameters | training time |
|---|---|---|---|---|---|
| always answer nothing | — | 0.671 | 0.000 | — | — |
| LSTM (baseline) | 0.168 | 0.676 | 0.224 | 0.8M | 0.2 min |
| GRU (baseline) | 0.173 | 0.680 | 0.223 | 0.8M | 0.5 min |
| BiLSTM + predicate flag | 0.529 | 0.785 | 0.643 | 0.8M | 1.0 min |
| BiLSTM + GloVe | 0.684 | 0.853 | 0.760 | 4.4M | 2.9 min |
| BiLSTM-CRF + GloVe | 0.738 | 0.872 | 0.764 | 4.4M | 5.0 min |
| RoBERTa-base | 0.868 | 0.933 | 0.905 | 124.7M | 18.6 min |
| RoBERTa-base + CRF | 0.880 | 0.935 | 0.902 | 124.7M | 18.4 min |
| RNN encoder-decoder + attention (beam 16) | — | 0.703 | 0.243 | 6.3M | 24.0 min |
| FLAN-T5-base (beam 4) | — | 0.926 | 0.894 | 247.6M | 100.5 min |

## Token F1 per role (only (sentence, role) pairs where the role is present)

| system | ARG0 (n=357) | ARG1 (n=662) | ARG2 (n=219) | ARGM-TMP (n=68) | ARGM-LOC (n=10) |
|---|---|---|---|---|---|
| LSTM (baseline) | 0.249 | 0.196 | 0.276 | 0.235 | 0.000 |
| GRU (baseline) | 0.285 | 0.181 | 0.268 | 0.191 | 0.000 |
| BiLSTM + predicate flag | 0.604 | 0.727 | 0.523 | 0.515 | 0.000 |
| BiLSTM + GloVe | 0.749 | 0.812 | 0.672 | 0.676 | 0.200 |
| BiLSTM-CRF + GloVe | 0.736 | 0.808 | 0.692 | 0.765 | 0.500 |
| RoBERTa-base | 0.908 | 0.933 | 0.821 | 0.926 | 0.600 |
| RoBERTa-base + CRF | 0.910 | 0.929 | 0.802 | 0.956 | 0.600 |
| RNN encoder-decoder + attention (beam 16) | 0.338 | 0.253 | 0.085 | 0.191 | 0.000 |
| FLAN-T5-base (beam 4) | 0.900 | 0.921 | 0.786 | 0.971 | 0.700 |

## Evaluation pitfall: counting padding

The same LSTM baseline evaluated three ways (padding scored as `O` during training and evaluation):

- token accuracy including padding: **0.917**
- token accuracy on real tokens: 0.844 (always predicting `O` gives 0.842)
- span F1: **0.136**

## Diagnostics

Taggers: ill-formed spans (`I-X` not preceded by `B-X`/`I-X`; gold has 0).

| tagger | invalid `I-` tags | sentences affected |
|---|---|---|
| LSTM (baseline) | 439 | 267 |
| GRU (baseline) | 397 | 271 |
| BiLSTM + predicate flag | 190 | 151 |
| BiLSTM + GloVe | 64 | 54 |
| BiLSTM-CRF + GloVe | 0 | 0 |
| RoBERTa-base | 24 | 23 |
| RoBERTa-base + CRF | 0 | 0 |

Generators: is the answer copied from the sentence?

| generator | answer tokens not in the sentence | non-empty answers that are a contiguous span of the sentence |
|---|---|---|
| RNN encoder-decoder + attention (beam 16) | 513/1375 (37.3%) | 440/911 (48.3%) |
| FLAN-T5-base (beam 4) | 1/3293 (0.0%) | 1349/1358 (99.3%) |

## Examples (first short validation sentences with at least two roles)

The most important thing about Disney is that it **[is]** a global brand .

| role | gold | LSTM (baseline) | BiLSTM-CRF + GloVe | RoBERTa-base | RNN encoder-decoder + attention (beam 16) | FLAN-T5-base (beam 4) |
|---|---|---|---|---|---|---|
| ARG1 | it | about Disney | it | it | it | it |
| ARG2 | a global brand | global brand | a global brand | a global brand | a recession | a global brand |

Are they going to be different from what they **[are]** now ?

| role | gold | LSTM (baseline) | BiLSTM-CRF + GloVe | RoBERTa-base | RNN encoder-decoder + attention (beam 16) | FLAN-T5-base (beam 4) |
|---|---|---|---|---|---|---|
| ARG1 | they | they | they | they | they | they |
| ARG2 | what | from what | what now | what | how | what |
| ARGM-TMP | now | ∅ | ∅ | now | ∅ | now |

Then , , on the bridge is Hong Kong and outside **[is]** Shenzhen .

| role | gold | LSTM (baseline) | BiLSTM-CRF + GloVe | RoBERTa-base | RNN encoder-decoder + attention (beam 16) | FLAN-T5-base (beam 4) |
|---|---|---|---|---|---|---|
| ARG1 | Shenzhen | the | outside | Shenzhen | The wheat | Shenzhen |
| ARG2 | outside | ∅ | Shenzhen | outside | in | Hong Kong |
| ARGM-TMP | ∅ | ∅ | ∅ | ∅ | ∅ | Then |

And we 're going to get **[started]** here /.

| role | gold | LSTM (baseline) | BiLSTM-CRF + GloVe | RoBERTa-base | RNN encoder-decoder + attention (beam 16) | FLAN-T5-base (beam 4) |
|---|---|---|---|---|---|---|
| ARG0 | we | we | we | ∅ | we | ∅ |
| ARG1 | ∅ | ∅ | ∅ | we | we | we |
| ARGM-LOC | here | ∅ | here | here | ∅ | here |

