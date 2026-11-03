# Data

Nothing in this folder is committed.

## SRL data (`train.json`, `valid.json`, `test.json`)

English sentences annotated with PropBank-style semantic roles for one predicate each.
The source and licence of the copy used here are not documented, so it is not redistributed;
any data in the same format can be used. Each file is one JSON object of parallel lists:

```json
{
  "text":         [["He", "would", "n't", "accept", "anything", "of", "value", "."], ...],
  "verb_index":   [3, ...],
  "srl_frames":   [["B-ARG0", "O", "O", "O", "B-ARG1", "I-ARG1", "I-ARG1", "O"], ...],
  "words_indices": [[...], ...]
}
```

`srl_frames` is absent from `test.json`, so all results are reported on `valid.json`.

| split | sentences | tokens | arguments |
|---|---|---|---|
| train | 13,310 | 298,914 | ARG1 10,695 · ARG0 6,851 · ARG2 3,312 · ARGM-TMP 1,202 · ARGM-LOC 131 |
| valid | 800 | 19,775 | ARG1 662 · ARG0 357 · ARG2 219 · ARGM-TMP 69 · ARGM-LOC 10 |
| test | 800 | — | unlabelled |

84% of validation tokens are `O`. Location and time arguments are single tokens (no `I-ARGM-*` tags).

## GloVe (`glove/glove.6B.300d.txt`)

[GloVe 6B](https://nlp.stanford.edu/projects/glove/) (Pennington et al., 2014), Public Domain Dedication and
License. Download `glove.6B.zip` and extract `glove.6B.300d.txt` here.

## Pre-trained transformers

`roberta-base` and `google/flan-t5-base` are downloaded automatically from the Hugging Face hub.
