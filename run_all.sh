#!/usr/bin/env bash
# Reproduce every result in results/. Run from the repository root.
# Needs the data in data/ (see data/README.md); transformer models download from the Hugging Face hub.
set -euo pipefail
PY=${PY:-python}
G=data/glove/glove.6B.300d.txt

# RNN taggers
$PY scripts/train_tagger.py --cell lstm --out results/lstm_baseline
$PY scripts/train_tagger.py --cell gru  --out results/gru_baseline
$PY scripts/train_tagger.py --cell lstm --combine add --count-padding --out results/pitfall_padding
$PY scripts/train_tagger.py --bidirectional --verb-flag-dim 16 --out results/bilstm_flag
$PY scripts/train_tagger.py --bidirectional --verb-flag-dim 16 --glove $G --hidden-dim 128 --num-layers 2 \
    --dropout 0.3 --epochs 15 --out results/bilstm_glove
$PY scripts/train_tagger.py --bidirectional --verb-flag-dim 16 --glove $G --hidden-dim 128 --num-layers 2 \
    --dropout 0.3 --epochs 15 --crf --out results/bilstm_glove_crf

# RNN encoder-decoder (SRL as question answering)
$PY scripts/train_seq2seq.py --glove $G --out results/rnn_seq2seq

# Pre-trained transformers
$PY scripts/train_transformer_tagger.py --out results/roberta_base
$PY scripts/train_transformer_tagger.py --crf --out results/roberta_base_crf
$PY scripts/train_t5_qa.py --out results/flan_t5_base_qa

$PY scripts/compare.py
$PY scripts/plot_comparison.py
$PY -m pytest -q tests
