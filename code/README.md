# FREUID pipeline — run instructions

This directory contains the training and inference code for the
**FREUID Challenge 2026** binary fraud-detection task. See
[REPORT.md](REPORT.md) for the technical report required by the
competition, and the root [README.md](../README.md) for the problem
statement.

## Layout

```
code/
├── README.md          # this file
├── REPORT.md          # competition-required technical report
├── requirements.txt   # pinned dependencies
├── configs/
│   └── baseline.yaml  # hyperparameters
├── scripts/
│   └── download_data.py  # Kaggle download or synthetic generator
├── src/
│   ├── data.py        # CSV loader, stratified split, FREUIDDataset
│   ├── data_selftest.py  # one-shot end-to-end test of data module
│   ├── metric.py      # FREUID Score, AuDET, APCER@1%BPCER
│   ├── model.py       # timm backbone + single-logit head
│   ├── train.py       # training entry point
│   └── infer.py       # TTA inference + submission.csv writer
└── evaluation/main.py # placeholder (per repo skeleton)
```

## Quick start

```bash
# 1. Install dependencies.
pip install -r code/requirements.txt

# 2. Get the data. Either from Kaggle (real competition data):
python code/scripts/download_data.py --competition freuid-challenge-2026
# Or, for a pipeline-only smoke test (random-noise images):
python code/scripts/download_data.py --synthetic

# 3. Train. Best checkpoint lands at runs/baseline/best.pt.
python -m code.src.train --config code/configs/baseline.yaml

# 4. Produce submission.csv.
python -m code.src.infer \
    --ckpt runs/baseline/best.pt \
    --sample-submission dataset/sample_submission.csv \
    --out submission.csv

# Submit submission.csv through the Kaggle competition page.
```

## Sanity checks

```bash
# Metric module is correct on synthetic scores:
PYTHONPATH=. python -m code.src.metric

# Data module reads CSVs, builds batches, stratified split is sane:
PYTHONPATH=. python -m code.src.data_selftest
```

## Tweakable knobs

All defaults live in `configs/baseline.yaml`. Common edits:

| Knob | What it does | Trade-off |
|---|---|---|
| `data.image_size` | Resize images to N×N | Larger → slower but more accurate |
| `data.batch_size` | Train batch size | Larger → faster on GPU, more memory |
| `model.backbone` | Any `timm` model name | Larger → stronger features, more compute |
| `optim.lr` | AdamW learning rate | Tune per backbone |
| `optim.epochs` | Number of epochs | More → better fit, overfit risk |
| `optim.amp` | bfloat16 autocast | No-op on CPU; speeds up modern GPUs |

Ensemble multiple checkpoints at inference time:

```bash
python -m code.src.infer \
    --ckpt runs/fold0/best.pt \
    --ckpt runs/fold1/best.pt \
    --ckpt runs/fold2/best.pt \
    --sample-submission dataset/sample_submission.csv \
    --out submission.csv
```

## What does NOT live in this repo

- The competition data (`dataset/train/`, `dataset/test/`, CSVs) —
  excluded via `.gitignore`. Re-download with
  `code/scripts/download_data.py`.
- Trained checkpoints (`runs/`, `*.pt`) — gitignored.
- Kaggle credentials (`~/.kaggle/kaggle.json`) — never stored here.
