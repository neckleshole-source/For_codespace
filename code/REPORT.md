# Technical Report — FREUID Challenge 2026

This is the short technical report required by the competition rules
([README.md](../README.md), "Code Requirements" section). It describes
the method, training data, model architecture, inference procedure and
external resources used by the submission in this repository.

## Method

A single binary classifier trained end-to-end with supervised
cross-entropy. The model outputs a fraud score in `[0, 1]` after a
sigmoid; the **FREUID Score** (harmonic mean of `1 - AuDET` and
`1 - APCER@1%BPCER`, lower is better) is used as the validation
metric for model selection.

Key design choices driven by the competition's threat model
(physical manipulations, GenAI edits, print-and-capture forgeries):

- **Cross-document generalization over per-template tuning.** We
  stratify validation by `(label, document_type)` so the local
  validation mix mirrors the private test set's cross-domain
  distribution. This favours models that learn document-agnostic
  fraud cues over template-specific artefacts.
- **Heavy but careful augmentation.** Mild geometric, mild colour,
  JPEG compression, light blur and small affine warps. The training
  set already contains physical and GenAI artefacts, so augmentation
  is intentionally not extreme — strong warps or aggressive noise
  would destroy the very cues the model needs to learn.
- **APCER@1%BPCER-conscious validation.** This is one of the two
  components of the FREUID Score. We track it directly via the
  harmonic-mean metric rather than relying on AUC alone, because a
  model with great AUC but poor strict-operating-point behaviour is
  penalised by the leaderboard formula.

## Training data

- **Primary**: the FREUID dataset provided by the competition (under
  `dataset/` after running `python code/scripts/download_data.py`).
- **External**: none used in this baseline. External data is allowed
  by the rules; any future use must be disclosed here and justified.
- **Split**: 80/20 train/validation, stratified on `(label, type)`.
  Five-fold CV indices are exposed via
  `code.src.data.stratified_kfold_indices` for hyperparameter
  sweeps.

## Model architecture

- **Backbone**: `timm`-implemented image classifier (default:
  `convnext_tiny.fb_in1k`). The backbone is initialised from ImageNet
  pretrained weights; the classification head is replaced by a
  single-logit linear layer.
- **Head**: `Linear(feat_dim, 1)` with a small Gaussian init on the
  weights and a zero bias so the model starts at the prior.
- **Dropout**: `drop_rate=0.1` on the backbone (timm-passed).
- **Output**: a single logit; `sigmoid(logit)` is the fraud score.

### Why ConvNeXt-tiny

For a CPU-friendly baseline, ConvNeXt-tiny offers a strong
accuracy/compute trade-off. On a GPU box the same code path accepts
larger backbones (e.g. `convnext_base`, `efficientnetv2_s`,
`swin_tiny`) by changing one config line. The choice is purely a
function of the available compute budget; the rest of the pipeline
is backbone-agnostic.

## Inference procedure

1. Load the best checkpoint by validation FREUID Score
   (`runs/<run>/best.pt`).
2. For each test image:
   - Apply the eval transform (resize, pad, centre-crop,
     ImageNet-normalise).
   - Run a forward pass.
   - Run a horizontal-flip forward pass.
   - Average the two logits before the sigmoid (TTA).
3. Write `submission.csv` with two columns `id,label` — where
   `label` is the fraud score, *not* a class — in the exact row order
   of `dataset/sample_submission.csv`. Missing ids are filled with
   `0.5`; extra predictions are appended. All scores are clipped to
   `[0, 1]`.

Ensemble is supported by passing multiple `--ckpt` paths to
`code.src.infer`; the per-checkpoint predictions are averaged.

## Training recipe

| Hyperparameter | Value (baseline) |
|---|---|
| Image size | 160 (CPU) / 224 (GPU recommended) |
| Optimiser | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.05 |
| Schedule | Cosine with 1-epoch warmup |
| Epochs | 8 |
| Batch size | 32 |
| Loss | `BCEWithLogitsLoss(pos_weight=neg/pos)` |
| AMP | bfloat16 autocast (no-op on CPU) |
| EMA decay | 0.999 |
| Gradient clip | 1.0 |
| Validation metric | FREUID Score (lower is better) |

## Environment / dependencies

Pinned in `code/requirements.txt`. Reproduce with:

```bash
pip install -r code/requirements.txt
python code/scripts/download_data.py        # Kaggle download
python -m code.src.train --config code/configs/baseline.yaml
python -m code.src.infer \
    --ckpt runs/baseline/best.pt \
    --sample-submission dataset/sample_submission.csv \
    --out submission.csv
```

The pipeline runs end-to-end on CPU (verified on the synthetic
dataset shipped for testing). On a GPU box, install the CUDA build of
PyTorch instead of the CPU-only build and increase `--batch-size` and
`--image-size`.

## Known limitations and directions

- **Backbone size**: tuned for CPU; switching to a larger backbone
  (e.g. `convnext_base.fb_in22k_ft_in1k`) is the single biggest
  expected gain.
- **No multi-task signal**: `is_digital` and `type` are not
  currently used as auxiliary targets. Adding them as auxiliary
  heads is a known direction (forces shared semantic features
  rather than per-template memorisation).
- **No document-region prior**: the model treats each image as a
  single blob. ID documents have strong structural priors (MRZ,
  portrait region, signature box) that a region-aware model could
  exploit.
- **TTA**: only horizontal flip. Multi-scale TTA and rotation TTA
  are likely cheap wins at inference time.
