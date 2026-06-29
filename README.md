# The FREUID Challenge 2026 - IJCAI-ECAI
Detect next-generation identity document fraud across physical manipulations, GenAI-driven digital edits, and print-and-capture attacks.
Identity document fraud is changing quickly. Modern attackers can combine high-quality document templates, generative AI editing tools, physical print-and-capture workflows, and presentation attacks to create forgeries that do not look like the simple digital manipulations found in many existing benchmarks.

The FREUID Challenge 2026 asks participants to detect fraudulent identity documents in this more realistic threat setting. The competition is based on the FREUID dataset, a proprietary collection of high-fidelity bona-fide and fraudulent documents contributed by the Microblink Fraud Lab. The dataset is designed to expose failure modes that are often hidden by saturated metrics on purely digital datasets.
#Description 
Participants will train models to distinguish bona-fide identity documents from fraudulent ones. Fraud examples may include:

physical manipulations;
GenAI-driven multimodal edits created with accessible text-and-image tools;
print-and-capture forgeries that close the "analog hole" and suppress many fragile digital artifacts
The challenge deliberately emphasizes document types, scripts, layouts and languages that are under-represented in existing public benchmarks. Strong solutions should therefore generalize across document domains rather than overfit to a narrow set of visual artifacts.

Top-performing teams will be invited to present their methods at IJCAI-ECAI 2026 in Bremen, Germany.
# Why This Matters

Identity fraud is increasing in frequency and sophistication. Existing fraud detection benchmarks have helped the field, but many are now saturated or focused on artifacts that do not survive realistic capture pipelines. FREUID targets three open problems:

Cross-domain generalization: building models that remain accurate on under-represented document types and layouts.
Physical vs. digital artifacts: moving beyond GenAI pixel noise toward semantic, structural and physical inconsistencies.
Anti-fragility in vision models: developing detectors that adapt to evolving fraud strategies instead of memorizing known attack traces.
# Your Task
For each identity document image in the test set, submit a single real-valued score representing the probability or confidence that the document is fraudulent.

Higher scores should indicate a higher likelihood of fraud:

0 label: bona-fide / genuine document
1 label: attack / fraudulent document
The public leaderboard is computed on a validation subset. Final rankings will be computed on a private held-out test set.

# Code Requirements
To support reproducible research, prize-eligible teams must submit:

prediction files through Kaggle;
source code used to train and run the submitted model;
a short technical report describing the method, training data, model architecture, inference procedure and external resources used;
any required environment, dependency or container instructions needed to reproduce the final submission

# Evaluation
Submissions are evaluated with the FREUID Score, a combined metric that uses both the full detection error trade-off curve and a production-relevant operating point.

The score combines:

AuDET: Area under the Detection Error Trade-off curve. This measures the trade-off between false accepts and false rejects across all decision thresholds.
APCER @ 1% BPCER: Attack Presentation Classification Error Rate measured at a fixed 1% Bona-Fide Presentation Classification Error Rate. This captures performance at a strict false-alarm operating point.
Both components are bounded in [0, 1], where lower is better. The final FREUID Score converts each component to a "goodness" score and takes their harmonic mean:
``` text
g_audet = 1 - AuDET
g_apcer = 1 - APCER@1%BPCER
FREUID  = 1 - 2 * g_audet * g_apcer / (g_audet + g_apcer)
```
The combined score is also bounded in `[0, 1]`, and lower is better.

This combination rewards methods that perform well both globally and at the strict operating point. A model that performs well on the overall curve but fails at 1% BPCER will be penalized.
# Submission Format
For every `id` in the test set, submit one numeric fraud score. Higher values mean the document is more likely to be fraudulent.

The submission file should contain a header and use the following format:
``` text
id,label
000001,0.0123
000002,0.8741
000003,0.4310
```

# Metric Direction
The leaderboard metric is minimized:
``` text
lower FREUID Score = better ranking
```
# Files
The competition data will contain labeled training examples and unlabeled test examples.

● `train/` - training document images.

● `test/` - test document images for which participants submit predictions.

● `train.csv` - metadata for the training set, including the row id, image path, binary label, information if example is fully digital or recaptured and type of document.

● `test.csv` - metadata for the test set, including the row id and image path.

● `sample_submission.csv` - the required submission format.
The row id column is the stable key used by Kaggle to align submissions with hidden labels during scoring.

# Labels
The column `label` indicates whether a document is bona-fide (value `0`) or fraudulent (value `1`) while `is_digital` column indicates if example is fully digital (value `1`) or re-captured (printed + captured, value `0`). Column type indicates type of document (in a format `<country>/<document-type>`):

``` text
id,image_path,label,is_digital,type
000001,train/000001.jpg,0,0,USA/DL
000002,train/000002.jpg,1,0,SWITZERLAND/ID
000003,train/000003.jpg,1,1,CROATIA/ID
```
Participants should treat label=1 as the positive class: an attack or fraudulent document.

# Submission Format
For every row in `test.csv`, submit one numeric fraud score. Higher values should indicate higher confidence that the document is fraudulent.

``` text
id,label
000001,0.0123
000002,0.8741
000003,0.4310
```
Scores need to be calibrated probabilities within the range `[0, 1]`. The leaderboard metric uses the ranking and operating-point behavior of these scores through the FREUID Score, which combines AuDET with APCER at 1% BPCER.

# Dataset Description
The FREUID dataset is a proprietary collection of high-fidelity bona-fide and fraudulent identity document samples contributed by the Microblink Fraud Lab. It is designed to benchmark fraud detection systems under a more realistic threat model than purely digital manipulation datasets.

Each sample represents an identity document image. The task is binary fraud detection:

`0` - bona-fide / genuine document
`1` - attack / fraudulent document
Fraudulent examples are designed to cover a broad attack surface, including:

● Physical manipulations on printed and captured document substrates;

● GenAI-driven multimodal edits created with accessible text-and-image tools;

● Print-and-capture forgeries that suppress many fragile digital artifacts and close the "analog hole";

● combinations of physical, digital and recapture effects that require models to detect semantic, structural and physical inconsistencies rather than only pixel-level signatures.

The dataset deliberately includes under-represented document types, scripts, layouts and languages to test cross-document generalization. Strong solutions should perform well across document domains rather than overfit to a small number of template-specific or generator-specific traces.

---

# End-to-end usage (this repo)

This repo runs end-to-end on the shipped sample dataset (64 train / 16 test
images). The same commands work on the full Kaggle FREUID dataset — only
the image folders change.

## 1. Setup

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
pip install google-adk

# dataset/ already contains train/, test/ and the three CSVs (see .gitignore
# to confirm what is shipped). Real-data download path is in
# code/scripts/download_data.py.
```

## 2. Train

```bash
PYTHONPATH=. python -m code.src.train \
    --config code/configs/baseline.yaml \
    --image-size 96 \
    --batch-size 8 \
    --epochs 6 \
    --out-dir runs/freuid_real
```

Best checkpoint → `runs/freuid_real/best.pt`. Val FREUID Score is printed
each epoch. On the shipped 64-image dataset expect best FREUID ≈ 0.80
(modest — random ≈ 0.96 — expected with ~48 training images).

## 3. Produce `submission.csv`

```bash
PYTHONPATH=. python -m code.src.infer \
    --ckpt runs/freuid_real/best.pt \
    --sample-submission dataset/sample_submission.csv \
    --out submission.csv \
    --image-size 96
```

Output: 16 rows aligned to `sample_submission.csv` ids (`000001..000016`),
two columns `id,label`, scores in `[0, 1]`. Submit via the Kaggle
competition page.

## 4. Run the ADK fraud-detection agent

```bash
# one-off classify from the shell
PYTHONPATH=. python -c "
from my_agent.classifier import classify_document
print(classify_document('dataset/train/000001.jpg'))
"

# interactive chat (needs a real GOOGLE_API_KEY in env or .env)
export GOOGLE_API_KEY=...your-key...
adk run my_agent
adk web my_agent
```

The agent exposes a single tool, `classify_document(image_path)`, which
loads the trained checkpoint, applies the eval-time albumentations
pipeline (resize→pad→center-crop→ImageNet-normalize), runs the model with
horizontal-flip TTA, and returns:

```json
{
  "image_path": "...",
  "fraud_score": 0.49,
  "verdict": "review | bona_fide | fraudulent",
  "confidence": 0.04,
  "model": "convnext_tiny.fb_in1k"
}
```

Verdict thresholds: `fraud_score >= 0.65 → fraudulent`,
`fraud_score <= 0.35 → bona_fide`, otherwise `review` (manual secondary
inspection recommended). The model is a first-pass filter only.

## 5. Layout

```
dataset/                 train/test images + CSVs (gitignored except sample CSV)
my_agent/                ADK root agent + classifier tool wrapper
  ├── __init__.py
  ├── agent.py           root_agent + tool registration
  └── classifier.py      lazy-loaded model + classify_document()
code/                    training & inference pipeline (see code/README.md)
runs/freuid_real/        best.pt + history.json + train.log (gitignored)
submission.csv           produced by infer.py (gitignored)
```

## Known limits & directions

- 64-image shipped dataset → val score is noisy; do not trust a single
  baseline run. On the full Kaggle dataset the same code trains
  meaningfully and `code/README.md` lists the expected knobs
  (larger backbone, multi-task heads, region priors, multi-scale TTA).
- No GPU in this dev container → training is CPU-only at `image_size=96`.
  On a GPU box bump `--image-size 224 --batch-size 32` and try a larger
  backbone (`convnext_small` / `convnext_base`) for a bigger jump.
- The `classify_document` tool returns a probability, not a legal
  decision — downstream systems should still apply human review for
  the `review` band before rejecting or accepting a document.
