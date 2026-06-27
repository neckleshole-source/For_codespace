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
