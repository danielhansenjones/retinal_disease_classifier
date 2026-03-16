# Retinal Disease Classifier

Multi-label fundus image classifier for the ODIR-5K dataset, detecting 8 ocular conditions simultaneously from paired left/right eye images.

This project is a ground-up rework of my undergraduate capstone, rebuilt to apply what I've learned since about proper problem framing, training methodology, and experiment tracking. The original version used a softmax classifier on a dataset that is inherently multi-label - a fundamental mismatch. This version corrects that and several other design decisions, and is intended to demonstrate applied ML engineering judgment rather than just model accuracy.

---

## Problem Framing

Standard approaches to this dataset apply a softmax classifier - treating it as single-label, mutually exclusive classification. That framing is wrong. Patients frequently present with multiple simultaneous conditions (e.g. Diabetes + Hypertension), and softmax produces contradictory gradient signals when that happens.

This project reframes it correctly as **multi-label classification**:

| Approach | Loss | Activation | Problem |
|----------|------|------------|---------|
| Single-label (naive) | CrossEntropyLoss | Softmax | Forces probabilities to sum to 1, penalises co-occurring labels |
| Multi-label (correct) | BCEWithLogitsLoss | Sigmoid | Independent binary decision per class, handles co-occurrence correctly |

---

## Architecture

**Backbone:** EfficientNet-B4 via `timm` (pretrained on ImageNet)

**Dual-eye fusion:** Both left and right fundus images are passed through the same shared backbone in a single forward pass, features are concatenated, then classified jointly. Some conditions manifest differently or asymmetrically across eyes.

```
left_image  ──┐
               ├── EfficientNet-B4 (shared) ──> concat ──> Dropout(0.5) ──> Linear(features*2, 8)
right_image ──┘
```

**Head:** Single linear layer on concatenated left+right features. No sigmoid at training time - `BCEWithLogitsLoss` expects raw logits.

**Class imbalance:** Per-class `pos_weight = neg / pos` computed from training set and passed to the loss function. Prevents the Normal class (~49% prevalence) from dominating.

---

## Training

Two-phase strategy to preserve pretrained features:

**Phase 1 - Frozen backbone (5 epochs)**
Only the classification head is trained. High LR (1e-3) lets the head converge before touching the pretrained weights.

**Phase 2 - Full fine-tune (up to 25 epochs)**
All layers unfrozen. Low LR (1e-4) with cosine annealing. Gradient checkpointing enabled to reduce VRAM usage. Early stopping on val AUC (patience=7).

**Optimiser:** AdamW with weight decay 1e-3

**Augmentation:**
- RandomResizedCrop(448, scale=0.8–1.0)
- RandomHorizontalFlip, RandomVerticalFlip
- RandomRotation(15°)
- ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)

**Threshold tuning:** After training, per-class decision thresholds are swept on the validation set and selected to maximise F1. A single 0.5 threshold applied universally is clinically inappropriate - missing diabetic retinopathy has different consequences than a false positive on myopia.

---

## Results

Best run - EfficientNet-B4, macro-AUC **0.8741**

| Class     | Condition    | AUC       | F1    | Precision | Recall | Threshold |
|-----------|--------------|-----------|-------|-----------|--------|-----------|
| N         | Normal       | 0.786     | 0.635 | 0.504     | 0.859  | 0.375     |
| D         | Diabetes     | 0.832     | 0.676 | 0.611     | 0.758  | 0.339     |
| G         | Glaucoma     | 0.946     | 0.597 | 0.529     | 0.684  | 0.545     |
| C         | Cataract     | 0.969     | 0.871 | 0.934     | 0.816  | 0.526     |
| A         | AMD          | 0.931     | 0.640 | 0.870     | 0.506  | 0.979     |
| H         | Hypertension | 0.823     | 0.273 | 0.273     | 0.273  | 0.449     |
| M         | Myopia       | 0.995     | 0.874 | 0.929     | 0.825  | 0.353     |
| O         | Other        | 0.711     | 0.482 | 0.432     | 0.545  | 0.421     |
| **macro** |              | **0.874** |       |           |        |           |

**Notable:** Myopia (0.995) and Cataract (0.969) have strong visual signatures that the model captures reliably. Hypertension is the hardest class - the fundus signs (arteriovenous nicking, focal arteriolar narrowing) are subtle and only present in ~5% of cases, limiting training signal. Other is a noisy catch-all label by design.

A macro-AUC of **0.874** is competitive with published results on ODIR-5K. Reported scores in the literature and public leaderboards typically range from ~0.85 to ~0.93 for single-model approaches, with the upper end achieved by larger ensembles or models trained on external fundus datasets. This result sits solidly in that range as a single-model, ODIR-5K-only baseline.

**Where more data or compute would help most:**
- *Hypertension and Other* are the weakest classes. Both suffer from either label noise or low sample counts - more training examples would have a disproportionate impact here versus additional architecture complexity.
- *Dataset scale* is the primary ceiling. ODIR-5K has ~3,500 usable training images. Models trained on larger public fundus datasets (Messidor, EyePACS, APTOS) then fine-tuned here would likely push past 0.90.
- *EfficientNet-B5/B6 or ViT-based backbones* would extract richer features but require more VRAM and longer training - not justified at this data scale.

Experiment tracking via MLflow. Three comparable runs logged; each varied optimizer, learning rate, weight decay, and augmentation strength.

---

## How to Run

```bash
# Install dependencies
uv sync

# Train (results logged to mlruns/)
python main.py

# View experiment tracking UI
mlflow ui
```

Requires ODIR-5K dataset placed at `data/archive/`. Available on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k).

---

## Project Status

| Phase | Description                                               | Status |
|-------|-----------------------------------------------------------|--------|
| 1     | Data pipeline, model, training loop, per-class evaluation | Done   |
| 2     | MLflow experiment tracking, model registry                | Done   |

---

## Key Design Decisions

**Why BCEWithLogitsLoss over CrossEntropyLoss?**
Multi-label problem. Each class is an independent binary decision. CrossEntropyLoss assumes exactly one correct class.

**Why per-class threshold tuning?**
The optimal decision threshold varies by class and by the relative cost of false negatives vs false positives. A model that flags Glaucoma conservatively is preferable to one that misses it.

**Why dual-eye input?**
Conditions like Hypertension and AMD can manifest asymmetrically. Giving the model both eyes at classification time lets it reason over the pair jointly rather than treating each image independently.

**Why two-phase training?**
Fine-tuning all layers from the start with a high LR degrades pretrained ImageNet features. Warming up the head first stabilises training and consistently produces better final AUC.