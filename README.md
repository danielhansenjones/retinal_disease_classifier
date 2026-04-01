# Retinal Disease Classifier

Multi-label classifier for the ODIR-5K fundus dataset.
Detects 8 ocular conditions from paired left/right eye images using a dual-backbone ensemble with per-backbone normalization, CLAHE preprocessing, and test-time augmentation.

Rebuilt from a university capstone.
The original used softmax on a multi-label dataset - a fundamental framing error.
This version fixes that and a number of other decisions that quietly hurt the original's results.

---

## The Core Problem With Most Approaches to This Dataset

The majority of public notebooks on ODIR-5K use `CrossEntropyLoss` with softmax.
That's wrong.
Patients routinely present with multiple simultaneous conditions - Diabetes and Hypertension together, for instance - and softmax forces the probabilities to sum to 1, which produces contradictory gradients when multiple labels are true.

This is framed as multi-label from the ground up:

|                   | Loss              | Output                | Why                             |
|-------------------|-------------------|-----------------------|---------------------------------|
| Softmax (wrong)   | CrossEntropyLoss  | Mutually exclusive    | Penalises co-occurring labels   |
| Sigmoid (correct) | BCEWithLogitsLoss | Independent per class | Handles co-occurrence correctly |

---

## Architecture

### Dual-eye fusion

A single shared backbone processes both eyes in one forward pass.
Features from each eye are concatenated before the classification head.
This matters for conditions like Hypertension and AMD that can be asymmetric - the model sees both eyes at decision time.

```
left  ──┐
         ├── Backbone (shared) ──> concat ──> Dropout(0.5) ──> Linear(features*2, 8)
right ──┘
```

### Two backbones, trained independently

| Backbone            | Params | Feature dim | Input normalization              |
|---------------------|--------|-------------|----------------------------------|
| EfficientNet-B4     | 19M    | 1792        | ImageNet μ=(0.485, 0.456, 0.406) |
| Inception-ResNet-v2 | 55M    | 1536        | Inception μ=(0.5, 0.5, 0.5)      |

Inception-family pretrained weights expect inputs in [-1, 1].
Using ImageNet normalization with them isn't a minor detail - it shifts the activation distribution the pretrained features were built on, which matters most during the frozen warm-up phase when the head is calibrating to backbone output.

### Ensemble

Both models are loaded simultaneously at inference.
Each normalizes its own inputs internally, so there's a single data-loading path regardless of backbone.
Sigmoid probabilities are averaged across both models and all TTA views.

---

## Preprocessing

CLAHE is applied to every image before augmentation, on both train and val sets.
It runs on the L channel of LAB - not per-channel in RGB, which would shift white balance and produce colour casts.
This enhances local contrast for vessel and lesion visibility without touching hue or saturation.

- `clipLimit=2.0` - standard for fundus imaging; higher values amplify noise in dark retinal regions
- `tileGridSize=(8, 8)` - at 448×448, gives 56×56px tiles, right-sized for optic disc and macula variation

---

## Training

**Phase 1 - head only, 5 epochs, LR 1e-3**
Backbone frozen.
Gets the head to a reasonable starting point before touching pretrained weights.

**Phase 2 - full fine-tune, up to 25 epochs, LR 1e-4**
Cosine annealing, early stopping on val AUC (patience=7).
Gradient checkpointing on EfficientNet-B4 via timm's API; applied manually to the Inception-ResNet-v2 repeat blocks, which don't expose a checkpointing interface.

**Optimizer:** AdamW, weight decay 1e-3

**Augmentation (train only):** RandomResizedCrop(448, scale=0.8–1.0), HorizontalFlip, VerticalFlip, Rotation(15°), ColorJitter

**Class imbalance:** Per-class `pos_weight = neg/pos` from the training set, passed directly to BCEWithLogitsLoss.

**Threshold tuning:** Decision thresholds are optimized per class on the val set to maximize F1.
A fixed 0.5 cutoff applied uniformly ignores the fact that the cost of a missed Glaucoma diagnosis is not the same as a missed Myopia diagnosis.

**TTA:** Four deterministic views at inference - original, horizontal flip, vertical flip, 90° rotation.
Fundus images have no meaningful orientation, so all four are valid.
Probabilities are averaged before thresholding.

---

## Results

| Model               | macro-AUC |
|---------------------|-----------|
| EfficientNet-B4     | 0.870     |
| Inception-ResNet-v2 | 0.885     |
| **Ensemble + TTA**  | **0.888** |

The ensemble gains most in the low-prevalence classes (A, H, O) where single-model predictions are noisiest.
The two backbones make partially uncorrelated errors - EfficientNet's compound-scaled convolutions vs. Inception-ResNet's mixed-kernel residual blocks - which is what makes averaging useful.

**Ensemble + TTA - per-class breakdown**

| Class     | Condition    | AUC       | F1    | Precision | Recall | Threshold |
|-----------|--------------|-----------|-------|-----------|--------|-----------|
| N         | Normal       | 0.821     | 0.659 | 0.595     | 0.739  | 0.453     |
| D         | Diabetes     | 0.860     | 0.717 | 0.764     | 0.674  | 0.596     |
| G         | Glaucoma     | 0.959     | 0.649 | 0.667     | 0.633  | 0.714     |
| C         | Cataract     | 0.980     | 0.857 | 0.852     | 0.862  | 0.752     |
| A         | AMD          | 0.938     | 0.659 | 0.635     | 0.684  | 0.511     |
| H         | Hypertension | 0.816     | 0.306 | 0.220     | 0.500  | 0.330     |
| M         | Myopia       | 0.996     | 0.889 | 0.889     | 0.889  | 0.540     |
| O         | Other        | 0.737     | 0.518 | 0.491     | 0.548  | 0.513     |
| **macro** |              | **0.888** |       |           |        |           |

Myopia and Glaucoma are strong - distinctive visual signatures, clean labels.
Hypertension is the problem class across the board: subtle signs (arteriovenous nicking, focal arteriolar narrowing), ~5% prevalence, and limited positive examples.
Other is a noisy catch-all; its ceiling is a labeling problem, not a modeling one.

The gap to the top of the published range (~0.93) is not an architecture problem.
It's a data problem.
The teams hitting 0.93 are pre-training on EyePACS, MESSIDOR, or APTOS before touching ODIR-5K.
At 3,500 training images with the current label quality, the two classes that would need to move most (H and O) don't have enough signal to get there regardless of what sits on top.

---

## How to Run

```bash
uv sync
python main.py
```

```
Select an option:
  1) Train EfficientNet-B4
  2) Train Inception-ResNet-v2
  3) Evaluate ensemble (both checkpoints must exist)
  4) Exit
```

Checkpoints saved to `checkpoints/<backbone>/best_model.pt`.
Option 3 runs full ensemble inference with TTA and prints per-class metrics.

```bash
mlflow ui   # experiment tracking
```

Dataset: ODIR-5K placed at `data/archive/`. Available on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k).

### Inference API

The ensemble is served via FastAPI. Both backbone checkpoints and the ensemble threshold file must exist before the server will start.

```bash
# Generate ensemble thresholds (option 3 in main.py, only needed once)
python main.py

# Start the server
uvicorn src.api:app
```

```bash
curl -X POST http://localhost:8000/predict \
  -F left=@/path/to/left_fundus.jpg \
  -F right=@/path/to/right_fundus.jpg
```

Returns per-class probabilities and thresholded binary predictions:

```json
{
  "probabilities": {"N": 0.87, "D": 0.04, "G": 0.12, "C": 0.01, "A": 0.03, "H": 0.08, "M": 0.0, "O": 0.15},
  "predictions":   {"N": true, "D": false, "G": false, "C": false, "A": false, "H": false, "M": false, "O": false}
}
```

TTA is on by default (4 views averaged). Pass `?tta=false` to skip it - roughly 4x faster, slightly less accurate.

The preprocessing pipeline matches training exactly: CLAHE on the L channel, resize to 448×448, then the ensemble handles per-backbone normalization internally.

---

## Project Status

| Phase | Description                                               | Status |
|-------|-----------------------------------------------------------|--------|
| 1     | Data pipeline, model, training loop, per-class evaluation | Done   |
| 2     | MLflow experiment tracking, model registry                | Done   |
| 3     | CLAHE preprocessing, dual-backbone ensemble, TTA          | Done   |

---

## Design Decisions

### What worked

**Multi-label framing.**
The most consequential correctness fix in the whole project.
BCEWithLogitsLoss with independent sigmoid outputs is the only sensible choice for this dataset.
Softmax co-occurrence penalty isn't a subtle bias - it actively fights the gradient signal on patients with multiple conditions, which is a significant portion of ODIR-5K.

**Dual-eye input.**
Processing both fundus images jointly rather than independently gives the model access to inter-eye asymmetry, which is a real diagnostic signal for Hypertension and AMD.
The shared backbone with concatenated features also avoids running two separate forward passes, so there's no VRAM penalty for the dual-eye design.

**Two-phase training.**
Freezing the backbone for the first few epochs consistently produces better final AUC than end-to-end training from scratch.
Pretrained ImageNet features are worth preserving - a high-LR head run in epoch 1 will degrade them before the backbone has any chance to adapt.

**Per-class threshold tuning.**
Sweeping thresholds per class on the val set and selecting by F1 is correct for an imbalanced multi-label problem.
A fixed 0.5 threshold assumes balanced classes and symmetric false positive / false negative costs.
Neither is true here.

**Heterogeneous ensemble.**
Pairing EfficientNet-B4 with Inception-ResNet-v2 produces real diversity - compound-scaled convolutions vs. mixed-kernel residual blocks generate partially uncorrelated errors.
Two copies of the same backbone would add compute for minimal variance reduction.

**Correct per-backbone normalization.**
Inception-family pretrained weights expect inputs in [-1, 1], not ImageNet statistics.
Getting this wrong shifts the activation distribution at the input and degrades how much signal you recover from pretraining, particularly during Phase 1 when the backbone is frozen and the head is adapting to its output.

---

### What could be done differently

**CLAHE is unconfirmed although used in the literature.**
Adding CLAHE coincided with a regression in the EfficientNet-B4 run (0.874 → 0.870).
Run-to-run variance on a small val set makes it hard to isolate, but CLAHE may be hurting the Normal class by amplifying subtle variations in healthy images that can resemble early pathology.


**Two models are not enough.**
Averaging two models reduces variance, but the gains flatten quickly.
Teams hitting 0.93+ on this dataset typically run 5–10 model ensembles.
With two, any systematic bias shared between the architectures is still fully present in the final output.

**The val set is too small to trust individual runs.**
At ~700 patients and 80/20 split, per-class sample counts for rare conditions are in the dozens.
AUC estimates at that scale have wide confidence intervals - a 0.003 swing between runs is noise.
K-fold cross-validation would give a more reliable picture of actual generalization, at the cost of significantly more training time.

**Hypertension (H) is unsolved.**
AUC 0.816, F1 0.306.
The model can rank H cases but can't make reliable binary predictions on them.
~5% prevalence in a 3,500-image dataset means fewer than 200 positive training examples.
No architecture change fixes that - it's a data volume problem.

**Pretrained retinal weights are the right next investment.**
The step from 0.888 to the 0.93+ range requires starting from features trained on fundus images, not ImageNet.
Models pretrained on EyePACS or APTOS understand haemorrhages, exudates, and vessel structure in a way ImageNet pretraining doesn't transfer.
More ensembles or augmentation tuning won't close that gap.
