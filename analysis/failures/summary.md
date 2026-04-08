# Failure Analysis

For each class, the top 4 highest-confidence false positives and false negatives on the val set, with engineering-framed hypotheses about why each error mode happens. These are starting hypotheses derived from the per-class metrics and class priors - cross-check against the composites and edit anything the images contradict.

## N (Normal) - thr 0.453, 417 positives in val, FP=210, FN=109

The model overpredicts Normal by about 100 cases. Threshold tuner pushed it down to recover recall, which costs precision.

- top FP: ![](analysis/failures/N_fp.png)  Patients with mild D or H whose dominant visual signal still reads as healthy. Not a model fix - either more positive examples of subtle disease or a label-noise audit on cases that carry both N and a disease label.
- top FN: ![](analysis/failures/N_fn.png)  Healthy retinas with cosmetic features (peripheral pigmentation, drusen-like reflections, age-related changes) that look pathological. The conservative N threshold protects recall on disease classes at the cost of N precision; acceptable tradeoff given clinical asymmetry.

## D (Diabetes) - thr 0.596, 433 positives in val, FP=90, FN=143

The model is leaving recall on the table here - 143 misses on the most prevalent disease class. F1 tuning landed on a precision-favoring threshold; if the deployment context wanted higher recall, drop the threshold and accept more FPs.

- top FP: ![](analysis/failures/D_fp.png)  Vessel and exudate patterns that resemble diabetic retinopathy but are aging or hypertension-related. D and H share visual vocabulary in mild cases.
- top FN: ![](analysis/failures/D_fn.png)  Mild non-proliferative cases where the only sign is a few microaneurysms. These are small, low-contrast, and easy to miss at 448px input resolution.

## G (Glaucoma) - thr 0.714, 79 positives in val, FP=25, FN=29

Threshold is high (0.71) and absolute error counts are small. Failures here are individually meaningful rather than a systemic pattern.

- top FP: ![](analysis/failures/G_fp.png)  Likely large optic cup variants - a cup-to-disc ratio looks high but is congenital, not glaucomatous. Hard to disambiguate without OCT or longitudinal imaging, neither of which the model has access to.
- top FN: ![](analysis/failures/G_fn.png)  Early glaucoma with subtle disc rim thinning that is invisible at 448px. Engineering fix worth its cost: a disc-cropped second-pass model at higher resolution for G specifically.

## C (Cataract) - thr 0.752, 87 positives in val, FP=13, FN=14

Strong class, isolated failures. Both error types likely tied to image quality rather than retinal pathology – cataracts manifest as image haze, which the model can confuse with photo-acquisition issues.

- top FP: ![](analysis/failures/C_fp.png)  Underexposed or hazy fundus photos where the lens is actually clear. The model is reading "low-contrast image" as "cataract."
- top FN: ![](analysis/failures/C_fn.png)  Mild cataracts where the fundus is still photographable. The diagnostic signal lives in lens clarity; TTA flips do not help when the issue is global haze.

## A (AMD) - thr 0.511, 79 positives in val, FP=31, FN=27

Errors are roughly balanced, but the absolute count is high relative to class size. ~80 training positives are borderline thin for a class with this much visual variance (dry vs wet, early vs advanced).

- top FP: ![](analysis/failures/A_fp.png)  Drusen-adjacent presentations – peripheral drusen or hard exudates that look macular but are not. Model cannot reliably separate macular from peri-macular findings.
- top FN: ![](analysis/failures/A_fn.png)  Early dry AMD with subtle macular changes. Same root cause as the FPs – too few positive examples for the model to learn the macular boundary.

## H (Hypertension) - thr 0.330, 44 positives in val, FP=78, FN=22

The headline failure mode of the model. The threshold is rock-bottom (0.33), and the model still predicts H 100 times when only 44 are positive. Pure data-volume problem - ~200 training positives is not enough to learn the H/non-H boundary at all. Architecture changes will not fix this; only more H-positive data or a hypertension-specific dataset will.

- top FP: ![](analysis/failures/H_fp.png)  The model is using H as a catch-all for "something looks subtly abnormal, but I cannot place it." Expect a mix of mild D, normal aging vessels, and image-quality artefacts.
- top FN: ![](analysis/failures/H_fn.png)  Mild AV nicking and focal arteriolar narrowing that the model cannot see. These are exactly the cases more training data would unlock.

## M (Myopia) - thr 0.540, 63 positives in val, FP=7, FN=7

Almost perfectly tuned - 7 errors each direction on 63 positives. This class is essentially solved at this dataset scale.

- top FP: ![](analysis/failures/M_fp.png)  Tilted optic discs and peripapillary atrophy that look myopic but are normal anatomical variants.
- top FN: ![](analysis/failures/M_fn.png)  Early myopia where the tessellated fundus pattern has not fully developed. Edge cases; not worth engineering effort to chase.

## O (Other) - thr 0.513, 299 positives in val, FP=172, FN=137

The O class is a labelling sink, not a disease. Errors here mostly reflect label-set design rather than model capability and would not be on my fix list.

- top FP: ![](analysis/failures/O_fp.png)  Genuinely abnormal images that the labeling protocol assigned to a specific class instead of O. The model identified abnormality correctly; the disagreement is upstream.
- top FN: ![](analysis/failures/O_fn.png)  Same direction – the model assigned a specific disease class, and the label said O. Not actually wrong in any clinically useful sense.
