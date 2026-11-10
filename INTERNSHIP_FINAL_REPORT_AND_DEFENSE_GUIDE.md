# WeKan AI in Healthcare Internship: Final Research Defense Guide & Technical Summary

**Project Title:** Quality-Gated Weak Supervision with Medical Foundation Models: An Empirical Multi-Dataset Replication and Verification Analysis  
**Authors:** Vijay Aditya R V (vijayadityarv@gmail.com), Harini N (harinin006@gmail.com), Regella Krishna Saketh (krishnasaketh566@gmail.com)  
**Institution:** WeKan Enterprise Solutions India Pvt. Ltd., AI in Healthcare Internship Program  
**Target Venue:** IEEE Journal (Single-Column Submission Draft)  
**Base Reference Paper:** *Auto-Generating Weak Labels for Real & Synthetic Data to Improve Label-Scarce Medical Image Segmentation*, Deshpande et al. (Stanford University / MIDL 2024)

---

## 1. Executive Summary & Core Research Story

### The Problem
Medical image segmentation is severely constrained by **label scarcity**. Annotating ultrasound scans, dermoscopy images, or chest X-rays pixel-by-pixel requires expert radiologists and pathologists. While obtaining hundreds of gold-standard masks is practically infeasible in clinical pilot deployments, acquiring 20 to 50 labeled scans alongside an unannotated pool is commonplace.

### The Foundation Model Opportunity & Limitations
Foundation models like MedSAM (fine-tuned on 1.5M medical scans) offer powerful zero-shot segmentation, but:
1. They are **prompt-dependent** (requiring accurate bounding boxes or points).
2. Autonomous prompting modes either over-segment background noise (SAM) or collapse to blank masks (MedSAM).
3. Coarse bounding boxes lead to severe under- or over-segmentation.

### What Deshpande et al. (MIDL 2024) Did
Trained an initial lightweight UNet++ on $\le 25$ gold-standard (GS) images, generated automated prompts (boxes/points) from coarse predictions on unlabeled images, queried MedSAM for weak labels (WLs), discarded masks with $>97\%$ uniform area, and retrained UNet++ on GS + WL.

### What We Did (The WeKan Project)
1. **Faithfully replicated** the base pipeline on Breast Ultrasound (BUSI), Dermoscopy (ISIC 2018), and Thoracic Radiographs.
2. **Evaluated 5 novel quality-control enhancements:**
   - Sequential 5-tier Quality Cascade: (1) Random Forest Complexity Gate, (2) Multi-angle Rotation Consistency Check, (3) MedSAM Centroid/Box Refinement, (4) Morphological Gate V2, and (5) Gradient Boosting Radiomics Quality Predictor.
   - Curriculum Easy-First Sample Ordering.
   - Contour-Distance-Weighted Retraining Loss (attenuating boundary noise).
   - Self-Supervised SimCLR Contrastive Pretraining.
   - Dual-Perspective Test-Time Augmentation (TTA).
3. **Discovered the Volume-Precision Pareto Frontier:**
   - Multi-gate quality verification achieves extreme mask purity, but rejected 93.3% of BUSI scans and 97% of ISIC scans.
   - Training deep networks on tiny, ultra-pure sets (33 BUSI WLs, 6 ISIC WLs) yields lower test scores than baselines.
   - Conversely, high-volume regimes (100 WLs) achieve **0.4631 Dice** on box prompts (matching published 0.4661) and **0.6111 Dice** on point prompts (exceeding published 0.5302 by $+15.3\%$).

---

## 2. Complete Empirical Scorecard & Notebook Provenance

| Metric / Experiment | Published (MIDL 2024) | Replicated (Ours) | Difference ($\Delta$) | Notebook & Cell Index |
| :--- | :---: | :---: | :---: | :--- |
| **BUSI 25 GS Baseline** | 0.3059 | **0.4209** | $+0.1150$ | `BUSI_novelties.ipynb`, Cells 17, 37 |
| **BUSI 25 GS + 100 WL (Box)** | 0.4661 | **0.4631** | $-0.0030$ | `BUSI_novelties.ipynb`, Cell 40 |
| **BUSI 100 WL (Box) + Horiz TTA** | --- | **0.5041** | $+0.0410$ | `BUSI_novelties.ipynb`, Cell 40 |
| **BUSI 25 GS + 100 WL (Points)** | 0.5302 | **0.6111** | $\mathbf{+0.0809}$ | `BUSI_novelties.ipynb`, Cell 41 |
| **BUSI Enhanced (20 GS + 33 WL, DW Loss)**| --- | **0.3905** | $-0.0304$ vs GS | `BUSI_novelties.ipynb`, Cell 37 |
| **ISIC 2018 25 GS Baseline** | 0.6123* | **0.6779** | $+0.0656$ | `ISIC_novelties_FINAL.ipynb`, Cells 17, 36 |
| **ISIC Enhanced (20 GS + 6 WL, DW Loss)** | --- | **0.6149** | $-0.0630$ vs GS | `ISIC_novelties_FINAL.ipynb`, Cell 36 |
| **CANDID-PTX 25 GS Baseline** | 0.8182 | Published Ref | --- | `Paper (1).pdf`, Table 1 |
| **CANDID-PTX 25 GS + 100 WL (Box)** | 0.9096 | Published Ref | --- | `Paper (1).pdf`, Table 1 |

*\*Published ISIC evaluated on ISBI 2016 (1279 images); our replication evaluated on ISIC 2018 Task 1 (2594 images).*

---

## 3. Defense Cheatsheet: Answers to Critical Reviewer Questions

### Q1: Why did your point-prompt model (0.6111) beat the bounding-box model (0.4631) on BUSI?
> **Answer:** In breast ultrasound, acoustic speckle noise, posterior shadowing, and low tissue contrast make lesion boundaries ambiguous. A bounding box forced around the tumor often captures non-target hypoechoic tissue, causing MedSAM to bleed into healthy stroma. In contrast, our point-prompting strategy places the positive prompt on the lesion centroid and samples low-confidence background coordinates as negative anchor points. This gives MedSAM clear foreground/background polarity without bounding-box boundary confusion, unlocking a +0.1480 Dice increase.

### Q2: Why did the multi-gate cascade lower test scores (0.3905 on BUSI, 0.6149 on ISIC) compared to baselines?
> **Answer:** This reveals a fundamental scientific insight: **The Volume-Precision Pareto Frontier in Weak Supervision**. 
> - Deep convolutional networks (UNet++) have inherent capacity to filter out moderate, non-systematic label noise during backpropagation, provided they are trained on sufficient sample volume and geometric diversity.
> - Our five-stage cascade was parameterized conservatively: the Random Forest complexity gate, rotation consistency check ($\ge 0.20$ IoU), morphological gate, and quality regressor filtered out 93.3% of BUSI candidates and 97.0% of ISIC candidates.
> - Retraining on only 33 BUSI weak labels and 6 ISIC weak labels (further down-weighted by $\lambda = 0.5$ in the distance-weighted loss) starved the network of structural variability. Restoring volume to 100 weak labels yielded the winning scores (0.4631 box, 0.6111 points).

### Q3: Why do `eval.py` and `eval_tta.py` output different numbers on the exact same model weights?
> **Answer:** This stems from differences in metric aggregation protocols:
> - `eval.py` calculates standard **macro-averaging** (computing Dice per individual image and taking the mean across the 129 test scans).
> - `eval_tta.py` performs **micro-averaging / global pooling** (summing intersections across the entire dataset divided by the sum of unions).
> - Furthermore, differences in tensor unpadding interpolation (nearest-neighbor vs. bilinear) and epsilon smoothing on near-empty masks create small numerical offsets. Under `eval_tta.py`, our enhanced BUSI model scores 0.3966 vs. 0.3846 baseline, and TTA yields consistent within-script increments (+0.001 to +0.019 Dice).

### Q4: What is the status of the CANDID-PTX / SIIM-ACR experiment?
> **Answer:** In `CANDIDPTX_novelties4.ipynb`, the full pipeline was implemented in code (DICOM decompression, run-length decoding of pneumothorax masks, UNet++ training, multi-gate generation, and retraining). However, due to remote Colab runtime limits and Kaggle API download timeouts, test metrics were not saved. To maintain 100% scientific integrity, we do not claim unverified empirical numbers for this modality. Instead, we formulate a grounded theoretical framework adapting bilateral symmetry and visceral pleural distance losses to thoracic X-rays, aligned with the published CANDID-PTX progression (0.8182 to 0.9096).

### Q5: What happened with the SimCLR contrastive pretraining?
> **Answer:** Contrastive pretraining on unannotated scans was successfully executed and logged, showing steady NT-Xent loss reduction (0.7060 to 0.5020 on BUSI; 4.1071 to 3.6013 on ISIC). However, the resulting `.pth` checkpoint was not loaded into the downstream supervised training cell in the notebook. We report this transparently in our component audit table so reviewers know exactly what was executed and connected.

---

## 4. How to Submit & Compile on Overleaf

1. Download [`WeKan_IEEE_Journal_Overleaf.zip`](WeKan_IEEE_Journal_Overleaf.zip).
2. Go to [Overleaf](https://www.overleaf.com/) $\to$ **New Project** $\to$ **Upload Project**.
3. Drag and drop `WeKan_IEEE_Journal_Overleaf.zip`.
4. The project will compile automatically with pdfLaTeX into an IEEE journal manuscript (~14–16 pages).
