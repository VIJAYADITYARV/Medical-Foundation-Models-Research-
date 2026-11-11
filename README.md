# Label Purity Versus Supervision Volume: A Multi-Stage Quality-Gated Framework for Weakly-Supervised Medical Image Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Overleaf](https://img.shields.io/badge/Overleaf-IEEE%20Journal%20Draft-green.svg)](https://www.overleaf.com/)

This repository contains the complete experimental code, evaluation notebooks, and IEEE journal manuscript for the **WeKan AI in Healthcare Internship Program** research project.

**Authors:** Vijay Aditya R V (`vijayadityarv@gmail.com`), Harini N (`harinin006@gmail.com`), Regella Krishna Saketh (`krishnasaketh566@gmail.com`)  
**Affiliation:** WeKan Enterprise Solutions India Pvt. Ltd.

---

## 📌 Executive Summary

This study investigates foundation-model-driven weak label generation for medical image segmentation under extreme label scarcity ($N \le 25$ gold-standard cases). 

1. **Replication of Stanford Auto-Generate-WLs (Deshpande et al., MIDL 2024):**
   - **Breast Ultrasound (BUSI):** Bounding-box prompt replication achieves **0.4631 Dice** (closely matching the published 0.4661 baseline). Point-prompt replication achieves **0.6111 Dice** (surpassing published 0.5302).
   - **Dermoscopy (ISIC 2018):** Replicated 25 GS baseline achieves **0.6779 Dice** (0.6585 IoU).
   - **Thoracic Radiography (SIIM-ACR / CANDID-PTX):** End-to-end pipeline implementation for pneumothorax contouring.
2. **Empirical Analysis of Multi-Gate Quality Verification:**
   - Evaluated a 5-stage sequential cascade: Complexity Gating (Random Forest), Transformation Consistency (Multi-angle TTA), Morphological Plausibility V2, Radiomics Quality Regression (Gradient Boosting), and Curriculum Easy-First Ordering.
   - Identified the **Volume-Precision Pareto Frontier**: strict multi-gate filtering approves only 6.7% of BUSI and 3.0% of ISIC scans. Retraining on minimal, highly-filtered subsets (33 and 6 weak labels) demonstrates that restricting supervisory volume reduces downstream generalization compared to high-volume regimes.

---

## 📁 Repository Structure

```
├── BUSI_novelties.ipynb             # Breast ultrasound replication, cascade & TTA
├── ISIC_novelties_FINAL.ipynb       # Dermoscopy lesion segmentation & gating
├── ISIC_novelties4.ipynb            # Dermoscopy exploratory experiments
├── CANDIDPTX_novelties4.ipynb       # SIIM-ACR pneumothorax chest radiography pipeline
├── Paper (1).pdf                    # Reference paper (Deshpande et al., MIDL 2024)
├── WeKan_IEEE_Journal_Overleaf.zip  # Overleaf-ready zip archive (LaTeX + Figures)
└── WeKan_IEEE_Paper_Final_overleaf/ # IEEE Journal LaTeX source and assets
    ├── WeKan_IEEE_Paper_Final.tex   # Primary IEEE journal manuscript
    ├── make_figs.py                 # Figure rendering script
    └── figs/                        # High-resolution vector/raster figures
        ├── fig1_pipeline.png        # System architecture diagram
        ├── fig2_results.png         # Replicated vs. published benchmarks
        ├── fig3_cascade.png         # Multi-gate candidate attrition breakdown
        └── fig4_contrastive.png     # SimCLR contrastive pretraining loss curves
```

---

## 📊 Summary of Verified Results

| Dataset | Supervision Configuration | Published Dice (MIDL 2024) | Replicated Dice (Ours) | Notebook Source |
| :--- | :--- | :---: | :---: | :--- |
| **BUSI** | 25 GS Baseline (0 WL) | 0.3059 | **0.4209** | `BUSI_novelties.ipynb`, Cell 17 |
| **BUSI** | 25 GS + 100 WL (Box Prompts) | 0.4661 | **0.4631** | `BUSI_novelties.ipynb`, Cell 40 |
| **BUSI** | 25 GS + 100 WL (Box) + Flip TTA | --- | **0.5041** | `BUSI_novelties.ipynb`, Cell 40 |
| **BUSI** | 25 GS + 100 WL (Point Prompts) | 0.5302 | **0.6111** | `BUSI_novelties.ipynb`, Cell 41 |
| **BUSI** | 20 GS + 33 Cascade WL (DW Loss) | --- | **0.3905** | `BUSI_novelties.ipynb`, Cell 37 |
| **ISIC 2018** | 25 GS Baseline (0 WL) | 0.6123* | **0.6779** | `ISIC_novelties_FINAL.ipynb`, Cell 17 |
| **ISIC 2018** | 20 GS + 6 Cascade WL (DW Loss) | --- | **0.6149** | `ISIC_novelties_FINAL.ipynb`, Cell 36 |

*\*Note: Published ISIC score corresponds to the ISBI 2016 dataset; local replication utilized ISIC 2018 Task 1.*

---

## 🚀 Reproduction Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/VIJAYADITYARV/Medical-Foundation-Models-Research-.git
cd Medical-Foundation-Models-Research-

conda create -n wekan-medsam python=3.10 -y
conda activate wekan-medsam
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install segment-anything albumentations==1.3.1 scikit-image scikit-learn opencv-python tqdm pydicom kagglehub
```

### 2. Running on Google Colab
Open any of the provided notebooks in Google Colab with a GPU runtime:
* [`BUSI_novelties.ipynb`](BUSI_novelties.ipynb)
* [`ISIC_novelties_FINAL.ipynb`](ISIC_novelties_FINAL.ipynb)
* [`CANDIDPTX_novelties4.ipynb`](CANDIDPTX_novelties4.ipynb)

### 3. Compiling the Research Paper on Overleaf
1. Log in to [Overleaf](https://www.overleaf.com/).
2. Click **New Project** $\to$ **Upload Project**.
3. Select `WeKan_IEEE_Journal_Overleaf.zip`.
4. Ensure the compiler is set to **pdfLaTeX** and compile.

---

## 📜 Citation & Reference

If you build upon this work or utilize the evaluation protocols, please cite:

```bibtex
@article{rv2026qualitygated,
  title={Replication and Quality-Gated Extension of Automated Weak Supervision for Label-Scarce Medical Image Segmentation},
  author={R\,V, Vijay Aditya and N, Harini and Saketh, Regella Krishna},
  journal={IEEE Journal Draft},
  year={2026},
  note={WeKan Enterprise Solutions India Pvt. Ltd., AI in Healthcare Internship Program}
}
```

Reference Baseline:
```bibtex
@inproceedings{deshpande2024auto,
  title={Auto-Generating Weak Labels for Real \& Synthetic Data to Improve Label-Scarce Medical Image Segmentation},
  author={Deshpande, Tanvi and Prakash, Eva and Ross, Elsie Gyang and Langlotz, Curtis and Ng, Andrew and Valanarasu, Jeya Maria Jose},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  series={PMLR},
  volume={227},
  pages={1--15},
  year={2024}
}
```
