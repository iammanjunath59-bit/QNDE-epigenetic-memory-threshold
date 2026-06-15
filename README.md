# Code S1: A Cooperative Binding Stability Boundary for Epigenetic Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Analysis code and processed data for:

> M. Manjunath. "Nucleosome Deformation Dynamics Determine a Quantitative
> Stability Threshold for Epigenetic Memory."
> *Cell Systems* (2026). DOI: [PAPER DOI — to be updated on acceptance]

Zenodo archive (Code S1): DOI: [ZENODO DOI — to be updated on release]

---

## Overview

This repository contains all analysis scripts, pre-computed results, and
figure generation code for the manuscript. The central theoretical result is
the cooperative TF binding stability threshold:

```
Nc = ωκ / (2g²)
```

where ω is the TF dissociation rate, κ is the chromatin relaxation rate, and
g is the TF–nucleosome coupling strength. Scripts 01–10 validate this threshold
genome-scale using CRISPR screen data from human and mouse ESCs, with a
17-TF pluripotency reference panel and a 58-TF systematic negative control.

---

## Repository structure

```
QNDE-epigenetic-memory-threshold/
│
├── 01_ngene_assignment_human.py     # Assigns N_gene to 112 curated human genes
├── 02_ngene_assignment_mouse.py     # Assigns N_gene to 65 curated mouse genes
├── 03_encode_negative_control.py    # Downloads ENCODE peaks; computes general N_gene
├── 04_human_validation.py           # Primary threshold test in human ESCs
├── 05_mouse_validation.py           # Cross-species validation in mouse ESCs
├── 06_negative_control_analysis.py  # Specificity control (58 ENCODE TFs)
├── 07_gillespie_simulations.py      # Exact Gillespie stochastic simulations
├── 08_information_theory.py         # Channel capacity and pe(N) curves
├── 09_sensitivity_analysis.py       # Sensitivity to threshold and window choices
├── 10_cross_species_ratio.py        # Parameter-free cross-species ratio test
│
├── data/
│   ├── raw/
│   │   ├── rosen2024_scores.csv     # Human ESC screen scores (Rosen et al. 2024)
│   │   ├── li2018_scores.csv        # Mouse ESC screen scores (Li et al. 2018)
│   │   └── encode_manifest.csv      # 58 ENCODE accessions + md5 checksums
│   └── processed/
│       ├── human_ngene_112.csv      # N_gene assignments: 112 human genes
│       └── mouse_ngene_65.csv       # N_gene assignments: 65 mouse genes
│
├── params/
│   └── model_params.json            # All biophysical parameters (ω, κ, g, Jeff, Nc)
│
├── results/                         # Pre-computed outputs from scripts 04–10
│   ├── human_validation_results.json
│   ├── mouse_validation_results.json
│   ├── negative_control_results.json
│   ├── gillespie_fpt.csv
│   ├── gillespie_on_fraction.csv
│   ├── sensitivity_results.json
│   └── ratio_analysis.json
│
└── figures/                         # Figure generation scripts (not PNG/PDF)
    ├── fig1_human_validation.py
    ├── fig2_mouse_validation.py
    ├── fig3_information_theory.py
    ├── figS1_gillespie.py
    └── figS2_additional_validation.py
```

---

## Requirements

Python 3.9 or higher. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn, requests.
Full version-pinned list is in requirements.txt.

---

## Data sources

All biological data are re-analysed from published sources.

| Dataset | Reference | Accession |
|---|---|---|
| Human ESC CRISPR screen | Rosen et al. 2024, *Nat Commun* 15:8966 | GEO: GSE277069 |
| Mouse ESC CRISPR screen | Li et al. 2018, *Cell Rep* 24:489–502 | GEO: GSE107060 |
| OCT4/SOX2/NANOG ChIP-seq | Boyer et al. 2005, *Cell* 122:947–956 | GEO: GSE4825 |
| 13-TF ChIP-seq H1 hESC | Chen et al. 2008, *Cell* 133:926–940 | GEO: GSE11431 |
| BRD4/MED1/SE ChIP-seq | Whyte et al. 2013, *Cell* 153:307–319 | GEO: GSE44288 |
| MED1/MED12 ChIP-seq | Kagey et al. 2010, *Nature* 467:430–435 | GEO: GSE20673 |
| 58 ENCODE H1-hESC ChIP-seq | ENCODE Consortium 2012 | See encode_manifest.csv |

Pre-processed score files (data/raw/) and N_gene assignments (data/processed/)
are included in this repository. Raw ENCODE ChIP-seq peak files (~2 GB total)
are NOT included; script 03 downloads them automatically from encodeproject.org.

---

## Running the full analysis

Run scripts in order from the repository root directory:

```bash
# Step 1: Assign N_gene values (uses pre-computed co-occupancy tables)
python 01_ngene_assignment_human.py
python 02_ngene_assignment_mouse.py

# Step 2: Build ENCODE negative control
# NOTE: requires internet connection; downloads ~2 GB; takes 20-30 minutes
python 03_encode_negative_control.py

# Step 3: Statistical validation
python 04_human_validation.py
python 05_mouse_validation.py
python 06_negative_control_analysis.py

# Step 4: Stochastic simulations (takes ~10-15 minutes)
python 07_gillespie_simulations.py

# Step 5: Remaining analyses
python 08_information_theory.py
python 09_sensitivity_analysis.py
python 10_cross_species_ratio.py
```

All outputs are written to results/. Pre-computed results are already
included so figure scripts can be run without re-running the pipeline.

---

## Regenerating figures

All five manuscript figures can be regenerated from the pre-computed results:

```bash
python figures/fig1_human_validation.py
python figures/fig2_mouse_validation.py
python figures/fig3_information_theory.py
python figures/figS1_gillespie.py
python figures/figS2_additional_validation.py
```

Output goes to figures/ as both .pdf (vector, for submission) and
.png (600 DPI, for display). Arial font is used throughout; ensure
Arial is available on your system (standard on Windows and macOS).

---

## Key results (pre-computed)

The following results are reported in the manuscript and are in results/:

| Result | Value | Script |
|---|---|---|
| Human Cohen's d | 0.661 | 04 |
| Human permutation p | 0.0069 (69/10,000) | 04 |
| Human Pearson r (identity) | +0.384 | 04 |
| Human AUC (cross-validation) | 0.729 ± 0.102 | 04 |
| Mouse Cohen's d | 1.489 | 05 |
| Mouse permutation p | 0.0001 (1/10,000) | 05 |
| Cross-species ratio (observed) | 1.71 ± 0.20 | 05, 10 |
| Cross-species ratio (predicted) | 1.82 (g-independent) | 10 |
| Agreement | 6.3% | 10 |
| ENCODE identity r | −0.012 NS | 06 |
| ENCODE fitness r | −0.007 NS | 06 |

---

## Random seeds

All random seeds are fixed for full reproducibility:

| Analysis | Seed | Script |
|---|---|---|
| Permutation test (human) | 99 | 04 |
| Permutation test (mouse) | 99 | 05 |
| Cross-validation | 42 | 04 |
| Gillespie simulations | 42 | 07 |

---

## Note on script 03 (ENCODE download)

Script 03 downloads 58 ENCODE H1-hESC ChIP-seq peak files from
encodeproject.org. Each file is a BED.gz of IDR thresholded narrowPeaks
aligned to GRCh38. Total download size: approximately 2 GB.
md5 checksums are verified automatically after each download.
Experiment accessions and file accessions are in data/raw/encode_manifest.csv.

If you only want to reproduce the figures and statistical results, skip
script 03 — the output of the full pipeline (negative_control_results.json)
is already in results/.

---

## Citation

If you use this code, please cite:

```
M. Manjunath (2026). Nucleosome Deformation Dynamics Determine a
Quantitative Stability Threshold for Epigenetic Memory.
DOI: [PAPER DOI]
```

---

## Contact

M. Manjunath
Department of Genetics and Genomics, University of Mysore
Email: [iammanjunath59@gmail.com]
