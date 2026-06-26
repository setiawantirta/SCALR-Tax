# SCALR-Tax
### Scalable Taxonomic Classification using Hypervariable 16S rRNA Regions with Incremental PCA and Bayesian Optimized Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

# Overview

**SCALR-Tax** is the official reproducibility repository accompanying our manuscript submitted to the **Journal of Chemical Information and Modeling (ACS)**.

This repository provides the complete computational workflow used for scalable microbial taxonomic classification from 16S rRNA hypervariable regions. The pipeline integrates

- automatic sequence preprocessing
- taxonomy extraction
- ambiguity analysis
- feature vectorization
- Incremental PCA-based dimensionality reduction
- Bayesian-optimized machine learning
- comprehensive benchmarking and visualization

The objective of this repository is to ensure **full computational reproducibility** of the experiments reported in the manuscript.

---
![Pipeline](figures/pipeline.png)

# Repository Structure

```
SCALR-Tax/
│
├── benchmark/                 # Core source code
├── notebooks/
│     └── rki-20-nov2025.ipynb # Main reproducibility notebook
│
├── datasets/                  # Dataset location (after download)
├── results/
├── figures/
└── README.md
```

---

# Computational Workflow

The notebook reproduces the complete experimental pipeline used in the manuscript.

The workflow consists of the following stages:

1. Environment setup
2. Repository initialization
3. Dataset verification
4. Taxonomic level extraction
5. Ambiguous taxonomy analysis
6. Exploratory dataset visualization
7. Two-dimensional visualization
8. Sequence vectorization
9. Feature reduction using Incremental PCA
10. Bayesian optimization and benchmark execution
11. Model validation
12. Result export

Each stage is organized into an independent notebook section for easy execution and inspection.

---

# Dataset

The complete V3–V4 and V4–V5 hypervariable region datasets are hosted on Kaggle.

Download using:

```bash
#!/bin/bash

curl -L -o ~/Downloads/v3-v4-hyper-region-filter.zip \
https://www.kaggle.com/api/v1/datasets/download/indoborutoofficial/v3-v4-hyper-region-filter
```

or using Kaggle CLI

```bash
kaggle datasets download \
indoborutoofficial/v3-v4-hyper-region-filter
```

Extract the downloaded archive into the project directory:

```
datasets/
```

so that the notebook can locate all required input files.

---

# Installation

Clone the repository

```bash
git clone https://github.com/setiawantirta/SCALR-Tax.git

cd SCALR-Tax
```

Install dependencies

```bash
pip install -r requirements.txt
```

or create the conda environment if provided.

---

# Running the Notebook

Open

```
notebooks/rki-20-nov2025.ipynb
```

Execute every section sequentially.

The notebook is organized as follows.

| Section | Description |
|----------|-------------|
| Setting | Environment initialization |
| Check Data | Verify dataset integrity |
| Extract Level | Taxonomic hierarchy extraction |
| Check Ambiguous | Detection of ambiguous taxonomy |
| Plot Analysis Dataset | Dataset exploration |
| Plot 2 Component | Visualization |
| Vectorization | Sequence feature generation |
| Feature Reduction | Incremental PCA |
| Model Validation | Bayesian optimization and model evaluation |
| Finish | Export results |

Running all notebook cells reproduces the experiments presented in the manuscript.

---

# Expected Outputs

The notebook generates

- processed taxonomy tables
- extracted taxonomic hierarchy
- ambiguity reports
- feature matrices
- Incremental PCA models
- benchmark results
- trained classifiers
- evaluation metrics
- publication-quality figures

---

# Reproducibility

For complete reproducibility, please ensure

- identical Python version
- identical package versions
- downloaded Kaggle dataset
- sequential notebook execution

Random seeds are fixed wherever applicable to ensure deterministic results.

---

# Citation

If you use this repository, please cite our manuscript.

```
Citation information will be updated after publication.
```

---

# License

This project is released under the MIT License.

---

# Contact

**Setiawan Tirta**

GitHub

https://github.com/setiawantirta

---

### Reproducibility Statement

This repository contains the complete source code, notebook workflow, preprocessing pipeline, benchmarking framework, and dataset preparation instructions required to reproduce all computational experiments reported in the accompanying manuscript. Every figure, benchmark result, and evaluation presented in the paper can be regenerated directly from this repository using the publicly available datasets and the documented execution workflow.
