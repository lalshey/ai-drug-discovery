# 🧬 AI Drug Discovery Pipeline

A machine learning pipeline built to predict whether small molecule drugs bind to **EGFR** — one of the most well studied receptor proteins in lung cancer research. This project was built from scratch using real pharmaceutical bioactivity data, and the workflow closely mirrors what computational chemistry teams at companies like Roche, Novartis and AstraZeneca actually do.

---

## Why I Built This

I wanted to build something that sits at the intersection of biology and machine learning — not just a textbook ML project, but something grounded in real science. Drug discovery is one of the most exciting applications of AI right now, and EGFR felt like the right target to start with given how extensively it's been studied in lung cancer.

---

## What It Does

You input a molecule as a SMILES string. The pipeline converts it into a molecular fingerprint, runs it through three independently trained models, and returns a consensus prediction on whether the molecule is likely to bind to EGFR — along with a full physicochemical analysis of the molecule itself.

The app doesn't just return a yes or no. It tells you:
- What each of the three models thinks individually
- How confident each model is
- Whether there's enough agreement to trust the prediction
- Whether the molecule has the physicochemical properties to actually work as a drug

---

## The Pipeline

**Data** — 3,071 real bioactivity records pulled from PubChem's pharmaceutical database, covering molecules experimentally tested against EGFR.

**Feature Engineering** — Each molecule's SMILES string is processed using RDKit into a 2048-bit Morgan Fingerprint — a binary representation of the chemical substructures present in the molecule. This is a standard approach in cheminformatics.

**Models Trained**

Three models were independently trained and benchmarked:

| Model | Accuracy | F1 Score |
|---|---|---|
| Random Forest | 85.0% | 0.92 |
| XGBoost | 85.0% | 0.92 |
| Graph Neural Network | 83.9% | 0.91 |

Random Forest and XGBoost both operate on the fingerprint vectors. The GNN treats each molecule as a graph — atoms as nodes, bonds as edges — and learns structural patterns directly. All three were benchmarked against each other; Random Forest and XGBoost tied, and the GNN came close behind. The convergence across three different algorithmic approaches gives confidence that 84-85% represents a genuine performance ceiling for this dataset size, not an overfitting artefact.

SMOTE oversampling and class-weight balancing were also explored to address the natural class imbalance in pharmaceutical screening data, but the original models proved most stable overall.

**Ensemble Consensus** — Rather than relying on any single model, all three predict simultaneously. A 2/3 majority vote determines the final verdict. This reduces false positives — a critical consideration when a wrong prediction could send a compound into expensive wet lab validation.

**Confidence Thresholding** — Predictions below 70% confidence are flagged as inconclusive rather than forced into an active/inactive label. In drug screening, an uncertain answer is more useful than a wrong confident one.

---

## Molecular Analytics

Beyond binding prediction, the app analyses each molecule's physicochemical properties — molecular weight, lipophilicity, hydrogen bonding capacity, polar surface area — and evaluates them against Lipinski's Rule of 5. A molecule that binds EGFR but can't be absorbed orally has limited clinical value, so this layer was important to include.

---

## The Web App

Built with Streamlit. Four tabs covering ensemble prediction with full per-model breakdown, model benchmarking charts, dataset insights, and project documentation. Anyone can drop in a SMILES string and get a complete computational profile of the molecule.

---

## Tech Stack

Python · RDKit · Scikit-learn · XGBoost · PyTorch Geometric · Streamlit · PubChem API · Pandas · NumPy · Matplotlib

---

## Running It Locally

```bash
git clone https://github.com/lalshey/ai-drug-discovery.git
cd ai-drug-discovery
conda create -n drugproject python=3.10
conda activate drugproject
pip install rdkit chembl-webresource-client scikit-learn xgboost pandas numpy streamlit matplotlib torch torch-geometric
python step
