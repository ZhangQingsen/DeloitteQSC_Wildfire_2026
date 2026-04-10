# Hybrid Quantum–Classical Representation Learning for High‑Precision Wildfire Risk Modeling

This repository contains the full implementation of **Hybrid Quantum–Classical Representation Learning for High‑Precision Spatio‑temporal Wildfire Risk Intensity**, submitted to the **2026 Deloitte Quantum Sustainability Challenge**.  
The project develops a compact, quantum‑enhanced regression framework capable of modeling **continuous wildfire risk intensity** under extreme class imbalance and non‑linear environmental interactions.

---

## 🔥 Project Overview

Wildfire risk modeling suffers from three structural challenges:

- **Extreme sparsity** (fire events < 1%)  
- **Non‑linear climatic interactions**  
- **Spatio‑temporal non‑stationarity**

This project introduces a **hybrid quantum–classical architecture** that embeds classical features into a **4‑qubit PQC bottleneck**, leveraging Hilbert‑space expressivity to achieve **high‑density latent representations** with a physical footprint under **17 KB**.

The full technical report is included in the repository (`/report/`) and submitted as part of Phase 1.

---


---

## 🧠 Methodology Summary

### **1. Data & Target Construction**
- Integrated **125k+** California wildfire panel records (2018–2023)  
- Spatial centroids via **pgeocode**  
- Cyclic month embeddings (sin/cos)  
- Continuous risk target via **Spatio‑temporal Kernel Intensity (STKI)**  
- All features standardized (Z‑score)

### **2. Classical Encoder (MLP‑Attention‑Residual)**
- Feature‑wise attention gate (SE‑style)  
- Residual blocks to prevent information dilution  
- Achieves **MSE ≈ 2.26 × 10⁻⁶**

### **3. Quantum Bottleneck (4‑Qubit PQC)**
- Ry/Rz data embedding  
- CNOT entanglement layers  
- Acts as a **high‑density latent compressor**  
- Physical footprint: **< 17 KB**

### **4. Evaluation**
- Train/val/test strictly from **2018–2021**  
- **2022–2023** held out for zero‑shot forecasting  
- Metrics: R², MSE, prevalence‑matched classification, **risk@k%** tail‑recall

---

## 📊 Key Results

| Model | R² | MSE (×10⁻⁶) | risk@1% | Notes |
|------|----|--------------|---------|-------|
| Linear Regression | 0.6935 | 14928 | 0.0171 | Baseline |
| XGBoost | 0.9890 | 535.1 | 0.0102 | Strong ML benchmark |
| **MLP‑Att‑Res** | **0.9999** | **2.26** | **0.0096** | Best classical |
| **QNN‑Bottleneck** | 0.9985 | 71.4 | 0.0096 | 10× smaller footprint |

### **2023 Zero‑Shot Forecasting**
Using only spatial topology + cyclic month + historical climate averages, the model reconstructs a **smooth, geographically coherent risk surface**, demonstrating strong internalization of California’s risk geometry.

---

## 🚀 Quick Start (Evaluator‑Friendly)

This repository is **not intended as a runnable package**.  
However, evaluators can navigate the implementation as follows:

- **STKI target construction:** `src/stki/`  
- **Classical encoder:** `src/classical_models/att_res_mlp.py`  
- **Quantum bottleneck:** `src/quantum_models/qnn_bottleneck.py`  
- **Training pipeline:** `src/train.py`  
- **Ablation experiments:** `notebooks/`

All hyperparameters and architectural choices match those described in the submitted PDF.

---

## 🧭 Vision: Native Quantum Information Dynamics

Beyond modular PQC substitution, the project outlines a future direction where wildfire risk is modeled as **quantum state evolution**, enabling:

- Interference‑based amplification of **rare tail events**  
- Unitary dynamics for **stochastic hazard propagation**  
- Compact modeling of **multi‑modal risk distributions**

This direction is detailed in Section 5 of the report.

---

## 📚 References

Key references include:

- Diggle (2013) — Spatio‑temporal point processes  
- Ho et al. (2020) — DDPM time embeddings  
- Ronneberger et al. (2015) — U‑Net  
- Attention Residuals (2026)  
- U.S. Census TIGER/Line ZCTA shapefiles (2023)

Full citations are available in the PDF.

---

## 📩 Contact

**Author:** Qingsen Zhang  
**Affiliation:** Binghamton University  
**Email:** qzhang11@binghamton.edu  
**Team:** DeloitteQSC_Wildfire_QSen

---

If you use or reference this work, please cite the accompanying report.


