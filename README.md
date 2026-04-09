# Deloitte QSC 2026 — Wildfire Risk Modeling (Classical + Deep Learning + Quantum)

This repository contains my submission for **Deloitte QSC 2026**, focusing on **wildfire risk modeling** using a hybrid stack of:

- **Classical machine learning**
- **Deep neural networks**
- **Quantum machine learning (QML)** via PennyLane

The goal is to build a robust, interpretable, and forward‑looking pipeline for estimating wildfire risk across U.S. ZIP codes and months, using historical weather and spatial features.

---

## Project Overview

Wildfire risk is a complex function of **space**, **seasonality**, and **weather dynamics**.  
The dataset exhibits:

- Strong nonlinear interactions  
- Heavy‑tailed risk distribution  
- Sparse high‑risk regions  
- Panel structure (ZIP × month × year)

To address these challenges, the project is structured into three modeling layers:

---

## 1. Classical ML Baselines

Implemented models include:

- Linear Regression / Ridge / Lasso  
- Random Forest / Extra Trees  
- XGBoost  
- Classical MLPRegressor  

These baselines provide interpretability and fast iteration, and serve as a reference point for deeper models.

---

## 2. Deep Learning Models

Eight neural architectures were implemented from scratch in PyTorch:

- **MLPPlain / MLPResNet**  
- **MLPAttentionPlain / MLPAttentionRes**  
- **CNN1DPlain / CNN1DResNet**  
- **NAF1DPlain / NAF1DRes**

These models are trained on **2018–2020** data with an internal train/validation split, and evaluated on **2021** as the held‑out test year.

Deep learning models consistently outperform classical ML on MSE and tail‑risk metrics, indicating strong ability to capture smooth nonlinear structure in the KDE‑based risk target.

---

## 3. Quantum Machine Learning (QML)

Quantum models are explored to address two structural challenges in the data:

- **Heavy‑tailed risk distribution**  
- **Sparse high‑risk regions**

These are scenarios where **kernel‑based methods** and **non‑classical feature embeddings** may offer complementary benefits.

Planned QML components include:

### (a) Quantum Kernel Regression (QKR)
Using PennyLane’s quantum kernels to build a high‑expressivity similarity measure for ZIP×month×weather features.

### (b) Quantum Feature Maps + Classical DNN
A hybrid model where a shallow quantum circuit produces nonlinear quantum features that augment the deep neural network backbone.

### (c) Variational Quantum Regressor (VQR) *(optional)*
A fully quantum baseline for comparison under NISQ constraints.

The QML goal is **not** to outperform deep learning globally, but to explore whether quantum feature spaces provide advantages in **tail behavior**, **sparse regions**, or **expressivity**.

---

## Contact

Prepared for **Deloitte QSC 2026**  
Author: *Qingsen*  
Email: *zqs@binghamton.edu*
