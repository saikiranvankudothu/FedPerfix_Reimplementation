# FedPerfix on OrganAMNIST (Low-Resource Jupyter Implementation)

A step-by-step Jupyter Notebook re-implementation of **FedPerfix** (Prefix-Tuning based Personalization for Federated Vision Transformers) on the **OrganAMNIST** dataset from MedMNIST — optimized for **CPU**-only environments and small-scale experiments.

---

## What is This?

This project demonstrates:

- Prefix tuning for **personalized federated learning**
- Using **ViT-Small (Vision Transformer)** from `timm`
- Personalized prefixes and **shared backbone**
- Federated Averaging (FedAvg) only on shared weights
- Minimal setup: 4 clients, 1000 samples total

---

## Features

- [x] Data preprocessing to `.npz` format
- [x] Visualize client-wise class distribution
- [x] Prefix-tuned ViT using `timm`
- [x] Local training on client prefix only
- [x] FedAvg aggregation on shared base
- [x] Per-round accuracy visualization

---

## Setup

```bash
pip install medmnist timm ujson scikit-learn matplotlib
```

---

## Results (Sample)

- 4 clients trained on 1000 samples (non-IID)

- Prefix-tuned ViT-Small

- Accuracy improves round by round using FedPerfix

## Each client gets a personalized prefix, making this better than vanilla FedAvg on heterogeneous data.

## How to Run

### 1. Prepare the dataset:

- Use generate_medmnist.py or provided .npz samples

- Create folder: organamnist4/

### 2. Run all cells in the notebook FedPerfix_OrganAMNIST.ipynb:

- Starts with data loading

- Builds and trains prefix-tuned ViT models

- Aggregates base weights across rounds

- Evaluates and visualizes results

---

### References

FedPerfix Paper: https://arxiv.org/abs/2308.09160

MedMNIST: https://medmnist.com/

timm (Vision Transformers): https://github.com/huggingface/pytorch-image-models
