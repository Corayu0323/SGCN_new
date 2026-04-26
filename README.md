# 📌 SGCN: Subgraph-based Graph Convolutional Network

## 1. Project Overview

This project implements a **Subgraph-based Graph Convolutional Network (SGCN)** framework for scalable and robust training on large graphs.

The core idea is:

* Instead of training on the full graph
* We **sample multiple subgraphs**
* Train **independent local GNN models**
* Perform **truncation + parameter aggregation**

This design aims to:

* Improve **scalability** (large graphs)
* Enhance **robustness** (noise / wrong edges)
* Enable **parallel training**

---

## 2. Method Pipeline

```
Full Graph
   ↓
Subgraph Sampling (R subgraphs)
   ↓
Local Training (L epochs each)
   ↓
Validation Scoring
   ↓
Truncation (drop worst ρ%)
   ↓
Parameter Aggregation
   ↓
Updated Global Model
```

---

## 3. Core Components

### 3.1 Subgraph Sampling

Supported strategies:

* `random_node`
* `random_edge`
* `random_walk`
* `snowball`

---

### 3.2 Local Training

Each subgraph is trained:

* Independently
* Without parameter sharing
* For **L local epochs**

---

### 3.3 Truncation Mechanism

Keep only top-performing subgraphs:

l_{\text{top}} = \text{Top}_{\rho}(s_1, \dots, s_R)

---

### 3.4 Parameter Aggregation

\tilde{\theta} = \sum_{r \in l_{\text{top}}} w_r \theta_r

* `sgcn` (softmax weighting)
* `avg` (uniform averaging)
* `weighted` (linear weighting)

---

## 4. Supported Models

* GCN
* GraphSAGE
* GraphSAINT (baseline)
* **SGCN (proposed)**

---

## 5. Dataset

Currently tested on:

* **OGBN-Proteins** (large-scale biological graph)
* Multi-label node classification (112 labels)

---

## 6. Project Structure

```
project/
│
├── notebook/
│   └── main_experiment.ipynb
│
├── src/
│   ├── models.py          # GNN architectures
│   ├── train.py           # training loops
│   ├── sgcn.py            # SGCN logic
│   ├── sampling.py        # subgraph sampling
│   ├── utils.py           # utilities
│   └── logging_utils.py   # experiment logging
│
├── results/
│   ├── csv/
│   ├── figures/
│   └── exp_*/
│
└── README.md
```

---

## 7. Environment & Dependencies

### 7.1 Core Libraries

```bash
Python            3.10
torch             2.4.0+cu118
torch-geometric   2.7.0
ogb               latest
numpy             latest
scipy             latest
pandas            latest
scikit-learn      latest
networkx          latest
tqdm              latest
matplotlib        latest
pyyaml            latest
```

---

### 7.2 PyG Installation

```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
-f https://data.pyg.org/whl/torch-2.4.0+cu118.html

pip install torch-geometric
```

---

### 7.3 GPU Environment

Example:

```
GPU: Tesla P100-SXM2-16GB
CUDA: 11.8
```

---

## 8. Training Details

### Key Hyperparameters

| Parameter          | Description                       |
| ------------------ | --------------------------------- |
| `N_SUBGRAPHS (R)`  | number of subgraphs per epoch     |
| `LOCAL_EPOCHS (L)` | local training steps per subgraph |
| `TRUNCATION_RATIO` | fraction of subgraphs dropped     |
| `AGG_METHOD`       | aggregation method                |
| `HIDDEN_DIM`       | hidden size                       |
| `N_LAYERS`         | number of GNN layers              |

---

### Time Measurement

The framework explicitly records:

* `train_sampling_time`
* `train_epoch_time`
* `eval_time`
* `total_run_time`

Ensures **fair comparison with baseline methods**

---

## 9. Current Status

### ✅ Completed

* SGCN training pipeline
* Multiple sampling strategies
* Parameter aggregation
* Time profiling
* Baseline comparison (GCN / SAGE / SAINT)

### ⚠️ In Progress

* Robustness evaluation (noisy edges)
* Biological dataset migration (TCGA / PPI)
* Label trick correction (OGBN-Proteins)
* Multi-run statistical analysis

---

