# 🧠 AI Research Projects: Semantic Clustering & Nonlinear Classification

This repository contains two independent AI research projects focusing on semantic modeling in natural language processing (NLP) and performance evaluation in nonlinear classification tasks using classic and neural methods.

---

## 📁 Project Overview

| Project | Title | Domain |
|--------|-------|--------|
| **1** | Semantic Word Clustering via Textual Distance Graphs | NLP, Unsupervised Learning |
| **2** | Comparative Study of Linear vs. Neural Models on Nonlinear Data | Machine Learning, Supervised Learning |

---

## 🔍 Project 1 — Semantic Word Clustering

This project explores semantic relationships between words through textual co-occurrence analysis and graph-based distance metrics.

### 📝 Method Summary

- **Corpus**: Four novels from [Project Gutenberg](https://www.gutenberg.org/) were cleaned and merged.
- **Vocabulary**: A curated list of 100 semantically meaningful words selected using POS-tagging and frequency-based filtering.
- **Distance Metrics**:
  - **Average Positional Distance**: Greedy matching of word positions across the corpus.
  - **Graph-based Shortest Paths**: Co-occurrence graphs analyzed with **Dijkstra’s Algorithm** (using a window size of 20).
- **Clustering Techniques**:
  - KMeans
  - Hierarchical Clustering
  - DBSCAN
- **Evaluation Metrics**: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index, ARI, NMI

### 📊 Key Insights

- Positional-based clustering reflects narrative proximity.
- Graph-based distances capture local context co-occurrence better.
- Different methods expose distinct semantic groupings—potential for hybrid strategies.

---

## 🧪 Project 2 — Model Comparison on Nonlinear Classification Tasks

This project investigates how linear and neural models handle data with nonlinear decision boundaries.

### ⚙️ Experimental Setup

- **Synthetic Data**: Binary 2D classification with a curved boundary of the form `y = ax² + x`, varying curvature `a ∈ {0.0, 0.5, 1.0, 2.0}`.
- **Models Compared**:
  - Logistic Regression (linear baseline)
  - Feedforward Neural Network (1 hidden layer, 4 ReLU units)

### 📈 Metrics & Conditions

- **Evaluation**: Accuracy, F1-score, confusion matrix
- **Robustness tests**:
  - Varying curvature levels
  - Dataset size (`N = 100` to `5000`)
  - Class imbalance scenarios
  - Network architecture variations (depth, width)

### 📌 Observations

- Logistic Regression struggles with nonlinear boundaries and class imbalance.
- Small NNs show high accuracy and resilience with minimal complexity.
- Depth and unit allocation critically impact NN performance, with optimal results at ~3 layers.

---

## 🛠️ Technologies Used

- Python 3.10
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `networkx`, `nltk`, `seaborn`
- Environments: Jupyter Notebook, LaTeX (for figures/report)

---

## 📂 Repository Structure

```plaintext
ai-research-projects/
├── project1_semantic_clustering/
│   ├── word_distance_analysis.ipynb
│   ├── data/
│   │   └── [cleaned_novel_texts]
│   └── outputs/
│       └── heatmaps, clusters, evaluation plots
├── project2_nonlinear_classification/
│   ├── nonlinear_models_comparison.ipynb
│   └── figures/
│       └── decision_boundaries, metrics_charts
└── README.md
