R interface for cuML
================

[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/cuml)](https://cran.r-project.org/package=cuml)
<a href="https://www.r-pkg.org/pkg/cuml"><img src="https://cranlogs.r-pkg.org/badges/cuml?color=brightgreen" style=""></a>

This package provides a simple and intuitive R interface for RAPIDS
[cuML](https://github.com/rapidsai/cuml), a suite of GPU-accelerated machine
learning libraries powered by [CUDA](https://en.wikipedia.org/wiki/CUDA).
It is under active development, and currently implements R interfaces for algorithms listed below
(which is a subset of [algorithms supported by RAPIDS cuML](https://github.com/rapidsai/cuml#supported-algorithms)).

### Supported Algorithms
| Category | Algorithm | Notes |
| --- | --- | --- |
| **Clustering** |  Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | Only single-GPU implementation is supported at the moment |
|  | K-Means | Only single-GPU implementation is supported at the moment |
|  | Single-Linkage Agglomerative Clustering | |
| **Dimensionality Reduction** | Principal Components Analysis (PCA) | Only single-GPU implementation is supported at the moment |
| | Truncated Singular Value Decomposition (tSVD) | Only single-GPU implementation is supported at the moment |
| | Uniform Manifold Approximation and Projection (UMAP) | Only single-GPU implementation is supported at the moment |
| | Random Projection | |
| | t-Distributed Stochastic Neighbor Embedding (TSNE) | |
| **Nonlinear Models for Regression or Classification** | Random Forest (RF) Classification | Only single-GPU implementation is supported at the moment |
| | Random Forest (RF) Regression | Only single-GPU implementation is supported at the moment |
| | Inference for decision tree-based models in XGBoost or LightGBM formats using the CuML Forest Inference Library (FIL) | Requires linkage to the Treelite C library when {cuml} is installed. Treelite is used for model loading. |
|  | K-Nearest Neighbors (KNN) Classification | Uses [Faiss](https://github.com/facebookresearch/faiss) for Nearest Neighbors Query. |
|  | K-Nearest Neighbors (KNN) Regression | Uses [Faiss](https://github.com/facebookresearch/faiss) for Nearest Neighbors Query. |
|  | Support Vector Machine Classifier (SVC) | |
|  | Epsilon-Support Vector Regression (SVR) | |

