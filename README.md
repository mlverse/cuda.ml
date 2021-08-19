cuml4r: R interface for cuML
================

[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/cuml4r)](https://cran.r-project.org/package=cuml4r)
<a href="https://www.r-pkg.org/pkg/cuml4r"><img src="https://cranlogs.r-pkg.org/badges/cuml4r?color=brightgreen" style=""></a>

`cuml4r` provides a simple and intuitive R interface for
[cuML](https://github.com/rapidsai/cuml), a suite of GPU-accelerated machine
learning libraries powered by [CUDA](https://en.wikipedia.org/wiki/CUDA).

`cuml4r` is under active development, and currently implements R interfaces for cuML algorithms listed below.

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
| | Inference for decision tree-based models in XGBoost or LightGBM formats using the CuML Forest Inference Library (FIL) | Requires linkage to the Treelite C library when `cuml4r` is installed. Treelite is used for model loading. |
|  | Support Vector Machine Classifier (SVC) | |
|  | Epsilon-Support Vector Regression (SVR) | |

