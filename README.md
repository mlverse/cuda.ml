R interface for cuML
================

[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/cuml4r)](https://cran.r-project.org/package=cuml4r)
<a href="https://www.r-pkg.org/pkg/cuml4r"><img src="https://cranlogs.r-pkg.org/badges/cuml4r?color=brightgreen" style=""></a>

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




Quick install instructions for Ubuntu 20-04:

Install deps:
```
sudo apt install -y cmake
```


Install CUDA
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
# Add cuda executables to path 
(nvcc is needed for cuml installation)
```bash
echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

Install Miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
# consult https://rapids.ai/start.html for alternatives
```

Create and configure the conda env
```
# This is a relatively big download, may take a while
~/miniconda3/bin/conda create -n rapids-21.08 -c rapidsai -c nvidia -c conda-forge \
    rapids-blazing=21.08 python=3.8 cudatoolkit=11.2
```

Activate the conda env:
```bash
. ~/miniconda3/bin/activate
conda activate rapids-21.08
```


Install cuml the R package:
CRAN release:
```R
Rscript -e 'install.packages("cuml")'
```

Development version from github
```R
Rscript -e 'remotes::install_github("mlverse/cuml")'
```

