
<!-- README.md is generated from README.Rmd. Please edit that file -->
# cuda.ml

<!-- badges: start -->
[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/cuda.ml)](https://cran.r-project.org/package=cuda.ml) <a href="https://www.r-pkg.org/pkg/cuda.ml"><img src="https://cranlogs.r-pkg.org/badges/cuda.ml?color=brightgreen" style=""></a> <!-- badges: end -->

The goal of {cuda.ml} is to provide a simple and intuitive R interface for [RAPIDS cuML](https://github.com/rapidsai/cuml). RAPIDS cuML is a suite of GPU-accelerated machine learning libraries powered by [CUDA](https://en.wikipedia.org/wiki/CUDA). {cuda.ml} is under active development, and currently implements R interfaces for the algorithms listed below (which is a subset of [algorithms supported by RAPIDS cuML](https://github.com/rapidsai/cuml#supported-algorithms)).

### Supported Algorithms

<table style="width:17%;">
<colgroup>
<col width="5%" />
<col width="5%" />
<col width="5%" />
</colgroup>
<thead>
<tr class="header">
<th>Category</th>
<th>Algorithm</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><strong>Clustering</strong></td>
<td>Density-Based Spatial Clustering of Applications with Noise (DBSCAN)</td>
<td>Only single-GPU implementation is supported at the moment</td>
</tr>
<tr class="even">
<td></td>
<td>K-Means</td>
<td>Only single-GPU implementation is supported at the moment</td>
</tr>
<tr class="odd">
<td></td>
<td>Single-Linkage Agglomerative Clustering</td>
<td></td>
</tr>
<tr class="even">
<td><strong>Dimensionality Reduction</strong></td>
<td>Principal Components Analysis (PCA)</td>
<td>Only single-GPU implementation is supported at the moment</td>
</tr>
<tr class="odd">
<td></td>
<td>Truncated Singular Value Decomposition (tSVD)</td>
<td>Only single-GPU implementation is supported at the moment</td>
</tr>
<tr class="even">
<td></td>
<td>Uniform Manifold Approximation and Projection (UMAP)</td>
<td>Only single-GPU implementation is supported at the moment</td>
</tr>
<tr class="odd">
<td></td>
<td>Random Projection</td>
<td></td>
</tr>
<tr class="even">
<td></td>
<td>t-Distributed Stochastic Neighbor Embedding (TSNE)</td>
<td></td>
</tr>
<tr class="odd">
<td><strong>Linear Models for Regression or Classification</strong></td>
<td>Linear Regression (OLS)</td>
<td></td>
</tr>
<tr class="even">
<td></td>
<td>Linear Regression with Lasso or Ridge Regularization</td>
<td></td>
</tr>
<tr class="odd">
<td></td>
<td>Logistic Regression</td>
<td></td>
</tr>
<tr class="even">
<td><strong>Nonlinear Models for Regression or Classification</strong></td>
<td>Random Forest (RF) Classification</td>
<td>Only single-GPU implementation is supported at the moment</td>
</tr>
<tr class="odd">
<td></td>
<td>Random Forest (RF) Regression</td>
<td>Only single-GPU implementation is supported at the moment</td>
</tr>
<tr class="even">
<td></td>
<td>Inference for decision tree-based models in XGBoost or LightGBM formats using the CuML Forest Inference Library (FIL)</td>
<td>Requires linkage to the Treelite C library when {cuml} is installed. Treelite is used for model loading.</td>
</tr>
<tr class="odd">
<td></td>
<td>K-Nearest Neighbors (KNN) Classification</td>
<td>Uses <a href="https://github.com/facebookresearch/faiss">Faiss</a> for Nearest Neighbors Query.</td>
</tr>
<tr class="even">
<td></td>
<td>K-Nearest Neighbors (KNN) Regression</td>
<td>Uses <a href="https://github.com/facebookresearch/faiss">Faiss</a> for Nearest Neighbors Query.</td>
</tr>
<tr class="odd">
<td></td>
<td>Support Vector Machine Classifier (SVC)</td>
<td></td>
</tr>
<tr class="even">
<td></td>
<td>Epsilon-Support Vector Regression (SVR)</td>
<td></td>
</tr>
</tbody>
</table>

# Examples

## Using {cuda.ml} for supervised ML tasks through {parsnip}

{cuda.ml} provides {parsnip} bindings for supervised ML algorithms such as `rand_forest`, `nearest_neighbor`, `svm_rbf`, `svm_poly`, and `svm_linear`.

The following example shows how {cuda.ml} can be used as a {parsnip} engine to build a SVM classifier.

``` r
library(dplyr, warn.conflicts = FALSE)
library(parsnip)
library(cuda.ml)
set.seed(11235)

train_inds <- iris %>%
  mutate(ind = row_number()) %>%
  group_by(Species) %>%
  slice_sample(prop = 0.7)

train_data <- iris[train_inds$ind, ]
test_data <- iris[-train_inds$ind, ]

model <- svm_rbf(mode = "classification", rbf_sigma = 10, cost = 50) %>%
  set_engine("cuda.ml") %>%
  fit(Species ~ ., data = train_data)

preds <- predict(model, test_data)

cat("Confusion matrix:\n\n")
#> Confusion matrix:
preds %>%
  bind_cols(test_data %>% select(Species)) %>%
  yardstick::conf_mat(truth = Species, estimate = .pred_class)
#>             Truth
#> Prediction   setosa versicolor virginica
#>   setosa         15          0         0
#>   versicolor      0         12         1
#>   virginica       0          3        14
```

## Using {cuda.ml} for unsupervised ML tasks

The following example shows how {cuda.ml} can be used for unsupervised ML tasks such as k-means clustering.

``` r
library(cuda.ml)

clustering <- cuda_ml_kmeans(
  iris[, which(names(iris) != "Species")],
  k = 3, max_iters = 100
)

# Expected outcome: there is strong correlation
# between cluster labels and `iris$Species`
print(clustering)
#> $labels
#>   [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#>  [38] 1 1 1 1 1 1 1 1 1 1 1 1 1 0 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#>  [75] 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 0 0 0 0 2 0 0 0 0
#> [112] 0 0 2 2 0 0 0 0 2 0 2 0 2 0 0 2 2 0 0 0 0 0 2 0 0 0 0 2 0 0 0 2 0 0 0 2 0
#> [149] 0 2
#> 
#> $centroids
#>          [,1]     [,2]     [,3]     [,4]
#> [1,] 6.853846 3.076923 5.715385 2.053846
#> [2,] 5.006000 3.428000 1.462000 0.246000
#> [3,] 5.883607 2.740984 4.388525 1.434426
#> 
#> $inertia
#> [1] 78.85567
#> 
#> $n_iter
#> [1] 10

library(dplyr, warn.conflicts = FALSE)
tibble(cluster_id = clustering$labels, species = iris$Species) %>%
  group_by(cluster_id) %>% count(species)
#> # A tibble: 5 × 3
#> # Groups:   cluster_id [3]
#>   cluster_id species        n
#>        <int> <fct>      <int>
#> 1          0 versicolor     3
#> 2          0 virginica     36
#> 3          1 setosa        50
#> 4          2 versicolor    47
#> 5          2 virginica     14
```

## Using {cuda.ml} for visualizations

{cuda.ml} also features R interfaces for algorithms such as UMAP and t-SNE, which are useful when one needs to visualize clusters of high-dimensional data points by embedding them onto low-dimensional manifolds (i.e., 4 dimensions or fewer).

For example, the code snippet below shows how `cuda_ml_umap()` can be used to visualize the MNIST hand-written digits dataset, and also, the coloring based on the true label of each sample demonstrates how well the UMAP algorithm transforms different hand writings of the same digit into nearby points in a 2D embedding:

``` r
library(cuda.ml)
library(ggplot2)
library(magrittr)

# load mnist
source("data-raw/load-mnist.R")
str(mnist_images)
#>  int [1:28, 1:28, 1:60000] 0 0 0 0 0 0 0 0 0 0 ...
str(mnist_labels)
#>  int [1:60000(1d)] 5 0 4 1 9 2 1 3 1 4 ...


# flatten each image to a 1d array, combine into a matrix with 1 row per image
flatten <- function(img) {
  dim(img) <- NULL
  img
}

flattened_mnist_images <-
  mnist_images %>% asplit(3) %>% lapply(flatten) %>% do.call(rbind, .)

# embed
embedding <- cuda_ml_umap(
  flattened_mnist_images, n_components = 2, n_neighbors = 50,
  local_connectivity = 15, repulsion_strength = 10
)

str(embedding$transformed_data)
#>  num [1:60000, 1:2] -5.81 -31.26 11.1 7.35 11.87 ...

# visualize
embedding$transformed_data %>%
  as.data.frame() %>%
  dplyr::mutate(Label = factor(mnist_labels)) %>%
  ggplot(aes(x = V1, y = V2, color = Label)) +
  geom_point(alpha = .5, size = .5) +
  labs(title = "UMAP: Uniform Manifold Approximation and Projection",
       subtitle = "Two Dimensional Embedding of MNIST")
```

<img src="man/figures/README-umap-example-1.png" width="100%" />

From this type of visualization, we can qualitatively understand the following about the MNIST dataset:

-   The dataset can be reasonably classified into some number of categories.
-   The right number of categories may be any where between 9 and 11.
-   While there are some categories that are clearly distinguishable from others, there are others that have less clear boundaries with their neighbors.
-   A small fraction of data points did not fit particularly well into any of the categories.
-   Most data points belonging to the same digit category are clustered together in the UMAP output

## Installation

For a fully functional installation, {cuda.ml} needs:

-   an NVIDIA GPU with a working NVIDIA driver;
-   a CUDA Toolkit installation that provides `nvcc`;
-   normal R package build tools; and
-   either `uv` or Python with `pip`.

When those prerequisites are present, {cuda.ml} can bootstrap RAPIDS cuML from pip wheels during installation. You do not need conda, and you usually do not need to set `CUML_PREFIX` manually.

On a new Ubuntu installation, install R/build/Python prerequisites:

``` bash
sudo apt update
sudo apt install -y r-base-dev build-essential git cmake \
  python3 python3-pip python3-venv ubuntu-drivers-common
```

Install the NVIDIA driver, reboot, and verify that the driver can see your GPU:

``` bash
sudo ubuntu-drivers install
sudo reboot

nvidia-smi
```

Install a CUDA Toolkit that includes `nvcc`. Use NVIDIA's CUDA Linux installation guide for your Ubuntu release to add the CUDA apt repository, then:

``` bash
sudo apt update
sudo apt install -y cuda-toolkit

nvcc --version
```

If the toolkit is installed but `nvcc` is not on `PATH`, set `CUDA_HOME` to the toolkit prefix before installing {cuda.ml}, for example:

``` bash
export CUDA_HOME=/usr/local/cuda
```

Then install {cuda.ml}:

``` r
install.packages("cuda.ml")
```

And verify that the installed package was linked with real cuML:

``` r
library(cuda.ml)
has_cuML()
```

If this returns `TRUE`, {cuda.ml} is using RAPIDS cuML. If it returns `FALSE`, the package installed in stub-only mode; check the install output for the first missing prerequisite.

### What happens during installation

The configure script first looks for an existing RAPIDS installation through `CUML_PREFIX` or `CUDA_PATH`. If no existing installation is found, and a working NVIDIA driver/GPU plus `nvcc` are available, it bootstraps RAPIDS cuML from pip wheels into a cache directory and links {cuda.ml} against that prefix.

The bootstrap prefers `uv` when available, then reticulate's managed `uv`, then `python -m pip`, `python3 -m pip`, `pip`, and `pip3`.

Useful environment variables:

-   `CUDA_HOME`: CUDA Toolkit prefix containing `bin/nvcc`.
-   `CUML_PREFIX`: existing RAPIDS prefix containing `include/cuml` and `lib/libcuml++.so`.
-   `CUML_BOOTSTRAP=0`: disable automatic RAPIDS pip bootstrap.
-   `CUML_BOOTSTRAP_CACHE`: cache directory for bootstrapped RAPIDS headers and libraries.
-   `CUML_PIP_VERSION`: RAPIDS pip wheel version to install.

### CRAN and machines without GPUs

On CRAN, or on machines without a usable NVIDIA GPU/driver and `nvcc`, {cuda.ml} can still install in stub-only mode. In that mode `has_cuML()` returns `FALSE` and cuML-backed algorithms are not usable until the system prerequisites are installed and {cuda.ml} is reinstalled.

### Manual RAPIDS installations

If you already have RAPIDS cuML from pip, conda, or a source build, set `CUML_PREFIX` to a prefix containing `include/cuml` and `lib/libcuml++.so` before installing {cuda.ml}. In this case the automatic bootstrap is skipped.

### Development version

Install the development version from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("mlverse/cuda.ml")
```

To speed up recompilation times during development, set this env var:

``` bash
echo "export CUML4R_ENABLE_CCACHE=1" >> ~/.bashrc
. ~/.bashrc
```

## Appendix

<details> <summary>Inspect MNIST images</summary>

``` r
plot_mnist(1:64)
```

<img src="man/figures/README-mnist-1.png" width="100%" /> </details>
