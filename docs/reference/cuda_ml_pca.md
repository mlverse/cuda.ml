# Perform principal component analysis.

Compute principal component(s) of the input data. Each feature from the
input will be mean-centered (but not scaled) before the SVD computation
takes place.

## Usage

``` r
cuda_ml_pca(
  x,
  n_components = NULL,
  eig_algo = c("dq", "jacobi"),
  tol = 1e-07,
  n_iters = 15L,
  whiten = FALSE,
  transform_input = TRUE
)
```

## Arguments

- x:

  The input matrix or dataframe. Each data point should be a row and
  should consist of numeric values only.

- n_components:

  Number of principal component(s) to keep. Default: min(nrow(x),
  ncol(x)).

- eig_algo:

  Eigen decomposition algorithm to be applied to the covariance matrix.
  Valid choices are "dq" (divid-and-conquer method for symmetric
  matrices) and "jacobi" (the Jacobi method for symmetric matrices).
  Default: "dq".

- tol:

  Tolerance for singular values computed by the Jacobi method. Default:
  1e-7.

- n_iters:

  Maximum number of iterations for the Jacobi method. Default: 15.

- whiten:

  If TRUE, then de-correlate all components, making each component have
  unit variance and removing multi-collinearity. Default: FALSE.

- transform_input:

  If TRUE, then compute an approximate representation of the input data.
  Default: TRUE.

## Value

A PCA model object with the following attributes: - "components": a
matrix of `n_components` rows containing the top principal components. -
"explained_variance": amount of variance within the input data explained
by each component. - "explained_variance_ratio": fraction of variance
within the input data explained by each component. - "singular_values":
singular values (non-negative) corresponding to the top principal
components. - "mean": the column-wise mean of `x` which was used to
mean-center `x` first. - "transformed_data": (only present if
"transform_input" is set to TRUE) an approximate representation of input
data based on principal components. - "pca_params": opaque pointer to
PCA parameters which will be used for performing inverse transforms.

The model object can be used as input to the
[`cuda_ml_inverse_transform()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_inverse_transform.md)
function to map a representation based on principal components back to
the original feature space.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  iris.pca <- cuda_ml_pca(iris[1:4], n_components = 3)
  print(iris.pca)
}
```
