# Truncated SVD.

Dimensionality reduction using Truncated Singular Value Decomposition.

## Usage

``` r
cuda_ml_tsvd(
  x,
  n_components = 2L,
  eig_algo = c("dq", "jacobi"),
  tol = 1e-07,
  n_iters = 15L,
  transform_input = TRUE
)
```

## Arguments

- x:

  The input matrix or data frame. Each data point should be a row and
  should consist of numeric values only.

- n_components:

  Desired dimensionality of output data. Must be strictly less than
  `ncol(x)` (i.e., the number of features in input data). Default: 2.

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

- transform_input:

  If TRUE, then compute an approximate representation of the input data.
  Default: TRUE.

## Value

A TSVD model object with the following attributes:

- "components": a matrix of `n_components` rows to be used for
  dimensionality reduction on new data points.

- "explained_variance": (only present if "transform_input" is set to
  TRUE) amount of variance within the input data explained by each
  component.

- "explained_variance_ratio": (only present if "transform_input" is set
  to TRUE) fraction of variance within the input data explained by each
  component.

- "singular_values": The singular values corresponding to each
  component. The singular values are equal to the 2-norms of the
  `n_components` variables in the lower-dimensional space.

- "tsvd_params": opaque pointer to TSVD parameters which will be used
  for performing inverse transforms.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  oils <- modeldata::oils
  oil_predictors <- oils |>
    subset(select = -class) |>
    scale()

  oil_tsvd <- cuda_ml_tsvd(oil_predictors, n_components = 2)
  print(oil_tsvd)
}
```
