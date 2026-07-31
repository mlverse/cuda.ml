# Train a regularized linear regression model

This is the tidymodels-style linear regression interface. It dispatches
to
[`cuda_ml_ols()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_ols.md),
[`cuda_ml_ridge()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_ridge.md),
[`cuda_ml_lasso()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_lasso.md),
or
[`cuda_ml_elastic_net()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_elastic_net.md)
according to `penalty` and `mixture`. The named model functions remain
available when direct control over their solver arguments is needed.

## Usage

``` r
cuda_ml_linear_reg(formula, data, penalty = NULL, mixture = NULL, ...)
```

## Arguments

- formula:

  A model formula.

- data:

  A data frame containing predictors and outcome.

- penalty:

  A non-negative regularization strength, or `NULL` for no
  regularization.

- mixture:

  The proportion of regularization assigned to the L1 penalty, between 0
  and 1. When `NULL`, a lasso penalty is used.

- ...:

  Arguments passed to the selected named model function.

## Value

A fitted cuda.ml linear model.
