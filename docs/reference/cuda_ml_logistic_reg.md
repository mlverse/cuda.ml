# Train a logistic or multinomial regression model

Fits a factor outcome with cuML's quasi-Newton solver. Regularization
follows tidymodels conventions: `penalty` is the total regularization
strength and `mixture` is the proportion assigned to the L1 penalty. Set
`mixture = 0` for ridge, `mixture = 1` for lasso, or use an intermediate
value for an elastic-net penalty. Normalize predictors with a recipe
before fitting when scaling is required.

## Usage

``` r
cuda_ml_logistic_reg(x, ...)

# Default S3 method
cuda_ml_logistic_reg(x, ...)

# S3 method for class 'data.frame'
cuda_ml_logistic_reg(
  x,
  y,
  fit_intercept = TRUE,
  penalty = NULL,
  mixture = 0,
  tol = 1e-04,
  class_weight = NULL,
  sample_weight = NULL,
  max_iter = 1000L,
  linesearch_max_iter = 50L,
  lbfgs_memory = 5L,
  penalty_normalized = TRUE,
  ...
)

# S3 method for class 'matrix'
cuda_ml_logistic_reg(
  x,
  y,
  fit_intercept = TRUE,
  penalty = NULL,
  mixture = 0,
  tol = 1e-04,
  class_weight = NULL,
  sample_weight = NULL,
  max_iter = 1000L,
  linesearch_max_iter = 50L,
  lbfgs_memory = 5L,
  penalty_normalized = TRUE,
  ...
)

# S3 method for class 'formula'
cuda_ml_logistic_reg(
  formula,
  data,
  fit_intercept = TRUE,
  penalty = NULL,
  mixture = 0,
  tol = 1e-04,
  class_weight = NULL,
  sample_weight = NULL,
  max_iter = 1000L,
  linesearch_max_iter = 50L,
  lbfgs_memory = 5L,
  penalty_normalized = TRUE,
  ...
)

# S3 method for class 'recipe'
cuda_ml_logistic_reg(
  x,
  data,
  fit_intercept = TRUE,
  penalty = NULL,
  mixture = 0,
  tol = 1e-04,
  class_weight = NULL,
  sample_weight = NULL,
  max_iter = 1000L,
  linesearch_max_iter = 50L,
  lbfgs_memory = 5L,
  penalty_normalized = TRUE,
  ...
)
```

## Arguments

- x:

  Depending on the context:

  \* A \_\_data frame\_\_ of predictors. \* A \_\_matrix\_\_ of
  predictors. \* A \_\_recipe\_\_ specifying a set of preprocessing
  steps \* created from \[recipes::recipe()\]. \* A \_\_formula\_\_
  specifying the predictors and the outcome.

- ...:

  Optional arguments; currently unused.

- y:

  A numeric vector (for regression) or factor (for classification) of
  desired responses.

- fit_intercept:

  If TRUE, then the model tries to correct for the global mean of the
  response variable. If FALSE, then the model expects data to be
  centered. Default: TRUE.

- penalty:

  A non-negative regularization strength, or `NULL` for no
  regularization. Default: `NULL`.

- mixture:

  The proportion of regularization assigned to the L1 penalty, between 0
  and 1. Default: 0.

- tol:

  Stopping tolerance. Default: 1e-4.

- class_weight:

  `NULL`, `"balanced"`, or a named numeric vector with one non-negative
  weight per outcome level.

- sample_weight:

  A numeric vector with one non-negative weight per training
  observation, or `NULL`.

- max_iter:

  Maximum solver iterations. Default: 1000L.

- linesearch_max_iter:

  Maximum line-search iterations per solver iteration. Default: 50L.

- lbfgs_memory:

  Number of vectors retained by the L-BFGS approximation. Default: 5L.

- penalty_normalized:

  Whether to normalize regularization by the number of observations.
  Default: TRUE.

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a \_\_recipe\_\_ or \_\_formula\_\_ is used, `data` is specified
  as a \_\_data frame\_\_ containing the predictors and (if applicable)
  the outcome.

## Value

A classification model for use with
[`predict()`](https://rdrr.io/r/stats/predict.html).
