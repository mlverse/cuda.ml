# Train a linear model using elastic regression.

Train a linear model with combined L1 and L2 priors as the regularizer.

## Usage

``` r
cuda_ml_elastic_net(x, ...)

# Default S3 method
cuda_ml_elastic_net(x, ...)

# S3 method for class 'data.frame'
cuda_ml_elastic_net(
  x,
  y,
  alpha = 1,
  l1_ratio = 0.5,
  max_iter = 1000L,
  tol = 0.001,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
)

# S3 method for class 'matrix'
cuda_ml_elastic_net(
  x,
  y,
  alpha = 1,
  l1_ratio = 0.5,
  max_iter = 1000L,
  tol = 0.001,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
)

# S3 method for class 'formula'
cuda_ml_elastic_net(
  formula,
  data,
  alpha = 1,
  l1_ratio = 0.5,
  max_iter = 1000L,
  tol = 0.001,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
)

# S3 method for class 'recipe'
cuda_ml_elastic_net(
  x,
  data,
  alpha = 1,
  l1_ratio = 0.5,
  max_iter = 1000L,
  tol = 0.001,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
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

- alpha:

  Positive multiplier of the penalty term. Use
  [`cuda_ml_ols()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_ols.md)
  for an unpenalized linear model. Default: 1.

- l1_ratio:

  The ElasticNet mixing parameter, with 0 \<= l1_ratio \<= 1. For
  l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an
  L1 penalty. For 0 \< l1_ratio \< 1, the penalty is a combination of L1
  and L2. The penalty term is computed using the following formula:
  penalty = `alpha` \* `l1_ratio` \* \|\|w\|\|\_1 + 0.5 \* `alpha` \*
  (1 - `l1_ratio`) \* \|\|w\|\|^2_2 where \|\|w\|\|\_1 is the L1 norm of
  the coefficients, and \|\|w\|\|\_2 is the L2 norm of the coefficients.

- max_iter:

  The maximum number of coordinate descent iterations. Default: 1000L.

- tol:

  Stop the coordinate descent when the duality gap is below this
  threshold. Default: 1e-3.

- fit_intercept:

  If TRUE, then the model tries to correct for the global mean of the
  response variable. If FALSE, then the model expects data to be
  centered. Default: TRUE.

- selection:

  If "random", then instead of updating coefficients in cyclic order, a
  random coefficient is updated in each iteration. Default: "cyclic".

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a \_\_recipe\_\_ or \_\_formula\_\_ is used, `data` is specified
  as a \_\_data frame\_\_ containing the predictors and (if applicable)
  the outcome.

## Value

An elastic net regressor that can be used with the 'predict' S3 generic
to make predictions on new data points.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  model <- cuda_ml_elastic_net(
    formula = mpg ~ ., data = mtcars, alpha = 1e-3, l1_ratio = 0.6
  )
  cuda_ml_predictions <- predict(model, mtcars)

  # predictions will be comparable to those from a `glmnet` model with
  # `lambda` set to 1e-3 and `alpha` set to 0.6
  # (in `glmnet`, `lambda` is the weight of the penalty term, and `alpha` is
  #  the elastic mixing parameter between L1 and L2 penalties.

  if (requireNamespace("glmnet", quietly = TRUE)) {
    glmnet_model <- glmnet::glmnet(
      x = as.matrix(mtcars[names(mtcars) != "mpg"]), y = mtcars$mpg,
      alpha = 0.6, lambda = 1e-3, nlambda = 1, standardize = FALSE
    )

    glm_predictions <- predict(
      glmnet_model, as.matrix(mtcars[names(mtcars) != "mpg"]),
      s = 0
    )

    print(
      all.equal(
        as.numeric(glm_predictions),
        cuda_ml_predictions$.pred,
        tolerance = 1e-2
      )
    )
  }
}
```
