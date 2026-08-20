# Train a linear model using LASSO regression.

Train a linear model using LASSO (Least Absolute Shrinkage and Selection
Operator) regression.

## Usage

``` r
cuda_ml_lasso(x, ...)

# Default S3 method
cuda_ml_lasso(x, ...)

# S3 method for class 'data.frame'
cuda_ml_lasso(
  x,
  y,
  alpha = 1,
  max_iter = 1000L,
  tol = 0.001,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
)

# S3 method for class 'matrix'
cuda_ml_lasso(
  x,
  y,
  alpha = 1,
  max_iter = 1000L,
  tol = 0.001,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
)

# S3 method for class 'formula'
cuda_ml_lasso(
  formula,
  data,
  alpha = 1,
  max_iter = 1000L,
  tol = 0.001,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
)

# S3 method for class 'recipe'
cuda_ml_lasso(
  x,
  data,
  alpha = 1,
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

  - A **data frame** of predictors.

  - A **matrix** of predictors.

  - A **recipe** specifying a set of preprocessing steps created from
    [`recipes::recipe()`](https://recipes.tidymodels.org/reference/recipe.html).

  - A **formula** specifying the predictors and the outcome.

- ...:

  Optional arguments; currently unused.

- y:

  A numeric vector (for regression) or factor (for classification) of
  desired responses.

- alpha:

  Positive multiplier of the L1 penalty term. Use
  [`cuda_ml_ols()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_ols.md)
  for an unpenalized linear model. Default: 1.

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

  When a **recipe** or **formula** is used, `data` is specified as a
  **data frame** containing the predictors and (if applicable) the
  outcome.

## Value

A LASSO regressor that can be used with the 'predict' S3 generic to make
predictions on new data points.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  model <- cuda_ml_lasso(formula = mpg ~ ., data = mtcars, alpha = 1e-3)
  predictors <- subset(mtcars, select = -mpg)
  cuda_ml_predictions <- predict(model, predictors)

  # predictions will be comparable to those from a `glmnet` model with
  # `lambda` set to 1e-3 and `alpha` set to 1
  # (in `glmnet`, `lambda` is the weight of the penalty term, and `alpha` is
  #  the elastic mixing parameter between L1 and L2 penalties.

  if (requireNamespace("glmnet", quietly = TRUE)) {
    glmnet_model <- glmnet::glmnet(
      x = as.matrix(predictors), y = mtcars$mpg,
      alpha = 1, lambda = 1e-3, nlambda = 1, standardize = FALSE
    )

    glm_predictions <- predict(
      glmnet_model, as.matrix(predictors),
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
