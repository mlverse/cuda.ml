# Train a linear model using ridge regression.

Train a linear model with L2 regularization.

## Usage

``` r
cuda_ml_ridge(x, ...)

# Default S3 method
cuda_ml_ridge(x, ...)

# S3 method for class 'data.frame'
cuda_ml_ridge(x, y, alpha = 1, fit_intercept = TRUE, ...)

# S3 method for class 'matrix'
cuda_ml_ridge(x, y, alpha = 1, fit_intercept = TRUE, ...)

# S3 method for class 'formula'
cuda_ml_ridge(formula, data, alpha = 1, fit_intercept = TRUE, ...)

# S3 method for class 'recipe'
cuda_ml_ridge(x, data, alpha = 1, fit_intercept = TRUE, ...)
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

  Positive multiplier of the L2 penalty term. Use
  [`cuda_ml_ols()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_ols.md)
  for an unpenalized linear model. Default: 1.

- fit_intercept:

  If TRUE, then the model tries to correct for the global mean of the
  response variable. If FALSE, then the model expects data to be
  centered. Default: TRUE.

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as a
  **data frame** containing the predictors and (if applicable) the
  outcome.

## Value

A ridge regressor that can be used with the 'predict' S3 generic to make
predictions on new data points.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  model <- cuda_ml_ridge(formula = mpg ~ ., data = mtcars, alpha = 1e-3)
  cuda_ml_predictions <- predict(model, mtcars[names(mtcars) != "mpg"])

  # predictions will be comparable to those from a `glmnet` model with
  # `lambda` set to 2e-3 and `alpha` set to 0
  # (in `glmnet`, `lambda` is the weight of the penalty term, and `alpha` is
  #  the elastic mixing parameter between L1 and L2 penalties.

  if (requireNamespace("glmnet", quietly = TRUE)) {
    glmnet_model <- glmnet::glmnet(
      x = as.matrix(mtcars[names(mtcars) != "mpg"]), y = mtcars$mpg,
      alpha = 0, lambda = 2e-3, nlambda = 1, standardize = FALSE
    )

    glmnet_predictions <- predict(
      glmnet_model, as.matrix(mtcars[names(mtcars) != "mpg"]),
      s = 0
    )

    print(
      all.equal(
        as.numeric(glmnet_predictions),
        cuda_ml_predictions$.pred,
        tolerance = 1e-3
      )
    )
  }
}
```
