# Train an OLS model.

Train an ordinary least squares (OLS) model for regression tasks.

## Usage

``` r
cuda_ml_ols(x, ...)

# Default S3 method
cuda_ml_ols(x, ...)

# S3 method for class 'data.frame'
cuda_ml_ols(x, y, method = c("svd", "eig", "qr"), fit_intercept = TRUE, ...)

# S3 method for class 'matrix'
cuda_ml_ols(x, y, method = c("svd", "eig", "qr"), fit_intercept = TRUE, ...)

# S3 method for class 'formula'
cuda_ml_ols(
  formula,
  data,
  method = c("svd", "eig", "qr"),
  fit_intercept = TRUE,
  ...
)

# S3 method for class 'recipe'
cuda_ml_ols(x, data, method = c("svd", "eig", "qr"), fit_intercept = TRUE, ...)
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

- method:

  Must be one of {"svd", "eig", "qr"}.

  - "svd": compute SVD decomposition using Jacobi iterations.

  - "eig": use an eigendecomposition of the covariance matrix.

  - "qr": use the QR decomposition algorithm and solve `Rx = Q^T y`.

  If the number of features is larger than the sample size, then the
  "svd" algorithm will be force-selected because it is the only
  algorithm that can support this type of scenario.

  Default: "svd".

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

An OLS regressor that can be used with the 'predict' S3 generic to make
predictions on new data points.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  model <- cuda_ml_ols(formula = mpg ~ ., data = mtcars, method = "qr")
  predictors <- subset(mtcars, select = -mpg)
  predictions <- predict(model, predictors)

  # predictions will be comparable to those from a `stats::lm` model
  lm_model <- stats::lm(formula = mpg ~ ., data = mtcars, method = "qr")
  lm_predictions <- predict(lm_model, predictors)

  print(
    all.equal(
      as.numeric(lm_predictions),
      predictions$.pred,
      tolerance = 1e-3
    )
  )
}
```
