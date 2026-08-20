# Train a linear model using mini-batch stochastic gradient descent.

Train a linear model using mini-batch stochastic gradient descent.

## Usage

``` r
cuda_ml_sgd(x, ...)

# Default S3 method
cuda_ml_sgd(x, ...)

# S3 method for class 'data.frame'
cuda_ml_sgd(
  x,
  y,
  fit_intercept = TRUE,
  penalty = c("none", "l1", "l2", "elasticnet"),
  alpha = 1e-04,
  l1_ratio = 0.5,
  epochs = 1000L,
  tol = 0.001,
  shuffle = TRUE,
  learning_rate = c("constant", "invscaling", "adaptive"),
  eta0 = 0.001,
  power_t = 0.5,
  batch_size = 32L,
  n_iter_no_change = 5L,
  ...
)

# S3 method for class 'matrix'
cuda_ml_sgd(
  x,
  y,
  fit_intercept = TRUE,
  penalty = c("none", "l1", "l2", "elasticnet"),
  alpha = 1e-04,
  l1_ratio = 0.5,
  epochs = 1000L,
  tol = 0.001,
  shuffle = TRUE,
  learning_rate = c("constant", "invscaling", "adaptive"),
  eta0 = 0.001,
  power_t = 0.5,
  batch_size = 32L,
  n_iter_no_change = 5L,
  ...
)

# S3 method for class 'formula'
cuda_ml_sgd(
  formula,
  data,
  fit_intercept = TRUE,
  penalty = c("none", "l1", "l2", "elasticnet"),
  alpha = 1e-04,
  l1_ratio = 0.5,
  epochs = 1000L,
  tol = 0.001,
  shuffle = TRUE,
  learning_rate = c("constant", "invscaling", "adaptive"),
  eta0 = 0.001,
  power_t = 0.5,
  batch_size = 32L,
  n_iter_no_change = 5L,
  ...
)

# S3 method for class 'recipe'
cuda_ml_sgd(
  x,
  data,
  fit_intercept = TRUE,
  penalty = c("none", "l1", "l2", "elasticnet"),
  alpha = 1e-04,
  l1_ratio = 0.5,
  epochs = 1000L,
  tol = 0.001,
  shuffle = TRUE,
  learning_rate = c("constant", "invscaling", "adaptive"),
  eta0 = 0.001,
  power_t = 0.5,
  batch_size = 32L,
  n_iter_no_change = 5L,
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

- fit_intercept:

  If TRUE, then the model tries to correct for the global mean of the
  response variable. If FALSE, then the model expects data to be
  centered. Default: TRUE.

- penalty:

  Type of regularization to perform, must be one of {"none", "l1", "l2",
  "elasticnet"}.

  - "none": no regularization.

  - "l1": perform regularization based on the L1-norm (LASSO) which
    tries to minimize the sum of the absolute values of the
    coefficients.

  - "l2": perform regularization based on the L2 norm (Ridge) which
    tries to minimize the sum of the square of the coefficients.

  - "elasticnet": perform the Elastic Net regularization which is based
    on the weighted average of L1 and L2 norms.

  Default: "none".

- alpha:

  Multiplier of the penalty term. Default: 1e-4.

- l1_ratio:

  The ElasticNet mixing parameter, with 0 \<= l1_ratio \<= 1. For
  l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an
  L1 penalty. For 0 \< l1_ratio \< 1, the penalty is a combination of L1
  and L2. The penalty term is computed using the following formula:
  penalty = `alpha` \* `l1_ratio` \* \|\|w\|\|\_1 + 0.5 \* `alpha` \*
  (1 - `l1_ratio`) \* \|\|w\|\|^2_2 where \|\|w\|\|\_1 is the L1 norm of
  the coefficients, and \|\|w\|\|\_2 is the L2 norm of the coefficients.

- epochs:

  The number of times the model should iterate through the entire
  dataset during training. Default: 1000L.

- tol:

  Threshold for stopping training. Training will stop if (loss in
  current epoch) \> (loss in previous epoch) - `tol`. Default: 1e-3.

- shuffle:

  Whether to shuffle the training data after each epoch. Default: TRUE.

- learning_rate:

  Must be one of {"constant", "invscaling", "adaptive"}.

  - "constant": the learning rate will be kept constant.

  - "invscaling": (learning rate) = (initial learning rate) / pow(t,
    power_t) where `t` is the number of epochs and `power_t` is a
    tunable parameter of this model.

  - "adaptive": (learning rate) = (initial learning rate) as long as the
    training loss keeps decreasing. Each time the last
    `n_iter_no_change` consecutive epochs fail to decrease the training
    loss by `tol`, the current learning rate is divided by 5.

  Default: "constant".

- eta0:

  The initial learning rate. Default: 1e-3.

- power_t:

  The exponent used for calculating the invscaling learning rate.
  Default: 0.5.

- batch_size:

  The number of samples that will be included in each batch. Default:
  32L.

- n_iter_no_change:

  The maximum number of epochs to train if there is no improvement in
  the model. Default: 5.

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as a
  **data frame** containing the predictors and (if applicable) the
  outcome.

## Value

A linear model that can be used with the 'predict' S3 generic to make
predictions on new data points.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  model <- cuda_ml_sgd(
    mpg ~ ., mtcars,
    batch_size = 4, epochs = 50000,
    learning_rate = "adaptive", eta0 = 1e-5,
    penalty = "l2", alpha = 1e-5, tol = 1e-6,
    n_iter_no_change = 10
  )

  predictors <- subset(mtcars, select = -mpg)
  preds <- predict(model, predictors)
  print(all.equal(preds$.pred, mtcars$mpg, tolerance = 0.09))
}
```
