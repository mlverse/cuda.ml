# Train a SVM model.

Train a Support Vector Machine model for classification or regression
tasks.

## Usage

``` r
cuda_ml_svm(x, ...)

# Default S3 method
cuda_ml_svm(x, ...)

# S3 method for class 'data.frame'
cuda_ml_svm(
  x,
  y,
  cost = 1,
  kernel = c("rbf", "tanh", "polynomial", "linear"),
  gamma = NULL,
  coef0 = 0,
  degree = 3L,
  tol = 0.001,
  max_iter = NULL,
  nochange_steps = 1000L,
  cache_size = 1024,
  epsilon = 0.1,
  sample_weights = NULL,
  ...
)

# S3 method for class 'matrix'
cuda_ml_svm(
  x,
  y,
  cost = 1,
  kernel = c("rbf", "tanh", "polynomial", "linear"),
  gamma = NULL,
  coef0 = 0,
  degree = 3L,
  tol = 0.001,
  max_iter = NULL,
  nochange_steps = 1000L,
  cache_size = 1024,
  epsilon = 0.1,
  sample_weights = NULL,
  ...
)

# S3 method for class 'formula'
cuda_ml_svm(
  formula,
  data,
  cost = 1,
  kernel = c("rbf", "tanh", "polynomial", "linear"),
  gamma = NULL,
  coef0 = 0,
  degree = 3L,
  tol = 0.001,
  max_iter = NULL,
  nochange_steps = 1000L,
  cache_size = 1024,
  epsilon = 0.1,
  sample_weights = NULL,
  ...
)

# S3 method for class 'recipe'
cuda_ml_svm(
  x,
  data,
  cost = 1,
  kernel = c("rbf", "tanh", "polynomial", "linear"),
  gamma = NULL,
  coef0 = 0,
  degree = 3L,
  tol = 0.001,
  max_iter = NULL,
  nochange_steps = 1000L,
  cache_size = 1024,
  epsilon = 0.1,
  sample_weights = NULL,
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

- cost:

  A positive number for the cost of predicting a sample within or on the
  wrong side of the margin. Default: 1.

- kernel:

  Type of the SVM kernel function (must be one of "rbf", "tanh",
  "polynomial", or "linear"). Default: "rbf".

- gamma:

  The gamma coefficient (only relevant to polynomial, RBF, and tanh
  kernel functions, see explanations below). Default: 1 / (num
  features).

  The following kernels are implemented: - RBF K(x_1, x_2) = exp(-gamma
  \|x_1-x_2\|^2) - TANH K(x_1, x_2) = tanh(gamma \<x_1,x_2\> + coef0) -
  POLYNOMIAL K(x_1, x_2) = (gamma \<x_1,x_2\> + coef0)^degree - LINEAR
  K(x_1,x_2) = \<x_1,x_2\>, where \< , \> denotes the dot product.

- coef0:

  The 0th coefficient (only applicable to polynomial and tanh kernel
  functions, see explanations below). Default: 0.

  The following kernels are implemented: - RBF K(x_1, x_2) = exp(-gamma
  \|x_1-x_2\|^2) - TANH K(x_1, x_2) = tanh(gamma \<x_1,x_2\> + coef0) -
  POLYNOMIAL K(x_1, x_2) = (gamma \<x_1,x_2\> + coef0)^degree - LINEAR
  K(x_1,x_2) = \<x_1,x_2\>, where \< , \> denotes the dot product.

- degree:

  Degree of the polynomial kernel function (note: not applicable to
  other kernel types, see explanations below). Default: 3.

  The following kernels are implemented: - RBF K(x_1, x_2) = exp(-gamma
  \|x_1-x_2\|^2) - TANH K(x_1, x_2) = tanh(gamma \<x_1,x_2\> + coef0) -
  POLYNOMIAL K(x_1, x_2) = (gamma \<x_1,x_2\> + coef0)^degree - LINEAR
  K(x_1,x_2) = \<x_1,x_2\>, where \< , \> denotes the dot product.

- tol:

  Tolerance to stop fitting. Default: 1e-3.

- max_iter:

  Maximum number of outer iterations in SmoSolver. Default: 100 \* (num
  samples).

- nochange_steps:

  Number of steps with no change w.r.t convergence. Default: 1000.

- cache_size:

  Size of kernel cache (MiB) in device memory. Default: 1024.

- epsilon:

  Epsilon parameter of the epsilon-SVR model. There is no penalty for
  points that are predicted within the epsilon-tube around the target
  values. Please note this parameter is only relevant for regression
  tasks. Default: 0.1.

- sample_weights:

  Optional weight assigned to each input data point.

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a \_\_recipe\_\_ or \_\_formula\_\_ is used, `data` is specified
  as a \_\_data frame\_\_ containing the predictors and (if applicable)
  the outcome.

## Value

A SVM classifier / regressor object that can be used with the 'predict'
S3 generic to make predictions on new data points.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  # Classification

  model <- cuda_ml_svm(
    formula = Species ~ .,
    data = iris,
    kernel = "rbf"
  )

  predictions <- predict(model, iris[names(iris) != "Species"])

  # Regression

  model <- cuda_ml_svm(
    formula = mpg ~ .,
    data = mtcars,
    kernel = "rbf"
  )

  predictions <- predict(model, mtcars)
}
```
