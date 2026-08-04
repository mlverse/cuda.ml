# Use cuda.ml with tidymodels

cuda.ml registers parsnip engines for its supervised models. This gives
the models the same specification, fitting, and prediction interface as
other parsnip engines while training with cuML. Use the direct cuda.ml
API when you need an algorithm that has no parsnip specification or
detailed control over a solver.

The examples are not evaluated when this vignette is built. Install the
complete backend before running them. See [Getting
started](https://mlverse.github.io/cuda.ml/articles/cuda-ml.md) for
installation and runtime setup.

## Available model specifications

Loading cuda.ml registers the engine name `"cuda.ml"` for these parsnip
specifications:

| parsnip specification                                                                  | mode           | prediction types    |
|:---------------------------------------------------------------------------------------|:---------------|:--------------------|
| [`linear_reg()`](https://parsnip.tidymodels.org/reference/linear_reg.html)             | regression     | `"numeric"`         |
| [`logistic_reg()`](https://parsnip.tidymodels.org/reference/logistic_reg.html)         | classification | `"class"`, `"prob"` |
| [`multinom_reg()`](https://parsnip.tidymodels.org/reference/multinom_reg.html)         | classification | `"class"`, `"prob"` |
| [`rand_forest()`](https://parsnip.tidymodels.org/reference/rand_forest.html)           | classification | `"class"`, `"prob"` |
| [`rand_forest()`](https://parsnip.tidymodels.org/reference/rand_forest.html)           | regression     | `"numeric"`         |
| [`nearest_neighbor()`](https://parsnip.tidymodels.org/reference/nearest_neighbor.html) | classification | `"class"`, `"prob"` |
| [`nearest_neighbor()`](https://parsnip.tidymodels.org/reference/nearest_neighbor.html) | regression     | `"numeric"`         |
| [`svm_rbf()`](https://parsnip.tidymodels.org/reference/svm_rbf.html)                   | classification | `"class"`           |
| [`svm_rbf()`](https://parsnip.tidymodels.org/reference/svm_rbf.html)                   | regression     | `"numeric"`         |
| [`svm_poly()`](https://parsnip.tidymodels.org/reference/svm_poly.html)                 | classification | `"class"`           |
| [`svm_poly()`](https://parsnip.tidymodels.org/reference/svm_poly.html)                 | regression     | `"numeric"`         |
| [`svm_linear()`](https://parsnip.tidymodels.org/reference/svm_linear.html)             | classification | `"class"`           |
| [`svm_linear()`](https://parsnip.tidymodels.org/reference/svm_linear.html)             | regression     | `"numeric"`         |

The SVM engines do not register probability predictions. In particular,
`predict(fitted_svm, new_data, type = "prob")` is not supported. Use
[`logistic_reg()`](https://parsnip.tidymodels.org/reference/logistic_reg.html),
[`multinom_reg()`](https://parsnip.tidymodels.org/reference/multinom_reg.html),
[`rand_forest()`](https://parsnip.tidymodels.org/reference/rand_forest.html),
or
[`nearest_neighbor()`](https://parsnip.tidymodels.org/reference/nearest_neighbor.html)
when a classification workflow requires probabilities.

Use
[`logistic_reg()`](https://parsnip.tidymodels.org/reference/logistic_reg.html)
for a two-level outcome and
[`multinom_reg()`](https://parsnip.tidymodels.org/reference/multinom_reg.html)
for an outcome with more than two levels. Both specifications call
[`cuda_ml_logistic_reg()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_logistic_reg.md);
the outcome determines whether cuda.ml uses its binary or multinomial
loss.

## Set the engine

Create a parsnip specification, set its mode when the specification
supports more than one, and select the cuda.ml engine:

``` r
library(cuda.ml)
library(parsnip)

forest_spec <- rand_forest(
  mtry = 2,
  trees = 500,
  min_n = 5
) |>
  set_mode("classification") |>
  set_engine(
    "cuda.ml",
    max_depth = 20L,
    n_bins = 256L,
    seed = 1L
  )

forest_fit <- fit(forest_spec, Species ~ ., data = iris)

class_predictions <- predict(forest_fit, iris, type = "class")
probabilities <- predict(forest_fit, iris, type = "prob")
```

Arguments in the model specification, such as `mtry`, are common parsnip
arguments. Arguments in
[`set_engine()`](https://parsnip.tidymodels.org/reference/set_engine.html),
such as `max_depth`, are specific to cuda.ml. Put each argument in only
one place.

## Preprocess with a recipe

cuda.ml’s supervised models require numeric predictors. Scaling is
especially important for KNN and SVM models because their fits depend on
distances or margins. It is also usually appropriate for penalized
linear and logistic models. Fit preprocessing parameters on the training
data only, then apply the same recipe to assessment or production data.

This example normalizes the predictors before fitting an exact KNN
classifier. It uses
[`prep()`](https://recipes.tidymodels.org/reference/prep.html) and
[`bake()`](https://recipes.tidymodels.org/reference/bake.html)
explicitly so the boundary between preprocessing and GPU model fitting
is visible.

``` r
library(cuda.ml)
library(parsnip)
library(recipes)

set.seed(1)
training_rows <- sample(seq_len(nrow(iris)), 120)
iris_train <- iris[training_rows, ]
iris_test <- iris[-training_rows, ]

iris_recipe <- recipe(Species ~ ., data = iris_train) |>
  step_normalize(all_numeric_predictors())

iris_recipe <- prep(iris_recipe, training = iris_train)
train_processed <- bake(iris_recipe, new_data = NULL)
test_processed <- bake(iris_recipe, new_data = iris_test)

knn_spec <- nearest_neighbor(neighbors = 5, dist_power = 2) |>
  set_mode("classification") |>
  set_engine(
    "cuda.ml",
    algo = "brute",
    metric = "euclidean"
  )

knn_fit <- fit(knn_spec, Species ~ ., data = train_processed)
test_predictors <- test_processed[names(test_processed) != "Species"]

results <- cbind(
  truth = test_processed$Species,
  predict(knn_fit, test_predictors, type = "class"),
  predict(knn_fit, test_predictors, type = "prob")
)
results
```

The parsnip KNN engine defaults to `algo = "ivfflat"` and
`metric = "euclidean"`. `ivfflat` performs approximate neighbor search.
Set `algo = "brute"`, as above, when exact search is required. The
direct
[`cuda_ml_knn()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_knn.md)
interface defaults to brute-force search. The cuda.ml engine does not
map parsnip’s `weight_func` argument; leave it as `NULL`.

Tree models do not generally need normalization. A recipe can still be
useful for creating numeric indicators or applying other preprocessing
learned from the training set.

## Tune common arguments

cuda.ml registers standard dials parameter metadata for the following
parsnip arguments:

| specification                                                                                                                                                  | registered arguments                       |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|
| [`linear_reg()`](https://parsnip.tidymodels.org/reference/linear_reg.html)                                                                                     | `penalty`, `mixture`                       |
| [`logistic_reg()`](https://parsnip.tidymodels.org/reference/logistic_reg.html), [`multinom_reg()`](https://parsnip.tidymodels.org/reference/multinom_reg.html) | `penalty`, `mixture`                       |
| [`rand_forest()`](https://parsnip.tidymodels.org/reference/rand_forest.html)                                                                                   | `mtry`, `trees`, `min_n`                   |
| [`nearest_neighbor()`](https://parsnip.tidymodels.org/reference/nearest_neighbor.html)                                                                         | `neighbors`, `dist_power`                  |
| [`svm_rbf()`](https://parsnip.tidymodels.org/reference/svm_rbf.html)                                                                                           | `cost`, `margin`, `rbf_sigma`              |
| [`svm_poly()`](https://parsnip.tidymodels.org/reference/svm_poly.html)                                                                                         | `cost`, `margin`, `degree`, `scale_factor` |
| [`svm_linear()`](https://parsnip.tidymodels.org/reference/svm_linear.html)                                                                                     | `cost`, `margin`                           |

These arguments can use
[`tune::tune()`](https://hardhat.tidymodels.org/reference/tune.html) in
a tuning workflow. `margin` maps to the epsilon tube and affects SVM
regression only; do not tune it for an SVM classifier.

Engine arguments are not registered as dials parameters. Keep them fixed
in
[`set_engine()`](https://parsnip.tidymodels.org/reference/set_engine.html)
unless you define an explicit dials parameter and range for the tuning
workflow. Useful engine arguments include:

| engine                              | examples of engine-specific arguments                                                                                                                                      |
|:------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| linear regression                   | `fit_intercept`; route-specific options described below                                                                                                                    |
| logistic and multinomial regression | `fit_intercept`, `tol`, `class_weight`, `max_iter`, `linesearch_max_iter`, `lbfgs_memory`, `penalty_normalized`                                                            |
| random forest                       | `bootstrap`, `sample_fraction`, `max_depth`, `max_leaves`, `n_bins`, `min_samples_leaf`, `split_criterion`, `min_impurity_decrease`, `max_batch_size`, `n_streams`, `seed` |
| nearest neighbor                    | `algo`, `metric`                                                                                                                                                           |
| SVM                                 | `coef0`, `tol`, `max_iter`, `nochange_steps`, `cache_size`, `sample_weights`                                                                                               |

Consult the corresponding `cuda_ml_*()` reference page before setting
these arguments. For example, random-forest split criteria differ
between classification and regression, and approximate KNN algorithms
support fewer distance metrics than brute-force KNN.

### Linear regression routing

The
[`linear_reg()`](https://parsnip.tidymodels.org/reference/linear_reg.html)
engine selects a cuda.ml solver from `penalty` and `mixture`:

| values                                      | direct function                                                                               |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------|
| `penalty = NULL` or `penalty = 0`           | [`cuda_ml_ols()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_ols.md)                 |
| positive `penalty`, `mixture = 0`           | [`cuda_ml_ridge()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_ridge.md)             |
| positive `penalty`, `mixture = 1` or `NULL` | [`cuda_ml_lasso()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_lasso.md)             |
| positive `penalty`, `0 < mixture < 1`       | [`cuda_ml_elastic_net()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_elastic_net.md) |

Only pass options supported by the selected function. For example,
`method` is an OLS option, while `max_iter`, `tol`, and `selection`
apply to the lasso and elastic-net routes. If the solver itself is part
of the decision, use the named direct functions so that the relationship
between the function and its arguments remains explicit.

## Choose between parsnip and the direct API

Use parsnip when you want to compare engines through common
specifications, use a tidymodels tuning workflow, or consume standard
parsnip prediction types. Parsnip delegates training and prediction to
cuda.ml’s public model functions.

Use the direct API when you need:

- an unsupervised or transformation algorithm such as PCA, UMAP,
  clustering, or t-SNE;
- a supervised algorithm without a matching parsnip specification;
- direct matrix, data-frame, formula, or recipe methods; or
- solver controls that do not fit cleanly into a portable model
  specification.

For example, cuda.ml’s direct SVM interface supports a `"tanh"` kernel,
but parsnip registration is limited to the RBF, polynomial, and linear
SVM specifications.

``` r
direct_fit <- cuda_ml_svm(
  Species ~ .,
  data = iris,
  kernel = "tanh",
  cost = 2,
  gamma = 0.1,
  coef0 = 0
)

direct_predictions <- predict(
  direct_fit,
  iris[names(iris) != "Species"]
)
```

See [Save and restore
models](https://mlverse.github.io/cuda.ml/articles/model-persistence.md)
before moving a fitted model to another R process or deployment host.
