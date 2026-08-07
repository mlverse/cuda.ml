skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("Random Forest")

test_that("random forest classifier works as expected", {
  cuda_ml_rf_model <- cuda_ml_rand_forest(
    formula = Species ~ .,
    data = iris,
    trees = 200,
    bootstrap = FALSE,
    n_streams = 12L
  )
  sklearn_rf_model <- sklearn$ensemble$RandomForestClassifier(
    n_estimators = 200L,
    bootstrap = FALSE
  )
  sklearn_rf_model$fit(
    X = as.matrix(iris[which(names(iris) != "Species")]),
    y = as.integer(iris$Species)
  )

  cuda_ml_preds <- predict(
    cuda_ml_rf_model,
    iris[which(names(iris) != "Species")]
  )
  sklearn_preds <- sklearn_rf_model$predict(
    as.matrix(iris[which(names(iris) != "Species")])
  )

  expect_equal(
    as.integer(cuda_ml_preds$.pred_class),
    as.integer(sklearn_preds)
  )
})

test_that("random forest regressor works as expected", {
  cuda_ml_rf_model <- cuda_ml_rand_forest(
    formula = mpg ~ .,
    data = mtcars,
    trees = 100,
    bootstrap = FALSE,
    n_streams = 12L
  )
  cuda_ml_preds <- predict(
    cuda_ml_rf_model,
    mtcars[which(names(mtcars) != "mpg")]
  )

  expect_equal(cuda_ml_preds$.pred, mtcars$mpg, tolerance = 0.2)
})

test_that("random forest classifier returns R probability columns", {
  model <- cuda_ml_rand_forest(Species ~ ., iris, trees = 100L)

  probabilities <- predict(model, iris, type = "prob")
  per_tree <- cuda_ml_nvforest_predict_per_tree(model, iris)
  info <- cuda_ml_nvforest_info(model)

  expect_named(probabilities, paste0(".pred_", levels(iris$Species)))
  expect_equal(rowSums(probabilities), rep(1, nrow(iris)))
  expect_true(info$has_vector_leaves)
  expect_true(info$average_tree_output)
  expect_true(info$has_probability_output)
  expect_identical(info$treelite_postprocessor, "identity_multiclass")
  expect_equal(
    unname(apply(per_tree, c(1L, 3L), sum) / info$num_trees),
    unname(as.matrix(probabilities)),
    tolerance = 1e-6
  )
})

test_that("random forest binary classes honor the probability threshold", {
  data <- iris[iris$Species != "virginica", ]
  data$Species <- droplevels(data$Species)
  model <- cuda_ml_rand_forest(Species ~ ., data, trees = 100L)

  probabilities <- predict(model, data, type = "prob")
  classes <- predict(model, data, type = "class", threshold = 0.25)
  expected <- factor(
    levels(data$Species)[1L + (probabilities[[2L]] >= 0.25)],
    levels = levels(data$Species)
  )

  expect_identical(classes$.pred_class, expected)
})

test_that("random forest classifier works as expected through parsnip", {
  skip_if_not_installed("parsnip")
  library(parsnip)

  cuda_ml_rf_model <- rand_forest(trees = 200, mode = "classification") |>
    set_engine("cuda.ml", bootstrap = FALSE) |>
    fit(Species ~ ., data = iris)
  sklearn_rf_model <- sklearn$ensemble$RandomForestClassifier(
    n_estimators = 200L,
    bootstrap = FALSE
  )
  sklearn_rf_model$fit(
    X = as.matrix(iris[which(names(iris) != "Species")]),
    y = as.integer(iris$Species)
  )

  cuda_ml_preds <- predict(
    cuda_ml_rf_model,
    iris[which(names(iris) != "Species")]
  )
  sklearn_preds <- sklearn_rf_model$predict(
    as.matrix(iris[which(names(iris) != "Species")])
  )

  expect_equal(
    as.integer(cuda_ml_preds$.pred_class),
    as.integer(sklearn_preds)
  )
})

test_that("random forest regressor works as expected through parsnip", {
  skip_if_not_installed("parsnip")
  library(parsnip)

  cuda_ml_rf_model <- rand_forest(trees = 200, mode = "regression") |>
    set_engine("cuda.ml", bootstrap = FALSE) |>
    fit(mpg ~ ., data = mtcars)
  cuda_ml_preds <- predict(
    cuda_ml_rf_model,
    mtcars[which(names(mtcars) != "mpg")]
  )

  expect_equal(cuda_ml_preds$.pred, mtcars$mpg, tolerance = 0.2)
})
