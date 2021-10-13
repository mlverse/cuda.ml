context("Random Forest")

test_that("random forest classifier works as expected", {
  cuda_ml_rf_model <- cuda_ml_rand_forest(
    formula = Species ~ ., data = iris, trees = 200, bootstrap = FALSE
  )
  sklearn_rf_model <- sklearn$ensemble$RandomForestClassifier(
    n_estimators = 200L, bootstrap = FALSE
  )
  sklearn_rf_model$fit(
    X = as.matrix(iris[which(names(iris) != "Species")]),
    y = as.integer(iris$Species)
  )

  cuda_ml_preds <- predict(
    cuda_ml_rf_model, iris[which(names(iris) != "Species")]
  )
  sklearn_preds <- sklearn_rf_model$predict(
    as.matrix(iris[which(names(iris) != "Species")])
  )

  expect_equal(
    as.integer(cuda_ml_preds$.pred_class), as.integer(sklearn_preds)
  )
})

test_that("random forest regressor works as expected", {
  cuda_ml_rf_model <- cuda_ml_rand_forest(
    formula = mpg ~ ., data = mtcars, trees = 100, bootstrap = FALSE
  )
  cuda_ml_preds <- predict(
    cuda_ml_rf_model, mtcars[which(names(mtcars) != "mpg")]
  )

  expect_equal(cuda_ml_preds$.pred, mtcars$mpg, tolerance = 0.2)
})

test_that("random forest classifier works as expected through parsnip", {
  require("parsnip")

  cuda_ml_rf_model <- rand_forest(trees = 200, mode = "classification") %>%
    set_engine("cuda.ml", bootstrap = FALSE) %>%
    fit(Species ~ ., data = iris)
  sklearn_rf_model <- sklearn$ensemble$RandomForestClassifier(
    n_estimators = 200L, bootstrap = FALSE
  )
  sklearn_rf_model$fit(
    X = as.matrix(iris[which(names(iris) != "Species")]),
    y = as.integer(iris$Species)
  )

  cuda_ml_preds <- predict(
    cuda_ml_rf_model, iris[which(names(iris) != "Species")]
  )
  sklearn_preds <- sklearn_rf_model$predict(
    as.matrix(iris[which(names(iris) != "Species")])
  )

  expect_equal(
    as.integer(cuda_ml_preds$.pred_class), as.integer(sklearn_preds)
  )
})

test_that("random forest regressor works as expected through parsnip", {
  require("parsnip")

  cuda_ml_rf_model <- rand_forest(trees = 200, mode = "regression") %>%
    set_engine("cuda.ml", bootstrap = FALSE) %>%
    fit(mpg ~ ., data = mtcars)
  cuda_ml_preds <- predict(
    cuda_ml_rf_model, mtcars[which(names(mtcars) != "mpg")]
  )

  expect_equal(cuda_ml_preds$.pred, mtcars$mpg, tolerance = 0.2)
})
