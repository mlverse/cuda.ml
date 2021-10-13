context("Support Vector Machine")

test_that("cuda_ml_svm() works as expected for binary classification tasks", {
  gen_cuda_ml_binary_svc_input <- function() {
    data <- iris
    data[, "is_versicolor"] <- factor(data[, "Species"] == "versicolor")

    data[, names(data) != "Species"]
  }
  cuda_ml_binary_svc_input <- gen_cuda_ml_binary_svc_input()

  gen_sklearn_binary_svc_input <- function() {
    ds <- sklearn_iris_dataset
    ds$target <- (ds$target == which(levels(iris$Species) == "versicolor"))

    ds
  }
  sklearn_binary_svc_input <- gen_sklearn_binary_svc_input()

  cuda_ml_binary_svc_model <- cuda_ml_svm(
    formula = is_versicolor ~ .,
    data = cuda_ml_binary_svc_input,
    kernel = "rbf"
  )
  cuda_ml_binary_svc_preds <- predict(
    cuda_ml_binary_svc_model,
    cuda_ml_binary_svc_input[, names(cuda_ml_binary_svc_input) != "is_versicolor"]
  )

  sklearn_binary_svc_model <- sklearn$svm$SVC(kernel = "rbf", gamma = "auto")
  sklearn_binary_svc_model$fit(
    sklearn_binary_svc_input$data, sklearn_binary_svc_input$target
  )
  sklearn_binary_svc_preds <- sklearn_binary_svc_model$predict(
    sklearn_binary_svc_input$data
  )

  expect_equal(
    as.logical(cuda_ml_binary_svc_preds$.pred_class),
    as.logical(sklearn_binary_svc_preds)
  )
})

test_that("cuda_ml_svm() works as expected for multi-class classification tasks", {
  cuda_ml_multiclass_svc_input <- iris[, names(iris) != "Species"]

  cuda_ml_multiclass_svc_model <- cuda_ml_svm(
    formula = Species ~ ., data = iris, kernel = "rbf"
  )
  cuda_ml_multiclass_svc_preds <- predict(
    cuda_ml_multiclass_svc_model, cuda_ml_multiclass_svc_input
  )

  sklearn_multiclass_svc_model <- sklearn$svm$SVC(kernel = "rbf", gamma = "auto")
  sklearn_multiclass_svc_model$fit(
    as.matrix(unname(iris[, names(iris) != "Species"])),
    as.integer(iris[["Species"]])
  )
  sklearn_multiclass_svc_preds <- sklearn_multiclass_svc_model$predict(
    as.matrix(unname(iris[, names(iris) != "Species"]))
  )

  expect_equal(
    as.integer(cuda_ml_multiclass_svc_preds$.pred_class),
    as.integer(sklearn_multiclass_svc_preds)
  )
})

test_that("cuda_ml_svm() works as expected for regression tasks", {
  cuda_ml_svr_model <- cuda_ml_svm(
    formula = mpg ~ ., data = mtcars, kernel = "rbf"
  )
  cuda_ml_svr_preds <- predict(
    cuda_ml_svr_model, mtcars[, names(mtcars) != "mpg"]
  )

  sklearn_svr_model <- sklearn$svm$SVR(kernel = "rbf", gamma = "auto")
  sklearn_svr_model$fit(
    sklearn_mtcars_dataset$data, sklearn_mtcars_dataset$target
  )
  sklearn_svr_preds <- sklearn_svr_model$predict(sklearn_mtcars_dataset$data)

  expect_equal(
    cuda_ml_svr_preds$.pred, as.numeric(sklearn_svr_preds),
    tolerance = 1e-3, scale = 1
  )
})

test_that("cuda_ml_svm() classification works as expected through parsnip", {
  require("parsnip")

  cuda_ml_multiclass_svc_input <- iris[, names(iris) != "Species"]

  cuda_ml_multiclass_svc_model <- svm_rbf(mode = "classification") %>%
    set_engine("cuda.ml") %>%
    fit(Species ~ ., data = iris)
  cuda_ml_multiclass_svc_preds <- predict(
    cuda_ml_multiclass_svc_model, cuda_ml_multiclass_svc_input
  )

  sklearn_multiclass_svc_model <- sklearn$svm$SVC(kernel = "rbf", gamma = "auto")
  sklearn_multiclass_svc_model$fit(
    as.matrix(unname(iris[, names(iris) != "Species"])),
    as.integer(iris[["Species"]])
  )
  sklearn_multiclass_svc_preds <- sklearn_multiclass_svc_model$predict(
    as.matrix(unname(iris[, names(iris) != "Species"]))
  )

  expect_equal(
    as.integer(cuda_ml_multiclass_svc_preds$.pred_class),
    as.integer(sklearn_multiclass_svc_preds)
  )
})

test_that("cuda_ml_svm() regression works as expected through parsnip", {
  require("parsnip")

  cuda_ml_svr_model <- cuda_ml_svm(
    formula = mpg ~ ., data = mtcars, kernel = "rbf"
  )
  cuda_ml_svr_model <- svm_rbf(mode = "regression") %>%
    set_engine("cuda.ml") %>%
    fit(mpg ~ ., data = mtcars)
  cuda_ml_svr_preds <- predict(
    cuda_ml_svr_model, mtcars[, names(mtcars) != "mpg"]
  )

  sklearn_svr_model <- sklearn$svm$SVR(kernel = "rbf", gamma = "auto")
  sklearn_svr_model$fit(
    sklearn_mtcars_dataset$data, sklearn_mtcars_dataset$target
  )
  sklearn_svr_preds <- sklearn_svr_model$predict(sklearn_mtcars_dataset$data)

  expect_equal(
    cuda_ml_svr_preds$.pred, as.numeric(sklearn_svr_preds),
    tolerance = 1e-3, scale = 1
  )
})
