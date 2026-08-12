skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("Support Vector Machine")

test_that("cuda_ml_svm() works as expected for binary classification tasks", {
  gen_cuda_ml_binary_svc_input <- function() {
    data <- scaled_penguin_predictors
    data$is_chinstrap <- factor(penguins$species == "Chinstrap")

    data
  }
  cuda_ml_binary_svc_input <- gen_cuda_ml_binary_svc_input()

  gen_sklearn_binary_svc_input <- function() {
    ds <- sklearn_penguins_dataset
    ds$target <- (
      ds$target == which(levels(penguins$species) == "Chinstrap")
    )

    ds
  }
  sklearn_binary_svc_input <- gen_sklearn_binary_svc_input()

  cuda_ml_binary_svc_model <- cuda_ml_svm(
    formula = is_chinstrap ~ .,
    data = cuda_ml_binary_svc_input,
    kernel = "rbf"
  )
  cuda_ml_binary_svc_preds <- predict(
    cuda_ml_binary_svc_model,
    cuda_ml_binary_svc_input[,
      names(cuda_ml_binary_svc_input) != "is_chinstrap"
    ]
  )

  sklearn_binary_svc_model <- sklearn$svm$SVC(kernel = "rbf", gamma = "auto")
  sklearn_binary_svc_model$fit(
    sklearn_binary_svc_input$data,
    sklearn_binary_svc_input$target
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
  data <- scaled_penguin_predictors
  data$species <- penguins$species
  cuda_ml_multiclass_svc_input <- scaled_penguin_predictors

  cuda_ml_multiclass_svc_model <- cuda_ml_svm(
    formula = species ~ .,
    data = data,
    kernel = "rbf"
  )
  cuda_ml_multiclass_svc_preds <- predict(
    cuda_ml_multiclass_svc_model,
    cuda_ml_multiclass_svc_input
  )

  sklearn_multiclass_svc_model <- sklearn$svm$SVC(
    kernel = "rbf",
    gamma = "auto"
  )
  sklearn_multiclass_svc_model$fit(
    as.matrix(unname(scaled_penguin_predictors)),
    as.integer(penguins[["species"]])
  )
  sklearn_multiclass_svc_preds <- sklearn_multiclass_svc_model$predict(
    as.matrix(unname(scaled_penguin_predictors))
  )

  expect_equal(
    as.integer(cuda_ml_multiclass_svc_preds$.pred_class),
    as.integer(sklearn_multiclass_svc_preds)
  )
})

test_that("cuda_ml_svm() works as expected for regression tasks", {
  cuda_ml_svr_model <- cuda_ml_svm(
    formula = mpg ~ .,
    data = mtcars,
    kernel = "rbf"
  )
  cuda_ml_svr_preds <- predict(
    cuda_ml_svr_model,
    mtcars[, names(mtcars) != "mpg"]
  )

  sklearn_svr_model <- sklearn$svm$SVR(kernel = "rbf", gamma = "auto")
  sklearn_svr_model$fit(
    sklearn_mtcars_dataset$data,
    sklearn_mtcars_dataset$target
  )
  sklearn_svr_preds <- sklearn_svr_model$predict(sklearn_mtcars_dataset$data)

  expect_equal(
    cuda_ml_svr_preds$.pred,
    as.numeric(sklearn_svr_preds),
    tolerance = 1e-3,
    scale = 1
  )
})

test_that("cuda_ml_svm() classification works as expected through parsnip", {
  skip_if_not_installed("parsnip")
  library(parsnip)

  data <- scaled_penguin_predictors
  data$species <- penguins$species
  cuda_ml_multiclass_svc_input <- scaled_penguin_predictors

  cuda_ml_multiclass_svc_model <- svm_rbf(mode = "classification") |>
    set_engine("cuda.ml") |>
    fit(species ~ ., data = data)
  cuda_ml_multiclass_svc_preds <- predict(
    cuda_ml_multiclass_svc_model,
    cuda_ml_multiclass_svc_input
  )

  sklearn_multiclass_svc_model <- sklearn$svm$SVC(
    kernel = "rbf",
    gamma = "auto"
  )
  sklearn_multiclass_svc_model$fit(
    as.matrix(unname(scaled_penguin_predictors)),
    as.integer(penguins[["species"]])
  )
  sklearn_multiclass_svc_preds <- sklearn_multiclass_svc_model$predict(
    as.matrix(unname(scaled_penguin_predictors))
  )

  expect_equal(
    as.integer(cuda_ml_multiclass_svc_preds$.pred_class),
    as.integer(sklearn_multiclass_svc_preds)
  )
})

test_that("cuda_ml_svm() regression works as expected through parsnip", {
  skip_if_not_installed("parsnip")
  library(parsnip)

  cuda_ml_svr_model <- cuda_ml_svm(
    formula = mpg ~ .,
    data = mtcars,
    kernel = "rbf"
  )
  cuda_ml_svr_model <- svm_rbf(mode = "regression") |>
    set_engine("cuda.ml") |>
    fit(mpg ~ ., data = mtcars)
  cuda_ml_svr_preds <- predict(
    cuda_ml_svr_model,
    mtcars[, names(mtcars) != "mpg"]
  )

  sklearn_svr_model <- sklearn$svm$SVR(kernel = "rbf", gamma = "auto")
  sklearn_svr_model$fit(
    sklearn_mtcars_dataset$data,
    sklearn_mtcars_dataset$target
  )
  sklearn_svr_preds <- sklearn_svr_model$predict(sklearn_mtcars_dataset$data)

  expect_equal(
    cuda_ml_svr_preds$.pred,
    as.numeric(sklearn_svr_preds),
    tolerance = 1e-3,
    scale = 1
  )
})
