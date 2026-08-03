nvforest_fixture_data <- matrix(
  c(0, 0, 0, 1, 1, 0, 1, 1, 2, 2, -1, -1),
  ncol = 2,
  byrow = TRUE
)

nvforest_format_cases <- list(
  list(
    label = "XGBoost UBJSON",
    file = "xgboost.ubj",
    model_type = "xgboost_ubj",
    expected = c(0.5, 0.5, 1.5, 1.5, 4, -2)
  ),
  list(
    label = "XGBoost JSON",
    file = "xgboost.json",
    model_type = "xgboost_json",
    expected = c(0.5, 0.5, 1.5, 1.5, 4, -2)
  ),
  list(
    label = "XGBoost legacy binary",
    file = "xgboost.model",
    model_type = "xgboost_legacy",
    expected = c(0.5, 0.5, 1.5, 1.5, 4, -2)
  ),
  list(
    label = "LightGBM",
    file = "lightgbm.txt",
    model_type = "lightgbm",
    expected = c(0.5, 1, 1, 1.5, 4.5, -2.5)
  ),
  list(
    label = "Treelite checkpoint",
    file = "treelite.checkpoint",
    model_type = "treelite_checkpoint",
    expected = c(0.5, 0.5, 1.5, 1.5, 4, -2)
  )
)

test_that("nvForest loads every advertised model format", {
  skip_if_not(cuda_ml_backend_info()$runtime_installed, "requires a runtime")

  for (case in nvforest_format_cases) {
    model <- cuda_ml_nvforest_load_model(
      test_path("fixtures", "nvforest", case$file),
      model_type = case$model_type,
      device = "cpu"
    )

    expect_identical(cuda_ml_nvforest_info(model)$task_type, "regression")
    expect_equal(
      predict(model, nvforest_fixture_data)$.pred,
      case$expected,
      tolerance = 1e-6,
      scale = 1,
      info = case$label
    )
  }
})

test_that("nvForest infers every documented model suffix", {
  skip_if_not(cuda_ml_backend_info()$runtime_installed, "requires a runtime")

  for (case in nvforest_format_cases[seq_len(4L)]) {
    fixture <- test_path("fixtures", "nvforest", case$file)
    model <- cuda_ml_nvforest_load_model(
      fixture,
      device = "cpu"
    )

    expect_equal(
      predict(model, nvforest_fixture_data)$.pred,
      case$expected,
      tolerance = 1e-6,
      scale = 1,
      info = case$label
    )

    uppercase_path <- tempfile(
      fileext = paste0(".", toupper(tools::file_ext(case$file)))
    )
    on.exit(unlink(uppercase_path), add = TRUE)
    expect_true(file.copy(fixture, uppercase_path))
    uppercase_model <- cuda_ml_nvforest_load_model(
      uppercase_path,
      device = "cpu"
    )
    expect_equal(
      predict(uppercase_model, nvforest_fixture_data)$.pred,
      case$expected,
      tolerance = 1e-6,
      scale = 1,
      info = case$label
    )
  }

  expect_error(
    cuda_ml_nvforest_load_model(
      test_path("fixtures", "nvforest", "treelite.checkpoint"),
      device = "cpu"
    ),
    "Cannot infer nvForest model type"
  )
})

test_that("nvForest supports CPU-only inference and restoration", {
  skip_if_not(cuda_ml_backend_info()$runtime_installed, "requires a runtime")

  x <- nvforest_fixture_data
  model <- cuda_ml_nvforest_load_model(
    test_path("fixtures", "nvforest", "xgboost.ubj"),
    model_type = "xgboost_ubj",
    device = "cpu"
  )
  expected <- nvforest_format_cases[[1L]]$expected
  predictions <- predict(model, x)
  info <- cuda_ml_nvforest_info(model)
  per_tree <- cuda_ml_nvforest_predict_per_tree(model, x)
  restored <- predict_saved_models_in_sub_proc(model, x)

  compatible <- unserialize(cuda_ml_serialize(model))
  provenance <- c(
    "cuda_version",
    "rapids_version",
    "nvforest_version",
    "platform"
  )
  compatible$backend[provenance] <- "different provenance"
  compatible_predictions <- predict(
    cuda_ml_unserialize(serialize(compatible, NULL)),
    x
  )

  incompatible <- unserialize(cuda_ml_serialize(model))
  incompatible$backend$treelite_version <- "5.0.0"
  expect_error(
    cuda_ml_unserialize(serialize(incompatible, NULL)),
    "requires backend `treelite_version`.*5.0.0"
  )

  expect_identical(info$device, "cpu")
  expect_identical(info$align_bytes, 64L)
  expect_identical(dim(per_tree), c(nrow(x), info$num_trees))
  expect_equal(predictions$.pred, expected, tolerance = 1e-6, scale = 1)
  expect_equal(compatible_predictions, predictions)
  expect_equal(restored$restored, predictions)
  expect_equal(restored$unbundled, predictions)
  expect_true(cuda_ml_runtime_audit())
  expect_true(cuda_ml_backend_info()$backend_loaded)
  expect_equal(predict(model, x), predictions)
})
