test_that("nvForest supports CPU-only inference and restoration", {
  skip_if_not(cuda_ml_backend_info()$runtime_installed, "requires a runtime")
  skip_if_not_installed("xgboost")

  x <- matrix(
    c(0, 0, 0, 1, 10, 10, 10, 11),
    ncol = 2,
    byrow = TRUE
  )
  y <- c(0, 0, 1, 1)
  training <- xgboost::xgb.DMatrix(x, label = y)
  xgb_model <- xgboost::xgb.train(
    params = list(objective = "reg:squarederror", max_depth = 2L),
    data = training,
    nrounds = 2L,
    verbose = 0L
  )
  path <- tempfile(fileext = ".ubj")
  on.exit(unlink(path))
  xgboost::xgb.save(xgb_model, path)

  model <- cuda_ml_nvforest_load_model(
    path,
    model_type = "xgboost_ubj",
    device = "cpu"
  )
  expected <- as.numeric(predict(xgb_model, x))
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
