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

test_that("nvForest checkpoint APIs fail before native loading", {
  expect_error(
    cuda_ml_nvforest_export(list(), tempdir(), "model"),
    "nvForest-backed model"
  )
  expect_error(
    cuda_ml_nvforest_import(tempdir(), "missing", device = "cpu"),
    "do not exist"
  )
})

test_that("the complete backend retains CPU nvForest inference", {
  skip_if_not(
    identical(Sys.getenv("CUDA_ML_FULL_CPU_NVFOREST_TESTS"), "true"),
    "requires the complete backend CPU inference environment"
  )
  info <- cuda_ml_backend_info()
  skip_if_not(info$runtime_installed, "requires the complete runtime")
  skip_if(
    info$nvforest_cpu_runtime_installed,
    "requires a cache without the slim CPU backend"
  )

  model <- cuda_ml_nvforest_load_model(
    test_path("fixtures", "nvforest", "xgboost.ubj"),
    device = "cpu"
  )

  expect_equal(
    predict(model, nvforest_fixture_data)$.pred,
    nvforest_format_cases[[1L]]$expected,
    tolerance = 1e-6,
    scale = 1
  )
  expect_true(cuda_ml_backend_info()$backend_loaded)
  expect_false(cuda_ml_backend_info()$nvforest_cpu_backend_loaded)
})

test_that("nvForest loads every advertised model format", {
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_runtime_installed,
    "requires the CPU-only nvForest runtime"
  )

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

test_that("nvForest checkpoint exports round-trip on CPU", {
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_runtime_installed ||
      cuda_ml_backend_info()$runtime_installed,
    "requires an nvForest runtime"
  )

  x <- nvforest_fixture_data
  model <- cuda_ml_nvforest_load_model(
    test_path("fixtures", "nvforest", "xgboost.ubj"),
    model_type = "xgboost_ubj",
    device = "cpu"
  )
  directory <- tempfile("nvforest-export-")
  dir.create(directory)
  on.exit(unlink(directory, recursive = TRUE))

  paths <- cuda_ml_nvforest_export(model, directory, "regression")

  expect_identical(names(paths), c("checkpoint", "metadata"))
  expect_identical(
    basename(paths),
    c("regression.treelite.checkpoint", "regression.cuda-ml.json")
  )
  expect_true(all(file.exists(paths)))
  expect_identical(sort(list.files(directory)), sort(basename(paths)))

  metadata <- jsonlite::read_json(paths[["metadata"]], simplifyVector = TRUE)
  expect_identical(metadata$format, "cuda_ml_nvforest_export")
  expect_identical(metadata$schema, 1L)
  expect_identical(
    metadata$checkpoint$file,
    "regression.treelite.checkpoint"
  )
  expect_match(metadata$checkpoint$sha256, "^[[:xdigit:]]{64}$")
  expect_identical(metadata$model$mode, "regression")

  imported <- cuda_ml_nvforest_import(
    directory,
    "regression",
    device = "cpu"
  )
  checkpoint <- cuda_ml_nvforest_load_model(
    paths[["checkpoint"]],
    model_type = "treelite_checkpoint",
    device = "cpu"
  )
  expected <- predict(model, x)

  expect_s3_class(imported, "cuda_ml_nvforest")
  expect_identical(cuda_ml_nvforest_info(imported)$device, "cpu")
  expect_equal(predict(imported, x), expected, tolerance = 1e-6)
  expect_equal(predict(checkpoint, x), expected, tolerance = 1e-6)
  expect_error(
    cuda_ml_nvforest_export(model, directory, "regression"),
    "already exist"
  )
  overwritten <- cuda_ml_nvforest_export(
    model,
    directory,
    "regression",
    overwrite = TRUE
  )
  expect_identical(overwritten, paths)
  expect_equal(
    predict(
      cuda_ml_nvforest_import(directory, "regression", device = "cpu"),
      x
    ),
    expected,
    tolerance = 1e-6
  )

  checkpoint_bytes <- readBin(
    paths[["checkpoint"]],
    what = "raw",
    n = file.info(paths[["checkpoint"]])$size
  )
  checkpoint_bytes[[1L]] <- as.raw(
    bitwXor(as.integer(checkpoint_bytes[[1L]]), 1L)
  )
  writeBin(checkpoint_bytes, paths[["checkpoint"]])
  expect_error(
    cuda_ml_nvforest_import(directory, "regression", device = "cpu"),
    "SHA-256"
  )
})

test_that("nvForest infers every documented model suffix", {
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_runtime_installed,
    "requires the CPU-only nvForest runtime"
  )

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
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_runtime_installed,
    "requires the CPU-only nvForest runtime"
  )

  x <- nvforest_fixture_data
  model <- cuda_ml_nvforest_load_model(
    test_path("fixtures", "nvforest", "xgboost.ubj"),
    model_type = "xgboost_ubj",
    device = "cpu"
  )
  expected <- nvforest_format_cases[[1L]]$expected
  predictions <- predict(model, x)
  info <- cuda_ml_nvforest_info(model)
  leaf_ids <- cuda_ml_nvforest_leaf_ids(model, x)
  per_tree <- cuda_ml_nvforest_predict_per_tree(model, x)
  bundled <- bundle::bundle(model)
  bundled_object <- bundled$object
  bundled_state <- unserialize(bundled_object$state)
  unbundled <- bundle::unbundle(bundled)
  expect_error(
    bundle::bundle(model, device = "tpu"),
    "one of.*gpu.*cpu"
  )

  state_path <- tempfile(fileext = ".cuda-ml-state")
  on.exit(unlink(state_path))
  connection <- file(state_path, open = "wb")
  cuda_ml_serialize(model, connection)
  close(connection)
  restored <- callr::r(
    function(state_path, x) {
      suppressPackageStartupMessages(library(cuda.ml))

      connection <- file(state_path, open = "rb")
      model <- cuda_ml_unserialize(connection, device = "cpu")
      close(connection)

      list(
        info = cuda_ml_nvforest_info(model),
        predictions = predict(model, x)
      )
    },
    args = list(state_path = state_path, x = x),
    env = c(CUDA_VISIBLE_DEVICES = "-1"),
    stdout = "",
    stderr = ""
  )

  compatible <- unserialize(cuda_ml_serialize(model))
  provenance <- c(
    "cuda_version",
    "rapids_version",
    "nvforest_version",
    "platform"
  )
  compatible$backend[provenance] <- "different provenance"
  compatible_predictions <- predict(
    cuda_ml_unserialize(serialize(compatible, NULL), device = "cpu"),
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
  expect_identical(dim(leaf_ids), c(nrow(x), info$num_trees))
  expect_type(leaf_ids, "integer")
  expect_identical(dim(per_tree), c(nrow(x), info$num_trees))
  expect_equal(predictions$.pred, expected, tolerance = 1e-6, scale = 1)
  expect_equal(compatible_predictions, predictions)
  expect_identical(restored$info$device, "cpu")
  expect_equal(restored$predictions, predictions)
  expect_true(cuda_ml_runtime_audit(device = "cpu"))
  expect_true(cuda_ml_backend_info()$nvforest_cpu_backend_loaded)
  expect_equal(predict(model, x), predictions)
  expect_identical(bundled_object$device, "cpu")
  expect_false("inference" %in% names(bundled_state$payload))
  expect_identical(cuda_ml_nvforest_info(unbundled)$device, "cpu")
  expect_equal(predict(unbundled, x), predictions)
})
