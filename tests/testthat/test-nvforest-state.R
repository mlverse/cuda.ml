nvforest_state_data <- matrix(
  c(0, 0, 0, 1, 1, 0, 1, 1, 2, 2, -1, -1),
  ncol = 2,
  byrow = TRUE
)

test_that("nvForest states are device neutral", {
  skip_if_not(run_gpu_tests, "requires the GPU test environment")

  x <- nvforest_state_data
  model <- cuda_ml_nvforest_load_model(
    test_path("fixtures", "nvforest", "xgboost.ubj"),
    model_type = "xgboost_ubj",
    device = "gpu",
    layout = "layered",
    default_chunk_size = 8L,
    align_bytes = 128L
  )
  state <- unserialize(cuda_ml_serialize(model))

  expect_identical(state$model_abi, "cuda_ml_nvforest_model_state_v2")
  expect_false("inference" %in% names(state$payload))
  expect_named(
    state$payload,
    c(
      "model",
      "class_levels",
      "precision",
      "averaged_vector_leaf_probabilities",
      "blueprint"
    )
  )

  restored <- cuda_ml_unserialize(serialize(state, NULL))
  expect_identical(cuda_ml_nvforest_info(restored)$device, "gpu")
  expect_equal(predict(restored, x), predict(model, x))
})

test_that("one nvForest state restores on CPU and GPU", {
  skip_if_not(run_gpu_tests, "requires the GPU test environment")
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_runtime_installed,
    "requires the CPU-only nvForest runtime"
  )

  x <- nvforest_state_data
  model <- cuda_ml_nvforest_load_model(
    test_path("fixtures", "nvforest", "xgboost.ubj"),
    model_type = "xgboost_ubj",
    device = "gpu"
  )
  state <- cuda_ml_serialize(model)

  cpu_model <- cuda_ml_unserialize(state, device = "cpu")
  gpu_model <- cuda_ml_unserialize(state, device = "gpu")

  expect_identical(cuda_ml_nvforest_info(cpu_model)$device, "cpu")
  expect_identical(cuda_ml_nvforest_info(gpu_model)$device, "gpu")
  expect_equal(predict(cpu_model, x), predict(model, x), tolerance = 1e-6)
  expect_equal(predict(gpu_model, x), predict(model, x), tolerance = 1e-6)

  cpu_first <- callr::r(
    function(state, x) {
      suppressPackageStartupMessages(library(cuda.ml))

      cpu_model <- cuda_ml_unserialize(state, device = "cpu")
      cpu_prediction <- predict(cpu_model, x)
      gpu_model <- cuda_ml_unserialize(state, device = "gpu")

      list(
        cpu = cpu_prediction,
        gpu = predict(gpu_model, x),
        info = cuda_ml_backend_info()
      )
    },
    args = list(state = state, x = x),
    stdout = "",
    stderr = ""
  )

  expect_equal(cpu_first$cpu, predict(model, x), tolerance = 1e-6)
  expect_equal(cpu_first$gpu, predict(model, x), tolerance = 1e-6)
  expect_true(cpu_first$info$nvforest_cpu_backend_loaded)
  expect_true(cpu_first$info$backend_loaded)
})

test_that("restore-time nvForest options are applied", {
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_runtime_installed,
    "requires the CPU-only nvForest runtime"
  )

  model <- cuda_ml_nvforest_load_model(
    test_path("fixtures", "nvforest", "xgboost.ubj"),
    model_type = "xgboost_ubj",
    device = "cpu"
  )
  restored <- cuda_ml_unserialize(
    cuda_ml_serialize(model),
    device = "cpu",
    layout = "layered",
    precision = "single",
    default_chunk_size = 8L,
    align_bytes = 128L
  )
  info <- cuda_ml_nvforest_info(restored)

  expect_identical(info$device, "cpu")
  expect_identical(info$layout, "layered")
  expect_identical(info$precision, "single")
  expect_identical(info$default_chunk_size, 8L)
  expect_identical(info$align_bytes, 128L)
})

test_that("legacy nvForest states retain their stored inference settings", {
  skip_if_not(run_gpu_tests, "requires the GPU test environment")

  x <- nvforest_state_data
  model <- cuda_ml_nvforest_load_model(
    test_path("fixtures", "nvforest", "xgboost.ubj"),
    model_type = "xgboost_ubj",
    device = "gpu",
    layout = "layered"
  )
  state <- unserialize(cuda_ml_serialize(model))
  state$model_abi <- "cuda_ml_nvforest_model_state"
  state$payload$inference <- list(
    device = 1L,
    device_id = -1L,
    layout = 2L,
    precision = -1L,
    default_chunk_size = 0L,
    align_bytes = 0L
  )
  state$payload$precision <- NULL
  class(state) <- c(
    "cuda_ml_nvforest_model_state",
    "cuda_ml_model_state"
  )

  restored <- cuda_ml_unserialize(serialize(state, NULL))

  expect_identical(cuda_ml_nvforest_info(restored)$device, "gpu")
  expect_identical(cuda_ml_nvforest_info(restored)$layout, "layered")
  expect_equal(predict(restored, x), predict(model, x))
})

test_that("nvForest restore options reject other model states", {
  state <- readRDS(test_path("fixtures", "linear-model-state-schema-1.rds"))

  options <- list(
    device = "cpu",
    device_id = 0L,
    layout = "layered",
    precision = "single",
    default_chunk_size = 8L,
    align_bytes = 128L
  )
  for (option in names(options)) {
    args <- c(list(state), options[option])
    expect_error(
      do.call(cuda_ml_unserialize, args),
      "only supported for nvForest",
      info = option
    )
  }
})

test_that("GPU-trained random forests restore for CPU inference", {
  skip_if_not(run_gpu_tests, "requires the GPU test environment")
  skip_if_not(
    identical(Sys.getenv("CUDA_ML_CROSS_DEVICE_TESTS"), "true"),
    "requires the isolated cross-device deployment environment"
  )
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_runtime_installed,
    "requires the CPU-only nvForest runtime"
  )

  model <- cuda_ml_rand_forest(Species ~ ., iris, trees = 50L, seed = 1L)
  predictor_names <- names(iris)[names(iris) != "Species"]
  data <- iris[rev(predictor_names)]
  expected_class <- predict(model, data, type = "class")
  expected_prob <- predict(model, data, type = "prob")
  state <- unserialize(cuda_ml_serialize(model))

  expect_identical(state$model_abi, "cuda_ml_rand_forest_model_state_v2")
  expect_false("inference" %in% names(state$payload))

  bundle_path <- tempfile(fileext = ".rds")
  export_directory <- tempfile("cuda-ml-nvforest-export-")
  cache <- tempfile("cuda-ml-cpu-deployment-")
  dir.create(export_directory)
  on.exit(
    unlink(c(bundle_path, export_directory, cache), recursive = TRUE),
    add = TRUE
  )
  saveRDS(bundle::bundle(model, device = "cpu"), bundle_path)
  cuda_ml_nvforest_export(model, export_directory, "iris-random-forest")

  deployed <- callr::r(
    function(state, data, bundle_path, export_directory, cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      suppressPackageStartupMessages(library(cuda.ml))
      cuda_ml_install(device = "cpu")

      model <- cuda_ml_unserialize(state, device = "cpu")
      unbundled <- bundle::unbundle(readRDS(bundle_path))
      imported <- cuda_ml_nvforest_import(
        export_directory,
        "iris-random-forest",
        device = "cpu"
      )
      list(
        class = class(model),
        imported_class = class(imported),
        info = cuda_ml_nvforest_info(model),
        imported_info = cuda_ml_nvforest_info(imported),
        bundle_info = cuda_ml_nvforest_info(unbundled),
        backend = cuda_ml_backend_info(),
        dlls = names(getLoadedDLLs()),
        class_prediction = predict(model, data, type = "class"),
        probability_prediction = predict(model, data, type = "prob"),
        imported_class_prediction = predict(imported, data, type = "class"),
        imported_probability_prediction = predict(
          imported,
          data,
          type = "prob"
        ),
        bundled_prediction = predict(unbundled, data, type = "class")
      )
    },
    args = list(
      state = serialize(state, NULL),
      data = data,
      bundle_path = bundle_path,
      export_directory = export_directory,
      cache = cache
    ),
    env = c(CUDA_VISIBLE_DEVICES = "-1"),
    stdout = "",
    stderr = ""
  )

  expect_true("cuda_ml_rand_forest" %in% deployed$class)
  expect_true("cuda_ml_rand_forest" %in% deployed$imported_class)
  expect_identical(deployed$info$device, "cpu")
  expect_identical(deployed$imported_info$device, "cpu")
  expect_identical(deployed$bundle_info$device, "cpu")
  expect_true(deployed$backend$nvforest_cpu_backend_loaded)
  expect_true(deployed$backend$nvforest_cpu_runtime_installed)
  expect_false(deployed$backend$runtime_installed)
  expect_false(deployed$backend$backend_loaded)
  expect_true("cuda.ml.nvforest" %in% deployed$dlls)
  expect_false("cuda.ml" %in% deployed$dlls)
  expect_equal(deployed$class_prediction, expected_class)
  expect_equal(
    deployed$probability_prediction,
    expected_prob,
    tolerance = 1e-6
  )
  expect_equal(deployed$imported_class_prediction, expected_class)
  expect_equal(
    deployed$imported_probability_prediction,
    expected_prob,
    tolerance = 1e-6
  )
  expect_equal(deployed$bundled_prediction, expected_class)

  state$model_abi <- "cuda_ml_rand_forest_model_state"
  state$payload$inference <- list(
    device = 1L,
    device_id = -1L,
    layout = 0L,
    precision = -1L,
    default_chunk_size = 0L,
    align_bytes = 0L
  )
  state$payload$precision <- NULL
  class(state) <- c(
    "cuda_ml_rand_forest_model_state",
    "cuda_ml_model_state"
  )
  legacy <- cuda_ml_unserialize(serialize(state, NULL))

  expect_equal(predict(legacy, data, type = "class"), expected_class)
  expect_equal(
    predict(legacy, data, type = "prob"),
    expected_prob,
    tolerance = 1e-6
  )
})

test_that("GPU-trained random forest regressors restore on CPU", {
  skip_if_not(run_gpu_tests, "requires the GPU test environment")
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_runtime_installed,
    "requires the CPU-only nvForest runtime"
  )

  model <- cuda_ml_rand_forest(
    mpg ~ log(disp) + cyl + hp,
    mtcars,
    trees = 50L,
    seed = 1L
  )
  data <- mtcars[c("hp", "disp", "cyl")]
  expected <- predict(model, data)

  restored <- cuda_ml_unserialize(
    cuda_ml_serialize(model),
    device = "cpu"
  )
  export_directory <- tempfile("cuda-ml-nvforest-export-")
  dir.create(export_directory)
  on.exit(unlink(export_directory, recursive = TRUE), add = TRUE)
  paths <- cuda_ml_nvforest_export(
    model,
    export_directory,
    "mtcars-random-forest"
  )
  metadata <- jsonlite::read_json(paths[["metadata"]], simplifyVector = TRUE)
  imported <- cuda_ml_nvforest_import(
    export_directory,
    "mtcars-random-forest",
    device = "cpu"
  )

  expect_s3_class(restored, "cuda_ml_rand_forest")
  expect_s3_class(imported, "cuda_ml_rand_forest")
  expect_identical(
    metadata$model$feature_names,
    c("log(disp)", "cyl", "hp")
  )
  expect_identical(cuda_ml_nvforest_info(restored)$device, "cpu")
  expect_identical(cuda_ml_nvforest_info(imported)$device, "cpu")
  expect_equal(predict(restored, data), expected, tolerance = 1e-6)
  expect_equal(predict(imported, data), expected, tolerance = 1e-6)
})
