test_that("native operations require an explicit runtime installation", {
  skip_if_not(
    identical(Sys.getenv("CUDA_ML_RUNTIME_TESTS"), "true"),
    "set CUDA_ML_RUNTIME_TESTS=true to exercise the managed runtime"
  )
  skip_if_not(
    cuda_ml_backend_info()$backend_available,
    "requires a published backend"
  )

  cache <- tempfile("cuda-ml-functional-cache-")
  state <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      library(cuda.ml)
      error <- tryCatch(
        {
          cuda_ml_kmeans(matrix(c(0, 0, 1, 1), ncol = 2), k = 2)
          NULL
        },
        error = conditionMessage
      )
      list(error = error, cache_exists = dir.exists(cache))
    },
    args = list(cache = cache)
  )

  expect_match(state$error, "cuda_ml_install()", fixed = TRUE)
  expect_false(state$cache_exists)
})

test_that("the managed runtime can be prewarmed without loading the backend", {
  skip_if_not(
    identical(Sys.getenv("CUDA_ML_RUNTIME_TESTS"), "true"),
    "set CUDA_ML_RUNTIME_TESTS=true to exercise the managed runtime"
  )
  skip_if_not(
    cuda_ml_backend_info()$backend_available,
    "requires a published backend"
  )

  cache <- Sys.getenv(
    "CUDA_ML_CACHE_DIR",
    unset = tempfile("cuda-ml-functional-cache-")
  )
  Sys.setenv(CUDA_ML_CACHE_DIR = cache)
  result <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      library(cuda.ml)
      cuda_ml_install()
      list(
        dll_loaded = "cuda.ml" %in% names(getLoadedDLLs()),
        cache = list.files(cache, recursive = TRUE, all.files = TRUE)
      )
    },
    args = list(cache = cache)
  )

  expect_false(result$dll_loaded)
  expect_true(any(grepl("[.]complete$", result$cache)))
  expect_false(any(grepl(
    "libnvrtc-a49e67e8[.]so[.]13[.]0[.]88$",
    result$cache
  )))
  expect_true(any(grepl(
    "libnvrtc[.]so[.]13$",
    result$cache
  )))

  messages <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      library(cuda.ml)
      messages <- character()
      withCallingHandlers(
        cuda_ml_install(),
        message = function(cnd) {
          messages <<- c(messages, conditionMessage(cnd))
          invokeRestart("muffleMessage")
        }
      )
      messages
    },
    args = list(cache = cache)
  )
  expect_identical(messages, character())
})

test_that("the CPU-only nvForest backend can be installed independently", {
  skip_if_not(
    identical(Sys.getenv("CUDA_ML_CPU_RUNTIME_TESTS"), "true"),
    "set CUDA_ML_CPU_RUNTIME_TESTS=true to exercise the CPU runtime"
  )
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_backend_available,
    "requires a published CPU-only nvForest backend"
  )

  cache <- tempfile("cuda-ml-cpu-functional-cache-")
  result <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      library(cuda.ml)
      cuda_ml_install(device = "cpu")
      list(
        info = cuda_ml_backend_info(),
        dlls = names(getLoadedDLLs()),
        runtime_files = list.files(
          file.path(cuda_ml_backend_info()$nvforest_cpu_runtime_path, "lib")
        ),
        third_party_licenses = readLines(file.path(
          cuda_ml_backend_info()$nvforest_cpu_runtime_path,
          "lib",
          "THIRD-PARTY-LICENSES.txt"
        ))
      )
    },
    args = list(cache = cache)
  )

  expect_true(result$info$nvforest_cpu_runtime_installed)
  expect_true(dir.exists(result$info$nvforest_cpu_runtime_path))
  expect_false(result$info$nvforest_cpu_backend_loaded)
  expect_false(result$info$runtime_installed)
  expect_false("cuda.ml.nvforest" %in% result$dlls)
  expect_setequal(
    result$runtime_files,
    c(
      "backend.dcf",
      paste0("cuda.ml.nvforest", .Platform$dynlib.ext),
      "THIRD-PARTY-LICENSES.txt"
    )
  )
  expect_true(any(grepl("^cuda[.]ml$", result$third_party_licenses)))
  expect_true(any(grepl("^nvForest 26[.]06[.]0$", result$third_party_licenses)))
  expect_true(any(grepl("^Treelite 4[.]7[.]0$", result$third_party_licenses)))
})

test_that("the CPU-only nvForest backend passes an explicit audit", {
  skip_if_not(
    cuda_ml_backend_info()$nvforest_cpu_runtime_installed,
    "requires the CPU-only nvForest backend"
  )

  audited <- callr::r(
    function() {
      library(cuda.ml)
      cuda_ml_runtime_audit(device = "cpu")
    }
  )

  expect_true(audited)
})

test_that("the installed runtime passes an explicit content audit", {
  skip_if_not(cuda_ml_backend_info()$runtime_installed, "requires a runtime")

  state <- callr::r(
    function() {
      library(cuda.ml)
      list(
        audited = cuda_ml_runtime_audit(),
        info = cuda_ml_backend_info()
      )
    }
  )

  expect_true(state$audited)
  expect_true(state$info$runtime_installed)
  expect_true(dir.exists(state$info$runtime_path))
  expect_false(state$info$backend_loaded)
})

test_that("R CMD check can audit the functional backend", {
  skip_if_not(cuda_ml_backend_info()$runtime_installed, "requires a runtime")

  audited <- callr::r(
    function() {
      Sys.setenv("_R_CHECK_PACKAGE_NAME_" = "cuda.ml")
      library(cuda.ml)
      cuda_ml_runtime_audit()
    }
  )

  expect_true(audited)
})
