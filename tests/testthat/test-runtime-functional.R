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

test_that("the installed runtime passes an explicit content audit", {
  skip_if_not(
    identical(Sys.getenv("CUDA_ML_RUNTIME_TESTS"), "true"),
    "set CUDA_ML_RUNTIME_TESTS=true to exercise the managed runtime"
  )
  skip_if_not(
    cuda_ml_backend_info()$backend_available,
    "requires a published backend"
  )

  expect_true(cuda_ml_runtime_audit())
  info <- cuda_ml_backend_info()
  expect_true(info$runtime_installed)
  expect_true(dir.exists(info$runtime_path))
  expect_false(info$backend_loaded)
})

test_that("R CMD check can audit the functional backend", {
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
  audited <- callr::r(
    function(cache) {
      do.call(
        Sys.setenv,
        setNames(
          list(cache, "cuda.ml"),
          c("CUDA_ML_CACHE_DIR", "_R_CHECK_PACKAGE_NAME_")
        )
      )
      library(cuda.ml)
      cuda_ml_runtime_audit()
    },
    args = list(cache = cache)
  )

  expect_true(audited)
})
