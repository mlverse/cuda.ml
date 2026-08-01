test_that("managed source installation only requires a host compiler", {
  cache <- tempfile("cuda-ml-source-cache-")

  state <- callr::r(
    function(cache) {
      Sys.setenv(
        CUDA_ML_CACHE_DIR = cache,
        CUDA_ML_CXX = file.path(cache, "missing-g++")
      )
      Sys.unsetenv(
        c(
          "CUDA_HOME",
          "CUML_PREFIX",
          "CUML_CUDA_ARCHITECTURES"
        )
      )
      suppressPackageStartupMessages(library(cuda.ml))

      error <- tryCatch(
        cuda_ml_install(source = TRUE),
        error = identity
      )
      list(
        error = conditionMessage(error),
        cache_exists = dir.exists(cache)
      )
    },
    args = list(cache = cache)
  )

  expect_match(state$error, "GNU C[+][+] 14 or newer")
  expect_false(grepl("CUDA_HOME|CUML_PREFIX", state$error))
  expect_false(state$cache_exists)
})

test_that("host source installation requires explicit build inputs", {
  cache <- tempfile("cuda-ml-source-cache-")

  state <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      Sys.unsetenv(
        c(
          "CUDA_HOME",
          "CUML_PREFIX",
          "CUML_CUDA_ARCHITECTURES",
          "CUDA_ML_CXX"
        )
      )
      suppressPackageStartupMessages(library(cuda.ml))

      error <- tryCatch(
        cuda_ml_install(source = TRUE, dependencies = "host"),
        error = identity
      )
      list(
        error = conditionMessage(error),
        cache_exists = dir.exists(cache)
      )
    },
    args = list(cache = cache)
  )

  expect_match(state$error, "requires explicit build inputs")
  expect_match(
    state$error,
    paste(
      "CUDA_HOME, CUML_PREFIX, CUML_CUDA_ARCHITECTURES,",
      "and CUDA_ML_CXX"
    ),
    fixed = TRUE
  )
  expect_false(state$cache_exists)
})
