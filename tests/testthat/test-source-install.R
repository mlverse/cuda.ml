test_that("source installation requires explicit host build inputs", {
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
