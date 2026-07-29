test_that("the managed runtime can be prewarmed without loading the backend", {
  skip_if_not(
    identical(Sys.getenv("CUDA_ML_RUNTIME_TESTS"), "true"),
    "set CUDA_ML_RUNTIME_TESTS=true to exercise the managed runtime"
  )
  skip_if_not(has_cuML(), "requires a functional backend")

  cache <- Sys.getenv(
    "CUDA_ML_CACHE_DIR",
    unset = tempfile("cuda-ml-functional-cache-")
  )
  Sys.setenv(CUDA_ML_CACHE_DIR = cache)
  install_runtime <- function(cache) {
    Sys.setenv(CUDA_ML_CACHE_DIR = cache)
    library(cuda.ml)
    cuda_ml_install()
    list(
      dll_loaded = "cuda.ml" %in% names(getLoadedDLLs()),
      cache = list.files(cache, recursive = TRUE, all.files = TRUE)
    )
  }
  installers <- list(
    callr::r_bg(
      install_runtime,
      args = list(cache = cache),
      stdout = "|",
      stderr = "|"
    ),
    callr::r_bg(
      install_runtime,
      args = list(cache = cache),
      stdout = "|",
      stderr = "|"
    )
  )
  on.exit(lapply(installers, function(process) process$kill()), add = TRUE)
  lapply(installers, function(process) process$wait(timeout = 3600000))
  first <- lapply(installers, function(process) process$get_result())

  expect_false(first[[1L]]$dll_loaded)
  expect_false(first[[2L]]$dll_loaded)
  expect_true(any(grepl("[.]complete$", first[[1L]]$cache)))
  expect_true(any(grepl("[.]complete$", first[[2L]]$cache)))
  expect_false(any(grepl(
    "libnvrtc-a49e67e8[.]so[.]13[.]0[.]88$",
    first[[1L]]$cache
  )))
  expect_true(any(grepl(
    "libnvrtc[.]so[.]13$",
    first[[1L]]$cache
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

test_that("version helpers load the prepared RAPIDS 26.06 backend", {
  skip_if_not(
    identical(Sys.getenv("CUDA_ML_RUNTIME_TESTS"), "true"),
    "set CUDA_ML_RUNTIME_TESTS=true to exercise the managed runtime"
  )
  skip_if_not(has_cuML(), "requires a functional backend")

  expect_identical(cuML_major_version(), "26")
  expect_identical(cuML_minor_version(), "6")
  expect_true("cuda.ml" %in% names(getLoadedDLLs()))
})

test_that("R CMD check uses the functional backend for public native calls", {
  skip_if_not(
    identical(Sys.getenv("CUDA_ML_RUNTIME_TESTS"), "true"),
    "set CUDA_ML_RUNTIME_TESTS=true to exercise the managed runtime"
  )
  skip_if_not(has_cuML(), "requires a functional backend")

  cache <- Sys.getenv(
    "CUDA_ML_CACHE_DIR",
    unset = tempfile("cuda-ml-functional-cache-")
  )
  version <- callr::r(
    function(cache) {
      do.call(
        Sys.setenv,
        setNames(
          list(cache, "cuda.ml"),
          c("CUDA_ML_CACHE_DIR", "_R_CHECK_PACKAGE_NAME_")
        )
      )
      library(cuda.ml)
      cuML_major_version()
    },
    args = list(cache = cache)
  )

  expect_identical(version, "26")
})
