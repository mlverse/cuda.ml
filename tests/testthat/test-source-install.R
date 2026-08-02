test_that("managed source installation only requires a host compiler", {
  cache <- tempfile("cuda-ml-source-cache-")

  state <- callr::r(
    function(cache) {
      bin <- tempfile("cuda-ml-source-bin-")
      dir.create(bin)
      file.symlink(Sys.which("getconf"), file.path(bin, "getconf"))
      file.create(file.path(bin, "apt"))
      Sys.chmod(file.path(bin, "apt"), mode = "0755")
      Sys.setenv(
        CUDA_ML_CACHE_DIR = cache,
        CUDA_ML_CXX = file.path(cache, "missing-g++"),
        PATH = bin
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
  expect_match(
    state$error,
    "nvForest 26.06 binary uses GCC 14 C++ symbol mangling",
    fixed = TRUE
  )
  expect_match(
    state$error,
    "sudo apt update && sudo apt install g++-14",
    fixed = TRUE
  )
  expect_match(
    state$error,
    'Sys.setenv(CUDA_ML_CXX = "/usr/bin/g++-14")',
    fixed = TRUE
  )
  expect_false(grepl("CUDA_HOME|CUML_PREFIX", state$error))
  expect_false(state$cache_exists)
})

test_that("managed source installation identifies an unselected compiler", {
  cache <- tempfile("cuda-ml-source-cache-")

  state <- callr::r(
    function(cache) {
      bin <- tempfile("cuda-ml-source-bin-")
      dir.create(bin)
      file.symlink(Sys.which("getconf"), file.path(bin, "getconf"))
      cxx <- file.path(bin, "g++-14")
      writeLines(c("#!/bin/sh", "echo 14.2.0"), cxx)
      Sys.chmod(cxx, mode = "0755")
      Sys.setenv(
        CUDA_ML_CACHE_DIR = cache,
        CUDA_ML_CXX = file.path(cache, "missing-g++"),
        PATH = bin
      )
      suppressPackageStartupMessages(library(cuda.ml))

      error <- tryCatch(
        cuda_ml_install(source = TRUE),
        error = identity
      )
      conditionMessage(error)
    },
    args = list(cache = cache)
  )

  expect_match(state, "GNU C[+][+] 14 or newer")
  expect_match(state, "g++-14 is installed but not selected", fixed = TRUE)
  expect_match(state, "Sys.setenv(CUDA_ML_CXX", fixed = TRUE)
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
