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

test_that("managed source installation discovers GNU C++ 14", {
  cache <- tempfile("cuda-ml-source-cache-")

  state <- callr::r(
    function(cache) {
      bin <- tempfile("cuda-ml-source-bin-")
      dir.create(bin)
      file.symlink(Sys.which("getconf"), file.path(bin, "getconf"))
      writeLines(c("#!/bin/sh", "echo 13.3.0"), file.path(bin, "g++"))
      writeLines(c("#!/bin/sh", "echo 14.2.0"), file.path(bin, "g++-14"))
      Sys.chmod(file.path(bin, c("g++", "g++-14")), mode = "0755")
      Sys.setenv(
        CUDA_ML_CACHE_DIR = cache,
        CUML_BOOTSTRAP_CACHE = file.path(cache, "bootstrap"),
        PATH = bin
      )
      Sys.unsetenv(
        c(
          "CUDA_HOME",
          "CUDA_VISIBLE_DEVICES",
          "CUML_PREFIX",
          "CUML_CUDA_ARCHITECTURES",
          "CUDA_ML_CXX"
        )
      )
      suppressPackageStartupMessages(library(cuda.ml))
      trace(
        "download.file",
        where = asNamespace("utils"),
        tracer = quote(stop("download reached")),
        print = FALSE
      )

      messages <- character()
      error <- withCallingHandlers(
        tryCatch(
          cuda_ml_install(source = TRUE),
          error = identity
        ),
        message = function(condition) {
          messages <<- c(messages, conditionMessage(condition))
          invokeRestart("muffleMessage")
        }
      )
      list(error = conditionMessage(error), messages = messages)
    },
    args = list(cache = cache)
  )

  expect_match(state$error, "Failed to download and verify", fixed = TRUE)
  expect_false(grepl("GNU C[+][+] 14 or newer", state$error))
  expect_true(any(grepl(
    "No CUDA-visible NVIDIA GPU was detected; using portable CUDA architectures.",
    state$messages,
    fixed = TRUE
  )))
})

test_that("managed source installation detects visible GPUs by default", {
  cache <- tempfile("cuda-ml-source-cache-")

  state <- callr::r(
    function(cache) {
      bin <- tempfile("cuda-ml-source-bin-")
      dir.create(bin)
      file.symlink(Sys.which("getconf"), file.path(bin, "getconf"))
      writeLines(c("#!/bin/sh", "echo 14.2.0"), file.path(bin, "g++-14"))
      writeLines(
        c(
          "#!/bin/sh",
          paste0(
            "printf '0, GPU-aaaa, 8.6\\n1, GPU-bbbb, 7.5\\n",
            "2, GPU-cccc, 8.6\\n3, GPU-dddd, 9.0\\n'"
          )
        ),
        file.path(bin, "nvidia-smi")
      )
      Sys.chmod(file.path(bin, c("g++-14", "nvidia-smi")), mode = "0755")
      Sys.setenv(
        CUDA_ML_CACHE_DIR = cache,
        CUDA_VISIBLE_DEVICES = "GPU-bbbb,GPU-aaaa,GPU-cccc",
        CUML_BOOTSTRAP_CACHE = file.path(cache, "bootstrap"),
        PATH = bin
      )
      Sys.unsetenv(
        c(
          "CUDA_HOME",
          "CUML_PREFIX",
          "CUML_CUDA_ARCHITECTURES",
          "CUDA_ML_CXX"
        )
      )
      suppressPackageStartupMessages(library(cuda.ml))
      trace(
        "download.file",
        where = asNamespace("utils"),
        tracer = quote(stop("download reached")),
        print = FALSE
      )

      messages <- character()
      error <- withCallingHandlers(
        tryCatch(
          cuda_ml_install(source = TRUE),
          error = identity
        ),
        message = function(condition) {
          messages <<- c(messages, conditionMessage(condition))
          invokeRestart("muffleMessage")
        }
      )
      list(error = conditionMessage(error), messages = messages)
    },
    args = list(cache = cache)
  )

  expect_match(state$error, "Failed to download and verify", fixed = TRUE)
  expect_true(any(grepl(
    "Detected CUDA architectures: 75-real, 86-real.",
    state$messages,
    fixed = TRUE
  )))
})

test_that("source installation can force portable architectures", {
  cache <- tempfile("cuda-ml-source-cache-")

  state <- callr::r(
    function(cache) {
      bin <- tempfile("cuda-ml-source-bin-")
      dir.create(bin)
      file.symlink(Sys.which("getconf"), file.path(bin, "getconf"))
      writeLines(c("#!/bin/sh", "echo 14.2.0"), file.path(bin, "g++-14"))
      Sys.chmod(file.path(bin, "g++-14"), mode = "0755")
      Sys.setenv(
        CUDA_ML_CACHE_DIR = cache,
        CUML_BOOTSTRAP_CACHE = file.path(cache, "bootstrap"),
        PATH = bin
      )
      Sys.unsetenv(
        c(
          "CUDA_HOME",
          "CUDA_VISIBLE_DEVICES",
          "CUML_PREFIX",
          "CUML_CUDA_ARCHITECTURES",
          "CUDA_ML_CXX"
        )
      )
      suppressPackageStartupMessages(library(cuda.ml))
      trace(
        "download.file",
        where = asNamespace("utils"),
        tracer = quote(stop("download reached")),
        print = FALSE
      )

      messages <- character()
      error <- withCallingHandlers(
        tryCatch(
          cuda_ml_install(source = TRUE, architectures = "portable"),
          error = identity
        ),
        message = function(condition) {
          messages <<- c(messages, conditionMessage(condition))
          invokeRestart("muffleMessage")
        }
      )
      list(error = conditionMessage(error), messages = messages)
    },
    args = list(cache = cache)
  )

  expect_match(state$error, "Failed to download and verify", fixed = TRUE)
  expect_true(any(grepl(
    "Using portable CUDA architectures.",
    state$messages,
    fixed = TRUE
  )))
})

test_that("native source architecture requires a detectable GPU", {
  cache <- tempfile("cuda-ml-source-cache-")

  state <- callr::r(
    function(cache) {
      bin <- tempfile("cuda-ml-source-bin-")
      dir.create(bin)
      file.symlink(Sys.which("getconf"), file.path(bin, "getconf"))
      writeLines(c("#!/bin/sh", "echo 14.2.0"), file.path(bin, "g++-14"))
      Sys.chmod(file.path(bin, "g++-14"), mode = "0755")
      Sys.setenv(CUDA_ML_CACHE_DIR = cache, PATH = bin)
      Sys.unsetenv(
        c(
          "CUDA_HOME",
          "CUDA_VISIBLE_DEVICES",
          "CUML_PREFIX",
          "CUML_CUDA_ARCHITECTURES",
          "CUDA_ML_CXX"
        )
      )
      suppressPackageStartupMessages(library(cuda.ml))

      error <- tryCatch(
        cuda_ml_install(source = TRUE, architectures = "native"),
        error = identity
      )
      list(
        error = conditionMessage(error),
        cache_exists = dir.exists(cache)
      )
    },
    args = list(cache = cache)
  )

  expect_match(
    state$error,
    'Unable to detect architectures = "native" with nvidia-smi.',
    fixed = TRUE
  )
  expect_match(state$error, 'architectures = "86-real"', fixed = TRUE)
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
