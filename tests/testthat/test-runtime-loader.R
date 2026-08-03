test_that("attaching cuda.ml is silent and side-effect free", {
  cache <- tempfile("cuda-ml-cache-")

  state <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      messages <- character()
      warnings <- character()
      output <- capture.output(
        withCallingHandlers(
          library(cuda.ml),
          message = function(cnd) {
            messages <<- c(messages, conditionMessage(cnd))
            invokeRestart("muffleMessage")
          },
          warning = function(cnd) {
            warnings <<- c(warnings, conditionMessage(cnd))
            invokeRestart("muffleWarning")
          }
        ),
        type = "output"
      )

      dlls <- getLoadedDLLs()
      list(
        messages = messages,
        warnings = warnings,
        output = output,
        cache_exists = dir.exists(cache),
        dll_loaded = any(vapply(
          dlls,
          function(dll) identical(dll[["name"]], "cuda.ml"),
          logical(1)
        ))
      )
    },
    args = list(cache = cache)
  )

  expect_identical(state$messages, character())
  expect_identical(state$warnings, character())
  expect_identical(state$output, character())
  expect_false(state$cache_exists)
  expect_false(state$dll_loaded)
})

test_that("backend metadata does not provision or load the backend", {
  cache <- tempfile("cuda-ml-cache-")

  state <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      suppressPackageStartupMessages(library(cuda.ml))
      value <- cuda_ml_backend_info()
      dlls <- getLoadedDLLs()

      list(
        value = value,
        cache_exists = dir.exists(cache),
        backend_dll_loaded = any(vapply(
          dlls,
          function(dll) identical(dll[["name"]], "cuda.ml"),
          logical(1)
        ))
      )
    },
    args = list(cache = cache)
  )

  expect_named(
    state$value,
    c(
      "package_version",
      "backend",
      "build_mode",
      "backend_available",
      "r_version",
      "cuda_version",
      "rapids_version",
      "nvforest_version",
      "treelite_version",
      "platform",
      "minimum_glibc",
      "minimum_driver",
      "architectures",
      "runtime_installed",
      "runtime_path",
      "backend_loaded"
    ),
    ignore.order = FALSE
  )
  expect_identical(state$value$backend, "download")
  expect_identical(state$value$build_mode, "release")
  expect_type(state$value$backend_available, "logical")
  expect_length(state$value$backend_available, 1L)
  expect_match(state$value$r_version, "^[0-9]+[.][0-9]+$")
  expect_identical(state$value$cuda_version, "13.2.2")
  expect_identical(state$value$rapids_version, "26.06")
  expect_identical(state$value$nvforest_version, "26.06.0")
  expect_identical(state$value$treelite_version, "4.7.0")
  expect_identical(state$value$platform, "linux-x86_64-glibc2.28")
  expect_identical(state$value$minimum_glibc, "2.28")
  expect_identical(state$value$minimum_driver, 580L)
  expect_identical(
    state$value$architectures,
    c(
      "75-real",
      "80-real",
      "86-real",
      "89-real",
      "90-real",
      "100-real",
      "120-real",
      "120-virtual"
    )
  )
  expect_false(state$cache_exists)
  expect_false(state$backend_dll_loaded)
  expect_false(state$value$backend_loaded)
})

test_that("cache cleanup is scoped to cuda.ml cache generations", {
  cache <- tempfile("cuda-ml-cache-")
  sentinel <- tempfile("cuda-ml-cache-sentinel-")
  generations <- c(
    "runtime-v2",
    "backends-v2",
    "runtime-v3",
    "backend-assets-v1",
    "backends-v3",
    "source-backends-v1",
    "backend-selection-v1",
    "source-toolchains-v1"
  )
  for (generation in generations) {
    dir.create(file.path(cache, generation), recursive = TRUE)
  }
  dir.create(sentinel)

  state <- callr::r(
    function(cache, sentinel, generations) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      library(cuda.ml)
      cuda_ml_cache_clean()
      list(
        sentinel = dir.exists(sentinel),
        generations = dir.exists(file.path(cache, generations))
      )
    },
    args = list(
      cache = cache,
      sentinel = sentinel,
      generations = generations
    )
  )

  expect_true(state$sentinel)
  expect_false(any(state$generations))
})

test_that("an unpublished backend fails before downloading the runtime", {
  skip_if_not(
    native_platform_supported,
    "requires the managed runtime platform"
  )

  cache <- tempfile("cuda-ml-cache-")
  old_cache <- Sys.getenv("CUDA_ML_CACHE_DIR", unset = NA_character_)
  Sys.setenv(CUDA_ML_CACHE_DIR = cache)
  on.exit({
    if (is.na(old_cache)) {
      Sys.unsetenv("CUDA_ML_CACHE_DIR")
    } else {
      Sys.setenv(CUDA_ML_CACHE_DIR = old_cache)
    }
  }, add = TRUE)

  skip_if(cuda_ml_backend_info()$backend_available, "backend is published")
  error <- expect_error(cuda_ml_install(), class = "error")

  expect_match(conditionMessage(error), "No prebuilt cuda.ml backend")
  expect_match(
    conditionMessage(error),
    paste0("R ", cuda_ml_backend_info()$r_version),
    fixed = TRUE
  )
  expect_match(
    conditionMessage(error),
    "linux-x86_64-glibc2.28",
    fixed = TRUE
  )
  expect_false(dir.exists(cache))
})

test_that("an uninstalled downloadable backend remains side-effect free", {
  cache <- tempfile("cuda-ml-cache-")

  state <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      suppressPackageStartupMessages(library(cuda.ml))
      before <- names(getLoadedDLLs())
      info <- cuda_ml_backend_info()

      list(
        info = info,
        backend_dll_loaded = "cuda.ml" %in%
          setdiff(names(getLoadedDLLs()), before)
      )
    },
    args = list(cache = cache)
  )

  expect_identical(state$info$backend, "download")
  expect_identical(state$info$build_mode, "release")
  expect_false(state$info$runtime_installed)
  expect_false(state$info$backend_loaded)
  expect_false(state$backend_dll_loaded)
})
