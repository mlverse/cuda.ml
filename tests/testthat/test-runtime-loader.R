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

test_that("backend metadata is a non-mutating package query", {
  cache <- tempfile("cuda-ml-cache-")

  state <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      suppressPackageStartupMessages(library(cuda.ml))
      before <- names(getLoadedDLLs())
      value <- cuda_ml_backend_info()

      list(
        value = value,
        cache_exists = dir.exists(cache),
        new_dlls = setdiff(names(getLoadedDLLs()), before)
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
      "cuda_version",
      "rapids_version",
      "nvforest_version",
      "treelite_version",
      "platform",
      "minimum_driver",
      "architectures",
      "runtime_installed",
      "runtime_path",
      "backend_loaded"
    ),
    ignore.order = FALSE
  )
  expect_true(state$value$backend %in% c("full", "stub"))
  if (state$value$backend == "full") {
    expect_true(state$value$build_mode %in% c("managed", "local"))
    expect_identical(state$value$cuda_version, "13.2.2")
    expect_identical(state$value$rapids_version, "26.06")
    expect_identical(state$value$nvforest_version, "26.06.0")
    expect_identical(state$value$treelite_version, "4.6.1")
    expect_identical(state$value$platform, "ubuntu-26.04-x86_64")
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
  } else {
    expect_identical(state$value$architectures, character())
  }
  expect_false(state$cache_exists)
  expect_identical(state$new_dlls, character())
  expect_false(state$value$backend_loaded)
})

test_that("cache cleanup is scoped to cuda.ml cache generations", {
  cache <- tempfile("cuda-ml-cache-")
  sentinel <- tempfile("cuda-ml-cache-sentinel-")
  dir.create(file.path(cache, "runtime-v2"), recursive = TRUE)
  dir.create(file.path(cache, "backends-v2"), recursive = TRUE)
  dir.create(sentinel)

  state <- callr::r(
    function(cache, sentinel) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      library(cuda.ml)
      cuda_ml_cache_clean()
      list(
        sentinel = dir.exists(sentinel),
        generations = dir.exists(file.path(
          cache,
          c("runtime-v2", "backends-v2")
        ))
      )
    },
    args = list(cache = cache, sentinel = sentinel)
  )

  expect_true(state$sentinel)
  expect_false(any(state$generations))
})

test_that("cuda_ml_install retries an interrupted locked download", {
  cache <- tempfile("cuda-ml-cache-")
  destination <- tempfile("cuda-ml-download-")
  contents <- charToRaw("verified runtime artifact")
  expected_hash <- digest::digest(
    contents,
    algo = "sha256",
    serialize = FALSE
  )
  attempts <- 0L
  old_cache <- Sys.getenv("CUDA_ML_CACHE_DIR", unset = NA_character_)
  Sys.setenv(CUDA_ML_CACHE_DIR = cache)
  on.exit({
    if (is.na(old_cache)) {
      Sys.unsetenv("CUDA_ML_CACHE_DIR")
    } else {
      Sys.setenv(CUDA_ML_CACHE_DIR = old_cache)
    }
    unlink(c(cache, destination), recursive = TRUE, force = TRUE)
  })

  local_mocked_bindings(
    download.file = function(url, destfile, ...) {
      attempts <<- attempts + 1L
      if (attempts == 1L) {
        writeBin(contents[seq_len(4L)], destfile)
        return(0L)
      }
      writeBin(contents, destfile)
      0L
    },
    .package = "utils"
  )
  local_mocked_bindings(
    cuda_ml_has_backend = function() TRUE,
    cuda_ml_platform = function() "ubuntu-26.04-x86_64",
    cuda_ml_prepare_runtime = function() {
      cuda.ml:::cuda_ml_download(
        "fixture",
        "https://example.invalid/fixture",
        destination,
        length(contents),
        expected_hash
      )
    },
    .package = "cuda.ml"
  )

  expect_true(cuda_ml_install())
  expect_identical(attempts, 2L)
  expect_identical(
    readBin(destination, what = "raw", n = length(contents)),
    contents
  )
})

test_that("stub builds direct cuda_ml_install users to R-universe", {
  skip_if(cuda_ml_backend_info()$backend == "full", "requires a stub build")
  skip_if_not(
    identical(unname(Sys.info()[["sysname"]]), "Linux") &&
      unname(Sys.info()[["machine"]]) %in% c("x86_64", "amd64"),
    "requires the managed runtime platform"
  )
  error <- expect_error(cuda_ml_install(), class = "error")
  r_version <- paste(
    R.version$major,
    strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][[1L]],
    sep = "."
  )

  expect_match(conditionMessage(error), "R-universe", fixed = TRUE)
  expect_match(
    conditionMessage(error),
    "Ubuntu 26.04 (Resolute) x86_64",
    fixed = TRUE
  )
  expect_match(
    conditionMessage(error),
    paste0(
      "https://mlverse.r-universe.dev/bin/linux/resolute-x86_64/",
      r_version,
      "/"
    ),
    fixed = TRUE
  )
})

test_that("stub metadata has no functional backend versions", {
  skip_if(cuda_ml_backend_info()$backend == "full", "requires a stub build")

  before <- names(getLoadedDLLs())
  info <- cuda_ml_backend_info()
  expect_identical(info$backend, "stub")
  expect_identical(info$build_mode, "stub")
  expect_false(info$runtime_installed)
  expect_false(info$backend_loaded)
  expect_identical(setdiff(names(getLoadedDLLs()), before), character())
})
