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

test_that("has_cuML is a non-mutating build capability query", {
  cache <- tempfile("cuda-ml-cache-")

  state <- callr::r(
    function(cache) {
      Sys.setenv(CUDA_ML_CACHE_DIR = cache)
      suppressPackageStartupMessages(library(cuda.ml))
      before <- names(getLoadedDLLs())
      value <- has_cuML()

      list(
        value = value,
        cache_exists = dir.exists(cache),
        new_dlls = setdiff(names(getLoadedDLLs()), before)
      )
    },
    args = list(cache = cache)
  )

  expect_type(state$value, "logical")
  expect_length(state$value, 1L)
  expect_false(state$cache_exists)
  expect_identical(state$new_dlls, character())
})

test_that("stub builds direct cuda_ml_install users to R-universe", {
  skip_if(has_cuML(), "requires a stub build")
  skip_if_not(
    identical(unname(Sys.info()[["sysname"]]), "Linux") &&
      unname(Sys.info()[["machine"]]) %in% c("x86_64", "amd64"),
    "requires the managed runtime platform"
  )
  expect_false(has_cuML())

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

test_that("cuda_ml_install rejects unsupported platforms before stub guidance", {
  skip_if(has_cuML(), "requires a stub build")

  supported <- identical(unname(Sys.info()[["sysname"]]), "Linux") &&
    unname(Sys.info()[["machine"]]) %in% c("x86_64", "amd64")
  if (supported) {
    suppressWarnings(
      testthat::with_mocked_bindings(
        expect_error(
          cuda_ml_install(),
          "unsupported-platform sentinel",
          fixed = TRUE
        ),
        cuda_ml_platform = function() {
          stop("unsupported-platform sentinel", call. = FALSE)
        },
        .package = "cuda.ml"
      )
    )
  } else {
    expect_error(cuda_ml_install(), "supports Linux x86_64", fixed = TRUE)
  }
})

test_that("stub version queries retain their public return values", {
  skip_if(has_cuML(), "requires a stub build")

  before <- names(getLoadedDLLs())
  expect_identical(cuML_major_version(), NA_character_)
  expect_identical(cuML_minor_version(), NA_character_)
  expect_false(cuda_ml_fil_enabled())
  expect_false(cuda_ml_rand_proj_enabled())
  expect_identical(setdiff(names(getLoadedDLLs()), before), character())
})
