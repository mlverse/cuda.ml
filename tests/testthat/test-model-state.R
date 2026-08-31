test_that("unversioned model states are rejected", {
  state <- structure(list(), class = "cuda_ml_pca_model_state")

  expect_error(
    cuda_ml_unserialize(serialize(state, NULL)),
    "Unversioned"
  )
})

linear_state_fixture <- function() {
  readRDS(test_path("fixtures", "linear-model-state-schema-1.rds"))
}

current_linear_state <- function() {
  state <- unserialize(linear_state_fixture())
  state$package_version <- as.character(utils::packageVersion("cuda.ml"))
  state
}

test_that("model states round trip through compressed file paths", {
  model <- cuda_ml_unserialize(linear_state_fixture())
  path <- tempfile(fileext = ".cuda-ml-state")
  on.exit(unlink(path))

  expect_null(cuda_ml_serialize(model, path, xdr = FALSE))
  expect_identical(readBin(path, "raw", n = 2L), as.raw(c(0x1f, 0x8b)))

  restored <- cuda_ml_unserialize(path)
  expect_s3_class(restored, "cuda_ml_ols")
  expect_identical(cuda_ml_serialize(restored), cuda_ml_serialize(model))
})

test_that("file paths read states written to caller-owned connections", {
  model <- cuda_ml_unserialize(linear_state_fixture())
  path <- tempfile(fileext = ".cuda-ml-state")
  on.exit(unlink(path))

  local({
    connection <- file(path, open = "wb")
    on.exit(close(connection))

    expect_null(cuda_ml_serialize(model, connection))
    expect_true(isOpen(connection))
  })

  restored <- cuda_ml_unserialize(path)
  expect_s3_class(restored, "cuda_ml_ols")
  expect_identical(cuda_ml_serialize(restored), cuda_ml_serialize(model))
})

test_that("model-state file paths are one nonempty string", {
  model <- cuda_ml_unserialize(linear_state_fixture())

  expect_error(
    cuda_ml_serialize(model, character()),
    "one nonempty file path"
  )
  expect_error(
    cuda_ml_unserialize(c("first", "second")),
    "one nonempty file path"
  )
})

test_that("a compatible state restores across package versions", {
  state <- unserialize(linear_state_fixture())
  state$package_version <- "0.4.0"

  expect_false(identical(
    state$package_version,
    as.character(utils::packageVersion("cuda.ml"))
  ))

  model <- cuda_ml_unserialize(serialize(state, NULL))
  expect_s3_class(model, "cuda_ml_ols")

  skip_if_not(run_gpu_tests, "requires the GPU test environment")
  expected <- unname(stats::predict(stats::lm(mpg ~ wt, mtcars), mtcars))

  expect_equal(predict(model, mtcars)$.pred, expected, tolerance = 1e-6)
})

test_that("schema, ABI, provenance, and payload errors are actionable", {
  state <- current_linear_state()
  state$schema <- 2L
  expect_error(
    cuda_ml_unserialize(serialize(state, NULL)),
    "schema 2.*supports schema 1"
  )

  state <- current_linear_state()
  state$model_abi <- "cuda_ml_linear_model_state_v2"
  expect_error(
    cuda_ml_unserialize(serialize(state, NULL)),
    "ABI.*cuda_ml_linear_model_state_v2.*not supported"
  )

  state <- current_linear_state()
  state$model_abi <- "cuda_ml_logistic_reg_model_state"
  expect_error(
    cuda_ml_unserialize(serialize(state, NULL)),
    "incompatible.*expected `cuda_ml_linear_model_state`"
  )

  state <- current_linear_state()
  state$package_version <- NULL
  expect_error(
    cuda_ml_unserialize(serialize(state, NULL)),
    "package-version provenance is missing"
  )

  state <- current_linear_state()
  state$payload <- NULL
  expect_error(
    cuda_ml_unserialize(serialize(state, NULL)),
    "payload is missing"
  )
})

test_that("pure-R states do not require a matching backend identity", {
  state <- unserialize(linear_state_fixture())
  state$backend[] <- "different provenance"

  model <- cuda_ml_unserialize(serialize(state, NULL))
  expect_s3_class(model, "cuda_ml_ols")

  skip_if_not(run_gpu_tests, "requires the GPU test environment")
  expected <- unname(stats::predict(stats::lm(mpg ~ wt, mtcars), mtcars))

  expect_equal(predict(model, mtcars)$.pred, expected, tolerance = 1e-6)
})

test_that("duplicate serialization aliases are not exported", {
  exports <- getNamespaceExports("cuda.ml")

  expect_false("cuda_ml_serialise" %in% exports)
  expect_false("cuda_ml_unserialise" %in% exports)
})

test_that("linear and logistic models use explicit portable states", {
  skip_if_not(run_gpu_tests, "requires the GPU test environment")

  linear <- cuda_ml_ols(mpg ~ ., mtcars)
  binary <- penguins[penguins$species != "Gentoo", ]
  binary$species <- droplevels(binary$species)
  logistic <- cuda_ml_logistic_reg(species ~ ., binary)

  restored_linear <- cuda_ml_unserialize(cuda_ml_serialize(linear))
  restored_logistic <- cuda_ml_unserialize(cuda_ml_serialize(logistic))
  saved_linear <- predict_saved_models_in_sub_proc(linear, mtcars)

  expect_equal(predict(restored_linear, mtcars), predict(linear, mtcars))
  expect_equal(saved_linear$restored, predict(linear, mtcars))
  expect_equal(saved_linear$unbundled, predict(linear, mtcars))
  expect_equal(
    predict(restored_logistic, binary, type = "prob"),
    predict(logistic, binary, type = "prob")
  )
})

test_that("bundle restores models through their versioned state", {
  skip_if_not(run_gpu_tests, "requires the GPU test environment")

  model <- cuda_ml_ols(mpg ~ ., mtcars)
  bundled <- bundle::bundle(model)
  restored <- bundle::unbundle(bundled)

  expect_equal(predict(restored, mtcars), predict(model, mtcars))
})
