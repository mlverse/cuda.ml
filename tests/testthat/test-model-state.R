test_that("unversioned model states are rejected", {
  state <- structure(list(), class = "cuda_ml_pca_model_state")

  expect_error(
    cuda_ml_unserialize(serialize(state, NULL)),
    "Unversioned"
  )
})

test_that("duplicate serialization aliases are not exported", {
  exports <- getNamespaceExports("cuda.ml")

  expect_false("cuda_ml_serialise" %in% exports)
  expect_false("cuda_ml_unserialise" %in% exports)
})

test_that("linear and logistic models use explicit portable states", {
  skip_if_not(run_gpu_tests, "requires the GPU test environment")

  linear <- cuda_ml_ols(mpg ~ ., mtcars)
  binary <- iris[iris$Species != "virginica", ]
  binary$Species <- droplevels(binary$Species)
  logistic <- cuda_ml_logistic_reg(Species ~ ., binary)

  restored_linear <- cuda_ml_unserialize(cuda_ml_serialize(linear))
  restored_logistic <- cuda_ml_unserialize(cuda_ml_serialize(logistic))

  expect_equal(predict(restored_linear, mtcars), predict(linear, mtcars))
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
