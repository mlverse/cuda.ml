skip_if_not(run_gpu_tests, "requires the GPU test environment")

test_that("linear_reg uses cuda.ml penalty and mixture semantics", {
  skip_if_not_installed("parsnip")

  specification <- parsnip::set_engine(
    parsnip::linear_reg(penalty = 1e-3, mixture = 0.5),
    "cuda.ml"
  )
  model <- parsnip::fit(specification, mpg ~ ., data = mtcars)

  predictions <- predict(model, mtcars)

  expect_named(predictions, ".pred")
  expect_equal(nrow(predictions), nrow(mtcars))
})

test_that("regularized regression composes with recipe normalization", {
  recipe <- recipes::recipe(mpg ~ ., data = mtcars) |>
    recipes::step_normalize(recipes::all_numeric_predictors())

  model <- cuda_ml_lasso(recipe, mtcars, alpha = 1e-3)
  predictions <- predict(model, mtcars)

  expect_named(predictions, ".pred")
  expect_equal(nrow(predictions), nrow(mtcars))
})
