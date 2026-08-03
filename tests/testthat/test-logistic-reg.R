skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("Logistic Regression")

iris_scaled <- scale(as.matrix(iris[names(iris) != "Species"]))
y <- iris$Species
subset <- c(1:50, 51:60, 141:150)
iris_subset <- iris_scaled[subset, ]
y_subset <- y[subset]

test_that("logistic regression works as expected", {
  model <- cuda_ml_logistic_reg(iris_scaled, y, max_iter = 100)
  preds <- predict(model, iris_scaled, type = "class")

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression is silent by default", {
  expect_silent({
    model <- cuda_ml_logistic_reg(iris_scaled, y, max_iter = 2)
    predict(model, iris_scaled, type = "class")
  })
})

test_that("multinomial regression returns probabilities for every class", {
  model <- cuda_ml_logistic_reg(iris_scaled, y, max_iter = 100)
  classes <- predict(model, iris_scaled, type = "class")
  probabilities <- predict(model, iris_scaled, type = "prob")

  expect_named(probabilities, paste0(".pred_", levels(y)))
  expect_equal(rowSums(probabilities), rep(1, nrow(iris_scaled)))
  expect_identical(
    max.col(as.matrix(probabilities)),
    as.integer(classes$.pred_class)
  )
  expect_gte(sum(levels(y)[max.col(as.matrix(probabilities))] == y), 140)
})

test_that("logistic regression works as expected with custom sample weight", {
  sample_weight <- c(rep(1, 50), rep(5, 20))

  model <- cuda_ml_logistic_reg(
    iris_subset,
    y_subset,
    max_iter = 100,
    sample_weight = sample_weight
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with custom class weight", {
  class_weight <- c(setosa = 1, versicolor = 5, virginica = 5)

  model <- cuda_ml_logistic_reg(
    iris_subset,
    y_subset,
    max_iter = 100,
    class_weight = class_weight
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with \"balanced\" class weight", {
  model <- cuda_ml_logistic_reg(
    iris_subset,
    y_subset,
    max_iter = 100,
    class_weight = "balanced"
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with custom sample weight and class weight", {
  sample_weight <- c(rep(1, 50), rep(2, 10), rep(3, 10))
  class_weight <- c(setosa = 1, versicolor = 5 / 2, virginica = 5 / 3)

  model <- cuda_ml_logistic_reg(
    iris_subset,
    y_subset,
    max_iter = 100,
    sample_weight = sample_weight,
    class_weight = class_weight
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with L1 regularization", {
  model <- cuda_ml_logistic_reg(
    iris_scaled,
    y,
    max_iter = 100,
    penalty = 2,
    mixture = 1
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with L2 regularization", {
  model <- cuda_ml_logistic_reg(
    iris_scaled,
    y,
    max_iter = 100,
    penalty = 2,
    mixture = 0
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with elasticnet regularization", {
  model <- cuda_ml_logistic_reg(
    iris_scaled,
    y,
    max_iter = 100,
    penalty = 2,
    mixture = 0.5
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic_reg uses the cuda.ml engine", {
  skip_if_not_installed("parsnip")

  data <- iris[iris$Species != "virginica", ]
  data$Species <- droplevels(data$Species)
  specification <- parsnip::set_engine(
    parsnip::logistic_reg(penalty = 0.01, mixture = 0),
    "cuda.ml"
  )
  model <- parsnip::fit(specification, Species ~ ., data = data)

  classes <- predict(model, data, type = "class")
  probabilities <- predict(model, data, type = "prob")

  expect_named(classes, ".pred_class")
  expect_named(probabilities, paste0(".pred_", levels(data$Species)))
})

test_that("multinom_reg uses the cuda.ml engine", {
  skip_if_not_installed("parsnip")

  specification <- parsnip::set_engine(
    parsnip::multinom_reg(penalty = 0.01, mixture = 0.5),
    "cuda.ml"
  )
  model <- parsnip::fit(specification, Species ~ ., data = iris)

  probabilities <- predict(model, iris, type = "prob")

  expect_named(probabilities, paste0(".pred_", levels(iris$Species)))
})
