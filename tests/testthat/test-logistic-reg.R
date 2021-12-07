context("Logistic Regression")

iris_scaled <- scale(as.matrix(iris[names(iris) != "Species"]))
y <- iris$Species
subset <- c(1:50, 51:60, 141:150)
iris_subset <- iris_scaled[subset, ]
y_subset <- y[subset]

test_that("logistic regression works as expected", {
  model <- cuda_ml_logistic_reg(iris_scaled, y, max_iters = 100)
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with custom sample weight", {
  sample_weight <- c(rep(1, 50), rep(5, 20))

  model <- cuda_ml_logistic_reg(
    iris_subset, y_subset, max_iters = 100, sample_weight = sample_weight
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with custom class weight", {
  class_weight <- c(setosa = 1, versicolor = 5, virginica = 5)

  model <- cuda_ml_logistic_reg(
    iris_subset, y_subset, max_iters = 100, class_weight = class_weight
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with \"balanced\" class weight", {
  model <- cuda_ml_logistic_reg(
    iris_subset, y_subset, max_iters = 100, class_weight = "balanced"
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with custom sample weight and class weight", {
  sample_weight <- c(rep(1, 50), rep(2, 10), rep(3, 10))
  class_weight <- c(setosa = 1, versicolor = 5 / 2, virginica = 5 / 3)

  model <- cuda_ml_logistic_reg(
    iris_subset, y_subset, max_iters = 100, sample_weight = sample_weight
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with L1 regularization", {
  model <- cuda_ml_logistic_reg(
    iris_scaled, y, max_iters = 100, penalty = "l1", C = 0.5
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with L2 regularization", {
  model <- cuda_ml_logistic_reg(
    iris_scaled, y, max_iters = 100, penalty = "l2", C = 0.5
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})

test_that("logistic regression works as expected with elasticnet regularization", {
  model <- cuda_ml_logistic_reg(
    iris_scaled, y, max_iters = 100, penalty = "elasticnet",
    C = 0.5, l1_ratio = 0.5
  )
  preds <- predict(model, iris_scaled)

  expect_gte(sum(preds$.pred_class == iris$Species), 140)
})
