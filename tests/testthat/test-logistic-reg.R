skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("Logistic Regression")

penguins_scaled <- scale(as.matrix(penguins[penguin_predictors]))
y <- penguins$species
subset_sizes <- c(Adelie = 50L, Chinstrap = 10L, Gentoo = 10L)
subset_rows <- unlist(Map(
  \(level, size) {
    rows <- which(y == level)
    rows[round(seq(1, length(rows), length.out = size))]
  },
  names(subset_sizes),
  subset_sizes
))
penguins_subset <- penguins_scaled[subset_rows, ]
y_subset <- y[subset_rows]
minimum_accuracy <- 14 / 15

test_that("logistic regression works as expected", {
  model <- cuda_ml_logistic_reg(penguins_scaled, y, max_iter = 100)
  preds <- predict(model, penguins_scaled, type = "class")

  expect_gte(mean(preds$.pred_class == penguins$species), minimum_accuracy)
})

test_that("logistic regression is silent by default", {
  expect_silent({
    model <- cuda_ml_logistic_reg(penguins_scaled, y, max_iter = 2)
    predict(model, penguins_scaled, type = "class")
  })
})

test_that("multinomial regression returns probabilities for every class", {
  model <- cuda_ml_logistic_reg(penguins_scaled, y, max_iter = 100)
  classes <- predict(model, penguins_scaled, type = "class")
  probabilities <- predict(model, penguins_scaled, type = "prob")

  expect_named(probabilities, paste0(".pred_", levels(y)))
  expect_equal(rowSums(probabilities), rep(1, nrow(penguins_scaled)))
  expect_identical(
    max.col(as.matrix(probabilities)),
    as.integer(classes$.pred_class)
  )
  expect_gte(
    mean(levels(y)[max.col(as.matrix(probabilities))] == y),
    minimum_accuracy
  )
})

test_that("logistic regression works as expected with custom sample weight", {
  class_counts <- table(y_subset)
  sample_weight <- as.numeric(max(class_counts) / class_counts[y_subset])

  model <- cuda_ml_logistic_reg(
    penguins_subset,
    y_subset,
    max_iter = 100,
    sample_weight = sample_weight
  )
  preds <- predict(model, penguins_scaled)

  expect_gte(mean(preds$.pred_class == penguins$species), minimum_accuracy)
})

test_that("logistic regression works as expected with custom class weight", {
  class_counts <- table(y_subset)
  class_weight <- max(class_counts) / class_counts

  model <- cuda_ml_logistic_reg(
    penguins_subset,
    y_subset,
    max_iter = 100,
    class_weight = class_weight
  )
  preds <- predict(model, penguins_scaled)

  expect_gte(mean(preds$.pred_class == penguins$species), minimum_accuracy)
})

test_that("logistic regression works as expected with \"balanced\" class weight", {
  model <- cuda_ml_logistic_reg(
    penguins_subset,
    y_subset,
    max_iter = 100,
    class_weight = "balanced"
  )
  preds <- predict(model, penguins_scaled)

  expect_gte(mean(preds$.pred_class == penguins$species), minimum_accuracy)
})

test_that("logistic regression works as expected with custom sample weight and class weight", {
  sample_weight_by_class <- c(Adelie = 1, Chinstrap = 2, Gentoo = 3)
  sample_weight <- unname(sample_weight_by_class[as.character(y_subset)])
  weighted_class_counts <- tapply(sample_weight, y_subset, sum)
  class_weight <- max(weighted_class_counts) / weighted_class_counts

  model <- cuda_ml_logistic_reg(
    penguins_subset,
    y_subset,
    max_iter = 100,
    sample_weight = sample_weight,
    class_weight = class_weight
  )
  preds <- predict(model, penguins_scaled)

  expect_gte(mean(preds$.pred_class == penguins$species), minimum_accuracy)
})

test_that("logistic regression works as expected with L1 regularization", {
  model <- cuda_ml_logistic_reg(
    penguins_scaled,
    y,
    max_iter = 100,
    penalty = 2,
    mixture = 1
  )
  preds <- predict(model, penguins_scaled)

  expect_gte(mean(preds$.pred_class == penguins$species), minimum_accuracy)
})

test_that("logistic regression works as expected with L2 regularization", {
  model <- cuda_ml_logistic_reg(
    penguins_scaled,
    y,
    max_iter = 100,
    penalty = 2,
    mixture = 0
  )
  preds <- predict(model, penguins_scaled)

  expect_gte(mean(preds$.pred_class == penguins$species), minimum_accuracy)
})

test_that("logistic regression works as expected with elasticnet regularization", {
  model <- cuda_ml_logistic_reg(
    penguins_scaled,
    y,
    max_iter = 100,
    penalty = 2,
    mixture = 0.5
  )
  preds <- predict(model, penguins_scaled)

  expect_gte(mean(preds$.pred_class == penguins$species), minimum_accuracy)
})

test_that("logistic_reg uses the cuda.ml engine", {
  skip_if_not_installed("parsnip")

  data <- penguins[penguins$species != "Gentoo", ]
  data$species <- droplevels(data$species)
  specification <- parsnip::set_engine(
    parsnip::logistic_reg(penalty = 0.01, mixture = 0),
    "cuda.ml"
  )
  model <- parsnip::fit(specification, species ~ ., data = data)

  classes <- predict(model, data, type = "class")
  probabilities <- predict(model, data, type = "prob")

  expect_named(classes, ".pred_class")
  expect_named(probabilities, paste0(".pred_", levels(data$species)))
})

test_that("multinom_reg uses the cuda.ml engine", {
  skip_if_not_installed("parsnip")

  specification <- parsnip::set_engine(
    parsnip::multinom_reg(penalty = 0.01, mixture = 0.5),
    "cuda.ml"
  )
  model <- parsnip::fit(specification, species ~ ., data = penguins)

  probabilities <- predict(model, penguins, type = "prob")

  expect_named(probabilities, paste0(".pred_", levels(penguins$species)))
})
