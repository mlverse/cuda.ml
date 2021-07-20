new_rand_forest_model <- function(mode, xptr, formula = NULL, labels = NULL) {
  structure(list(mode = mode, xptr = xptr, formula = formula, labels = labels), class = "cuml_rand_forest")
}

#' Train a random forest model.
#'
#' Train a random forest model for classification or regression tasks.
#'
#' @inheritParams model-with-numeric-input
#' @inheritParams supervised-model-with-numeric-output
#' @inheritParams supervised-model-formula-spec
#' @inheritParams supervised-model-classification-or-regression-mode
#' @inheritParams cuml-log-level
#' @param mtry The number of predictors that will be randomly sampled at each
#'   split when creating the tree models. Default: the square root of the total
#'   number of predictors.
#' @param trees An integer for the number of trees contained in the ensemble.
#'   Default: 100.
#' @param min_n An integer for the minimum number of data points in a node that
#'   are required for the node to be split further. Default: 2.
#' @param bootstrap Whether to perform bootstrap.
#'   If TRUE, each tree in the forest is built on a bootstrapped sample with
#'   replacement.
#'   If FALSE, the whole dataset is used to build each tree.
#' @param max_depth Maximum tree depth. Default: 16.
#' @param max_leaves Maximum leaf nodes per tree. Soft constraint. Default: -1
#'   (unlimited).
#' @param max_predictors_per_note_split Number of predictor to consider per node
#'   split. Default: square root of the total number predictors.
#' @param n_bins Number of bins used by the split algorithm. Default: 128.
#' @param min_samples_leaf The minimum number of data points in each leaf node.
#'   Default: 1.
#' @param split_criterion The criterion used to split nodes, can be "gini" or
#'   "entropy" for classifications, and "mse" or "mae" for regressions.
#'   Default: "gini" for classification; "mse" for regression.
#' @param min_impurity_decrease Minimum decrease in impurity requried for node
#'   to be spilt. Default: 0.
#' @param max_batch_size Maximum number of nodes that can be processed in a
#'   given batch. Default: 128.
#' @param n_streams Number of CUDA streams to use for building trees.
#'   Default: 8.
#'
#' @return A random forest model object.
#'
#' @examples
#' library(cuml4r)
#'
#' # Classification
#'
#' model <- cuml_rand_forest(
#'   iris,
#'   formula = Species ~ .,
#'   mode = "classification",
#'   trees = 100
#' )
#'
#' predictions <- predict(model, iris)
#'
#' print(predictions)
#'
#' cat(
#'   "Number of correct predictions: ",
#'   sum(predictions == iris[, "Species"]),
#'   "\n"
#' )
#'
#' # Regression
#'
#' model <- cuml_rand_forest(
#'   iris,
#'   formula = Species ~ .,
#'   mode = "regression",
#'   trees = 100
#' )
#'
#' predictions <- predict(model, iris)
#'
#' print(predictions)
#' print(round(predictions))
#'
#' cat(
#'   "Number of correct predictions: ",
#'   sum(as.integer(round(predictions)) == as.integer(iris[, "Species"])),
#'  "\n"
#' )
#'
#' @export
cuml_rand_forest <- function(
                             x,
                             y = NULL,
                             formula = NULL,
                             mode = c("classification", "regression"),
                             mtry = NULL,
                             trees = NULL,
                             min_n = NULL,
                             bootstrap = TRUE,
                             max_depth = 16,
                             max_leaves = -1,
                             max_predictors_per_note_split = NULL,
                             n_bins = 128,
                             min_samples_leaf = 1,
                             split_criterion = NULL,
                             min_impurity_decrease = 0,
                             max_batch_size = 128,
                             n_streams = 8,
                             cuml_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  mode <- match.arg(mode)
  split_criterion <- split_criterion %||% (
    switch(
      mode,
      classification = "gini",
      regression = "mse"
    )
  )
  split_criterion <- match_split_criterion(split_criterion, mode)
  cuml_log_level <- match_cuml_log_level(cuml_log_level)

  if (!is.null(formula)) {
    if (!inherits(x, "data.frame")) {
      stop("'x' must be a data.frame when predictor column(s) and response ",
           "column are specified using the formula syntax.")
    }
    response_col <- all.vars(formula)[[1]]
    predictor_cols <- labels(terms(formula, data = x))
    y <- x[, response_col]
    x <- x[, which(names(x) %in% predictor_cols)]
  } else if (!is.numeric(y)) {
    stop("'y' must be a numeric vector if predictor(s) and responses are not",
         " specified using the formula syntax.")
  }

  switch(
    mode,
    classification = {
      new_rand_forest_model(
        mode = mode,
        xptr = .rf_classifier_fit(
          input = as.matrix(x),
          labels = as.integer(y),
          n_trees = as.integer(trees),
          bootstrap = as.logical(bootstrap),
          max_samples = as.numeric(mtry %||% sqrt(ncol(x))) / ncol(x),
          n_streams = as.integer(n_streams),
          max_depth = as.integer(max_depth),
          max_leaves = as.integer(max_leaves),
          max_features = as.numeric(max_predictors_per_note_split %||% sqrt(ncol(x))) / ncol(x),
          n_bins = as.integer(n_bins),
          min_samples_leaf = as.integer(min_samples_leaf),
          min_samples_split = as.integer(min_n %||% 2L),
          split_criterion = split_criterion,
          min_impurity_decrease = as.numeric(min_impurity_decrease),
          max_batch_size = as.integer(max_batch_size),
          verbosity = cuml_log_level
        ),
        formula = formula,
        labels = levels(y)
      )
    },
    regression = {
      new_rand_forest_model(
        mode = mode,
        xptr = .rf_regressor_fit(
          input = as.matrix(x),
          responses = as.numeric(y),
          n_trees = as.integer(trees),
          bootstrap = as.logical(bootstrap),
          max_samples = as.numeric(mtry %||% sqrt(ncol(x))) / ncol(x),
          n_streams = as.integer(n_streams),
          max_depth = as.integer(max_depth),
          max_leaves = as.integer(max_leaves),
          max_features = as.numeric(max_predictors_per_note_split %||% sqrt(ncol(x))) / ncol(x),
          n_bins = as.integer(n_bins),
          min_samples_leaf = as.integer(min_samples_leaf),
          min_samples_split = as.integer(min_n %||% 2L),
          split_criterion = split_criterion,
          min_impurity_decrease = as.numeric(min_impurity_decrease),
          max_batch_size = as.integer(max_batch_size),
          verbosity = cuml_log_level
        ),
        formula = formula,
        labels = NULL
      )
    }
  )
}

#' Predict using a random forest model.
#'
#' Perform classification or regression tasks using a trained random forest
#' model.
#'
#' @param model A random forest model object.
#' @param x The input matrix or dataframe. Each data point should be a row and
#'   should consist of numeric values only.
#' @param cuml_log_level Log level within cuML library functions. Must be one of
#'   {"off", "critical", "error", "warn", "info", "debug", "trace"}.
#'   Default: off.
#'
#' @examples
#' library(cuml4r)
#'
#' model <- cuml_rand_forest(
#'   iris,
#'   formula = Species ~ .,
#'   trees = 100
#' )
#'
#' predictions <- predict(model, iris)
#'
#' print(predictions)
#'
#' @export
predict.cuml_rand_forest <- function(
                                     model,
                                     x,
                                     cuml_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  cuml_log_level <- match_cuml_log_level(cuml_log_level)

  if (!is.null(model$formula)) {
    predictor_cols <- labels(terms(model$formula, data = x))
    x <- x[, which(names(x) %in% predictor_cols)]
  }

  switch (
    model$mode,
    classification = {
      predictions <- .rf_classifier_predict(
        model_xptr = model$xptr,
        input = as.matrix(x),
        verbosity = cuml_log_level
      )
      if (!is.null(model$labels)) {
        predictions <- factor(
          predictions, levels = seq_along(model$labels), labels = model$labels
        )
      }
    },
    regression = {
      predictions <- .rf_regressor_predict(
        model_xptr = model$xptr,
        input = as.matrix(x),
        verbosity = cuml_log_level
      )
    }
  )

  predictions
}
