#' Train a random forest model.
#'
#' Train a random forest model for classification or regression tasks.
#'
#' @template model-with-numeric-input
#' @template supervised-model-with-numeric-output
#' @template supervised-model-formula-spec
#' @template supervised-model-classification-or-regression-mode
#' @template cuml-log-level
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
#' @return A random forest classifier / regressor object that can be used with
#'   the 'predict' S3 generic to make predictions on new data points.
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
#'   "\n"
#' )
#' @export
cuml_rand_forest <- function(x,
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
    switch(mode,
      classification = "gini",
      regression = "mse"
    )
  )
  split_criterion <- match_split_criterion(split_criterion, mode)
  cuml_log_level <- match_cuml_log_level(cuml_log_level)
  c(x, y) %<-% process_input_and_label_specs(x, y, formula)

  switch(mode,
    classification = {
      new_model(
        cls = "cuml_rand_forest",
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
        resp_var = y
      )
    },
    regression = {
      new_model(
        cls = "cuml_rand_forest",
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
        formula = formula
      )
    }
  )
}

#' @export
predict.cuml_rand_forest <- function(object,
                                     ...,
                                     cuml_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  model <- object
  x <- process_input_specs(rlang::dots_list(...)[[1]], model)
  cuml_log_level <- match_cuml_log_level(cuml_log_level)

  switch(model$mode,
    classification = {
      .rf_classifier_predict(
        model_xptr = model$xptr,
        input = as.matrix(x),
        verbosity = cuml_log_level
      ) %>%
        postprocess_classification_results(model)
    },
    regression = {
      .rf_regressor_predict(
        model_xptr = model$xptr,
        input = as.matrix(x),
        verbosity = cuml_log_level
      )
    }
  )
}
