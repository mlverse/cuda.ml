#' Train a random forest model.
#'
#' Train a random forest model for classification or regression tasks.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @template cuML-log-level
#' @param mtry The number of predictors that will be randomly sampled at each
#'   split when creating the tree models. Default: the square root of the total
#'   number of predictors.
#' @param trees An integer for the number of trees contained in the ensemble.
#'   Default: 100L.
#' @param min_n An integer for the minimum number of data points in a node that
#'   are required for the node to be split further. Default: 2L.
#' @param bootstrap Whether to perform bootstrap.
#'   If TRUE, each tree in the forest is built on a bootstrapped sample with
#'   replacement.
#'   If FALSE, the whole dataset is used to build each tree.
#' @param max_depth Maximum tree depth. Default: 16L.
#' @param max_leaves Maximum leaf nodes per tree. Soft constraint. Default: Inf
#'   (unlimited).
#' @param max_predictors_per_note_split Number of predictor to consider per node
#'   split. Default: square root of the total number predictors.
#' @param n_bins Number of bins used by the split algorithm. Default: 128L.
#' @param min_samples_leaf The minimum number of data points in each leaf node.
#'   Default: 1L.
#' @param split_criterion The criterion used to split nodes, can be "gini" or
#'   "entropy" for classifications, and "mse" or "mae" for regressions.
#'   Default: "gini" for classification; "mse" for regression.
#' @param min_impurity_decrease Minimum decrease in impurity requried for node
#'   to be spilt. Default: 0.
#' @param max_batch_size Maximum number of nodes that can be processed in a
#'   given batch. Default: 128L.
#' @param n_streams Number of CUDA streams to use for building trees.
#'   Default: 8L.
#'
#' @return A random forest classifier / regressor object that can be used with
#'   the 'predict' S3 generic to make predictions on new data points.
#'
#' @examples
#' library(cuda.ml)
#'
#' # Classification
#'
#' model <- cuda_ml_rand_forest(
#'   formula = Species ~ .,
#'   data = iris,
#'   trees = 100
#' )
#'
#' predictions <- predict(model, iris[names(iris) != "Species"])
#'
#' # Regression
#'
#' model <- cuda_ml_rand_forest(
#'   formula = mpg ~ .,
#'   data = mtcars,
#'   trees = 100
#' )
#'
#' predictions <- predict(model, mtcars[names(mtcars) != "mpg"])
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_rand_forest <- function(x, ...) {
  check_dots_used()
  UseMethod("cuda_ml_rand_forest")
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_rand_forest", x)
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.data.frame <- function(x, y, mtry = NULL, trees = NULL,
                                           min_n = 2L, bootstrap = TRUE,
                                           max_depth = 16L, max_leaves = Inf,
                                           max_predictors_per_note_split = NULL,
                                           n_bins = 128L, min_samples_leaf = 1L,
                                           split_criterion = NULL,
                                           min_impurity_decrease = 0,
                                           max_batch_size = 128L, n_streams = 8L,
                                           cuML_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace"),
                                           ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_rand_forest_bridge(
    processed = processed,
    mtry = mtry,
    trees = trees,
    min_n = min_n,
    bootstrap = bootstrap,
    max_depth = max_depth,
    max_leaves = max_leaves,
    max_predictors_per_note_split = max_predictors_per_note_split,
    n_bins = n_bins,
    min_samples_leaf = min_samples_leaf,
    split_criterion = split_criterion,
    min_impurity_decrease = min_impurity_decrease,
    max_batch_size = max_batch_size,
    n_streams = n_streams,
    cuML_log_level = cuML_log_level
  )
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.matrix <- function(x, y, mtry = NULL, trees = NULL, min_n = 2L,
                                       bootstrap = TRUE, max_depth = 16L,
                                       max_leaves = Inf,
                                       max_predictors_per_note_split = NULL,
                                       n_bins = 128L, min_samples_leaf = 1L,
                                       split_criterion = NULL,
                                       min_impurity_decrease = 0,
                                       max_batch_size = 128L, n_streams = 8L,
                                       cuML_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace"),
                                       ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_rand_forest_bridge(
    processed = processed,
    mtry = mtry,
    trees = trees,
    min_n = min_n,
    bootstrap = bootstrap,
    max_depth = max_depth,
    max_leaves = max_leaves,
    max_predictors_per_note_split = max_predictors_per_note_split,
    n_bins = n_bins,
    min_samples_leaf = min_samples_leaf,
    split_criterion = split_criterion,
    min_impurity_decrease = min_impurity_decrease,
    max_batch_size = max_batch_size,
    n_streams = n_streams,
    cuML_log_level = cuML_log_level
  )
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.formula <- function(formula, data, mtry = NULL, trees = NULL,
                                        min_n = 2L, bootstrap = TRUE,
                                        max_depth = 16L, max_leaves = Inf,
                                        max_predictors_per_note_split = NULL,
                                        n_bins = 128L, min_samples_leaf = 1L,
                                        split_criterion = NULL,
                                        min_impurity_decrease = 0,
                                        max_batch_size = 128L, n_streams = 8L,
                                        cuML_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace"),
                                        ...) {
  processed <- hardhat::mold(formula, data)

  cuda_ml_rand_forest_bridge(
    processed = processed,
    mtry = mtry,
    trees = trees,
    min_n = min_n,
    bootstrap = bootstrap,
    max_depth = max_depth,
    max_leaves = max_leaves,
    max_predictors_per_note_split = max_predictors_per_note_split,
    n_bins = n_bins,
    min_samples_leaf = min_samples_leaf,
    split_criterion = split_criterion,
    min_impurity_decrease = min_impurity_decrease,
    max_batch_size = max_batch_size,
    n_streams = n_streams,
    cuML_log_level = cuML_log_level
  )
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.recipe <- function(x, data, mtry = NULL, trees = NULL,
                                       min_n = 2L, bootstrap = TRUE,
                                       max_depth = 16L, max_leaves = Inf,
                                       max_predictors_per_note_split = NULL,
                                       n_bins = 128L, min_samples_leaf = 1L,
                                       split_criterion = NULL,
                                       min_impurity_decrease = 0,
                                       max_batch_size = 128L, n_streams = 8L,
                                       cuML_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace"),
                                       ...) {
  processed <- hardhat::mold(x, data)

  cuda_ml_rand_forest_bridge(
    processed = processed,
    mtry = mtry,
    trees = trees,
    min_n = min_n,
    bootstrap = bootstrap,
    max_depth = max_depth,
    max_leaves = max_leaves,
    max_predictors_per_note_split = max_predictors_per_note_split,
    n_bins = n_bins,
    min_samples_leaf = min_samples_leaf,
    split_criterion = split_criterion,
    min_impurity_decrease = min_impurity_decrease,
    max_batch_size = max_batch_size,
    n_streams = n_streams,
    cuML_log_level = cuML_log_level
  )
}

cuda_ml_rand_forest_bridge <- function(processed, mtry, trees, min_n, bootstrap,
                                       max_depth, max_leaves,
                                       max_predictors_per_note_split, n_bins,
                                       min_samples_leaf, split_criterion,
                                       min_impurity_decrease, max_batch_size,
                                       n_streams, cuML_log_level) {
  hardhat::validate_predictors_are_numeric(processed$predictors)
  hardhat::validate_outcomes_are_univariate(processed$outcomes)
  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]
  classification <- is.factor(y)
  if (identical(max_leaves, Inf)) {
    max_leaves <- -1L
  }

  # Default value for 'split_criterion' depends on whether a classification or a
  # regression task is being performed.
  split_criterion <- decision_tree_match_split_criterion(
    split_criterion,
    classification
  )
  cuML_log_level <- match_cuML_log_level(cuML_log_level)

  rand_forest_fit_impl <- ifelse(
    classification,
    cuda_ml_rand_forest_impl_classification,
    cuda_ml_rand_forest_impl_regression
  )

  rand_forest_fit_impl(
    processed = processed,
    mtry = mtry,
    trees = trees,
    min_n = min_n,
    bootstrap = bootstrap,
    max_depth = max_depth,
    max_leaves = max_leaves,
    max_predictors_per_note_split = max_predictors_per_note_split,
    n_bins = n_bins,
    min_samples_leaf = min_samples_leaf,
    split_criterion = split_criterion,
    min_impurity_decrease = min_impurity_decrease,
    max_batch_size = max_batch_size,
    n_streams = n_streams,
    cuML_log_level = cuML_log_level
  )
}

cuda_ml_rand_forest_impl_classification <- function(processed, mtry, trees, min_n,
                                                    bootstrap, max_depth,
                                                    max_leaves,
                                                    max_predictors_per_note_split,
                                                    n_bins, min_samples_leaf,
                                                    split_criterion,
                                                    min_impurity_decrease,
                                                    max_batch_size, n_streams,
                                                    cuML_log_level) {
  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  model_xptr <- .rf_classifier_fit(
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
    verbosity = cuML_log_level
  )

  new_model(
    cls = "cuda_ml_rand_forest",
    mode = "classification",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}

cuda_ml_rand_forest_impl_regression <- function(processed, mtry, trees, min_n,
                                                bootstrap, max_depth, max_leaves,
                                                max_predictors_per_note_split,
                                                n_bins, min_samples_leaf,
                                                split_criterion,
                                                min_impurity_decrease,
                                                max_batch_size, n_streams,
                                                cuML_log_level) {
  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  model_xptr <- .rf_regressor_fit(
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
    verbosity = cuML_log_level
  )
  new_model(
    cls = "cuda_ml_rand_forest",
    mode = "regression",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}

cuda_ml_get_state.cuda_ml_rand_forest <- function(model) {
  get_state_impl <- switch(model$mode,
    classification = .rf_classifier_get_state,
    regression = .rf_regressor_get_state
  )

  model_state <- list(
    mode = model$mode,
    rf = get_state_impl(model$xptr),
    blueprint = model$blueprint
  )

  new_model_state(model_state, "cuda_ml_rand_forest_model_state")
}

cuda_ml_set_state.cuda_ml_rand_forest_model_state <- function(model_state) {
  set_state_impl <- switch(model_state$mode,
    classification = .rf_classifier_set_state,
    regression = .rf_regressor_set_state
  )

  new_model(
    cls = "cuda_ml_rand_forest",
    mode = model_state$mode,
    xptr = set_state_impl(model_state$rf),
    blueprint = model_state$blueprint
  )
}

#' Make predictions on new data points.
#'
#' Make predictions on new data points using a CuML random forest model.
#'
#' @template predict
#' @template output-class-probabilities
#' @template cuML-log-level
#'
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_rand_forest <- function(object, x,
                                        output_class_probabilities = NULL,
                                        cuML_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace"),
                                        ...) {
  check_dots_used()

  processed <- hardhat::forge(x, object$blueprint)

  predict_cuda_ml_rand_forest_bridge(
    model = object,
    processed = processed,
    output_class_probabilities = output_class_probabilities,
    cuML_log_level = cuML_log_level
  )
}

predict_cuda_ml_rand_forest_bridge <- function(model,
                                               processed,
                                               output_class_probabilities,
                                               cuML_log_level) {
  cuML_log_level <- match_cuML_log_level(cuML_log_level)

  out <- switch(model$mode,
    classification = {
      predict_cuda_ml_rand_forest_classification_impl(
        model = model,
        processed = processed,
        output_class_probabilities = output_class_probabilities %||% FALSE,
        cuML_log_level = cuML_log_level
      )
    },
    regression = {
      if (!is.null(output_class_probabilities)) {
        stop("'output_class_probabilities' is not applicable for regression tasks!")
      }

      predict_cuda_ml_rand_forest_regression_impl(
        model = model,
        processed = processed,
        cuML_log_level = cuML_log_level
      )
    }
  )
  hardhat::validate_prediction_size(out, processed$predictors)

  out
}

predict_cuda_ml_rand_forest_classification_impl <- function(model,
                                                            processed,
                                                            output_class_probabilities,
                                                            cuML_log_level) {
  if (output_class_probabilities) {
    if (as.integer(cuML_major_version()) == 21 &&
      as.integer(cuML_minor_version()) < 8) {
      stop(
        "Class probabilities output for random forest classifier is only ",
        "supported by RAPIDS cuML 21.08 or above. Current version of ",
        "RAPIDS cuML linked with {cuda.ml} is v",
        paste0(cuML_major_version(), ".", cuML_minor_version()), "."
      )
    }

    preds <- .rf_classifier_predict_class_probabilities(
      model_xptr = model$xptr,
      input = as.matrix(processed$predictors)
    )

    postprocess_class_probabilities(preds, model)
  } else {
    preds <- .rf_classifier_predict(
      model_xptr = model$xptr,
      input = as.matrix(processed$predictors),
      verbosity = cuML_log_level
    )

    postprocess_classification_results(preds, model)
  }
}

predict_cuda_ml_rand_forest_regression_impl <- function(model, processed,
                                                        cuML_log_level) {
  preds <- .rf_regressor_predict(
    model_xptr = model$xptr,
    input = as.matrix(processed$predictors),
    verbosity = cuML_log_level
  )

  postprocess_regression_results(preds)
}

# register the CuML-based rand_forest model for parsnip
register_rand_forest_model <- function(pkgname) {
  for (mode in c("classification", "regression")) {
    parsnip::set_model_engine(
      model = "rand_forest", mode = mode, eng = pkgname
    )
  }

  parsnip::set_dependency(model = "rand_forest", eng = pkgname, pkg = pkgname)

  parsnip::set_model_arg(
    model = "rand_forest",
    eng = pkgname,
    parsnip = "mtry",
    original = "mtry",
    func = list(pkg = "dials", fun = "mtry"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "rand_forest",
    eng = pkgname,
    parsnip = "trees",
    original = "trees",
    func = list(pkg = "dials", fun = "trees"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "rand_forest",
    eng = pkgname,
    parsnip = "min_n",
    original = "min_n",
    func = list(pkg = "dials", fun = "min_n"),
    has_submodel = FALSE
  )

  for (mode in c("classification", "regression")) {
    parsnip::set_fit(
      model = "rand_forest",
      eng = pkgname,
      mode = mode,
      value = list(
        interface = "formula",
        protect = c("formula", "data"),
        func = c(pkg = pkgname, fun = "cuda_ml_rand_forest"),
        defaults = list(
          bootstrap = TRUE,
          max_depth = 16L,
          max_leaves = Inf,
          max_predictors_per_note_split = NULL,
          n_bins = 128L,
          min_samples_leaf = 1L,
          split_criterion = NULL,
          min_impurity_decrease = 0,
          max_batch_size = 128L,
          n_streams = 8L,
          cuML_log_level = "off"
        )
      )
    )

    parsnip::set_encoding(
      model = "rand_forest",
      eng = pkgname,
      mode = mode,
      options = list(
        predictor_indicators = "none",
        compute_intercept = FALSE,
        remove_intercept = FALSE,
        allow_sparse_x = TRUE
      )
    )
  }

  for (type in c("class", "prob")) {
    parsnip::set_pred(
      model = "rand_forest",
      eng = pkgname,
      mode = "classification",
      type = type,
      value = list(
        pre = NULL,
        post = NULL,
        func = c(fun = "predict"),
        args = list(
          quote(object$fit),
          quote(new_data),
          identical(type, "prob") # output_class_probabilities
        )
      )
    )
  }

  parsnip::set_pred(
    model = "rand_forest",
    eng = pkgname,
    mode = "regression",
    type = "numeric",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        quote(object$fit),
        quote(new_data)
      )
    )
  )
}
