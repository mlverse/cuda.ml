#' Train a random forest model
#'
#' Trains a cuML random forest for classification or regression and returns an
#' nvForest-backed model for inference.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @param mtry Number of predictors sampled at each split. When \code{NULL},
#'   classification uses the square root of the predictor count and regression
#'   uses all predictors.
#' @param trees Number of trees. Default: 100L.
#' @param min_n Minimum observations required to split a node. Default: 2L.
#' @param bootstrap Whether to sample observations with replacement.
#' @param sample_fraction Proportion of rows used for each tree, between 0 and
#'   1. This is separate from \code{mtry}, which controls predictor sampling.
#' @param max_depth Maximum tree depth. Default: 16L.
#' @param max_leaves Maximum leaves per tree, or \code{Inf} for no limit.
#' @param n_bins Number of candidate split bins. Default: 128L.
#' @param min_samples_leaf Minimum observations in a leaf. Default: 1L.
#' @param split_criterion Split criterion, or \code{NULL} for the mode default.
#'   Classification supports \code{"gini"} and \code{"entropy"}; regression
#'   supports \code{"mse"}, \code{"poisson"}, \code{"gamma"}, and
#'   \code{"inverse_gaussian"}.
#' @param min_impurity_decrease Minimum impurity decrease required for a split.
#' @param max_batch_size Maximum nodes processed in one batch. Default: 4096L.
#' @param n_streams Number of CUDA streams used while fitting. Default: 4L.
#' @param seed Random seed forwarded to cuML. When \code{NULL}, a seed is drawn
#'   from R's random-number generator, so \code{set.seed()} controls the fit.
#'
#' @return A random forest model for use with \code{predict()}.
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_rand_forest <- function(x, ...) {
  UseMethod("cuda_ml_rand_forest")
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_rand_forest", x)
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.data.frame <- function(
  x,
  y,
  mtry = NULL,
  trees = 100L,
  min_n = 2L,
  bootstrap = TRUE,
  sample_fraction = 1,
  max_depth = 16L,
  max_leaves = Inf,
  n_bins = 128L,
  min_samples_leaf = 1L,
  split_criterion = NULL,
  min_impurity_decrease = 0,
  max_batch_size = 4096L,
  n_streams = 4L,
  seed = NULL,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)
  cuda_ml_rand_forest_bridge(
    processed,
    mtry,
    trees,
    min_n,
    bootstrap,
    sample_fraction,
    max_depth,
    max_leaves,
    n_bins,
    min_samples_leaf,
    split_criterion,
    min_impurity_decrease,
    max_batch_size,
    n_streams,
    seed
  )
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.matrix <- function(
  x,
  y,
  mtry = NULL,
  trees = 100L,
  min_n = 2L,
  bootstrap = TRUE,
  sample_fraction = 1,
  max_depth = 16L,
  max_leaves = Inf,
  n_bins = 128L,
  min_samples_leaf = 1L,
  split_criterion = NULL,
  min_impurity_decrease = 0,
  max_batch_size = 4096L,
  n_streams = 4L,
  seed = NULL,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)
  cuda_ml_rand_forest_bridge(
    processed,
    mtry,
    trees,
    min_n,
    bootstrap,
    sample_fraction,
    max_depth,
    max_leaves,
    n_bins,
    min_samples_leaf,
    split_criterion,
    min_impurity_decrease,
    max_batch_size,
    n_streams,
    seed
  )
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.formula <- function(
  formula,
  data,
  mtry = NULL,
  trees = 100L,
  min_n = 2L,
  bootstrap = TRUE,
  sample_fraction = 1,
  max_depth = 16L,
  max_leaves = Inf,
  n_bins = 128L,
  min_samples_leaf = 1L,
  split_criterion = NULL,
  min_impurity_decrease = 0,
  max_batch_size = 4096L,
  n_streams = 4L,
  seed = NULL,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(formula, data)
  cuda_ml_rand_forest_bridge(
    processed,
    mtry,
    trees,
    min_n,
    bootstrap,
    sample_fraction,
    max_depth,
    max_leaves,
    n_bins,
    min_samples_leaf,
    split_criterion,
    min_impurity_decrease,
    max_batch_size,
    n_streams,
    seed
  )
}

#' @rdname cuda_ml_rand_forest
#' @export
cuda_ml_rand_forest.recipe <- function(
  x,
  data,
  mtry = NULL,
  trees = 100L,
  min_n = 2L,
  bootstrap = TRUE,
  sample_fraction = 1,
  max_depth = 16L,
  max_leaves = Inf,
  n_bins = 128L,
  min_samples_leaf = 1L,
  split_criterion = NULL,
  min_impurity_decrease = 0,
  max_batch_size = 4096L,
  n_streams = 4L,
  seed = NULL,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, data)
  cuda_ml_rand_forest_bridge(
    processed,
    mtry,
    trees,
    min_n,
    bootstrap,
    sample_fraction,
    max_depth,
    max_leaves,
    n_bins,
    min_samples_leaf,
    split_criterion,
    min_impurity_decrease,
    max_batch_size,
    n_streams,
    seed
  )
}

cuda_ml_rand_forest_bridge <- function(
  processed,
  mtry,
  trees,
  min_n,
  bootstrap,
  sample_fraction,
  max_depth,
  max_leaves,
  n_bins,
  min_samples_leaf,
  split_criterion,
  min_impurity_decrease,
  max_batch_size,
  n_streams,
  seed
) {
  hardhat::validate_predictors_are_numeric(processed$predictors)
  hardhat::validate_outcomes_are_univariate(processed$outcomes)

  x <- as.matrix(processed$predictors)
  outcome <- processed$outcomes[[1L]]
  classification <- is.factor(outcome)
  stopifnot(
    "The outcome must be a factor or numeric vector" = classification ||
      is.numeric(outcome)
  )
  if (classification) {
    validate_classification_outcome(outcome)
  }
  mtry <- mtry %||%
    if (classification) {
      max(1L, as.integer(sqrt(ncol(x))))
    } else {
      ncol(x)
    }
  stopifnot(
    "`mtry` must be a whole number between 1 and the number of predictors" = is.numeric(
      mtry
    ) &&
      length(mtry) == 1L &&
      is.finite(mtry) &&
      mtry == as.integer(mtry) &&
      mtry >= 1L &&
      mtry <= ncol(x),
    "`trees` must be one positive whole number" = is.numeric(trees) &&
      length(trees) == 1L &&
      is.finite(trees) &&
      trees == as.integer(trees) &&
      trees >= 1L,
    "`min_n` must be one whole number of at least 2" = is.numeric(min_n) &&
      length(min_n) == 1L &&
      is.finite(min_n) &&
      min_n == as.integer(min_n) &&
      min_n >= 2L,
    "`bootstrap` must be TRUE or FALSE" = is.logical(bootstrap) &&
      length(bootstrap) == 1L &&
      !is.na(bootstrap),
    "`sample_fraction` must be one finite number in (0, 1]" = is.numeric(
      sample_fraction
    ) &&
      length(sample_fraction) == 1L &&
      is.finite(sample_fraction) &&
      sample_fraction > 0 &&
      sample_fraction <= 1,
    "`max_depth` must be one positive whole number" = is.numeric(max_depth) &&
      length(max_depth) == 1L &&
      is.finite(max_depth) &&
      max_depth == as.integer(max_depth) &&
      max_depth >= 1L,
    "`max_leaves` must be positive infinity or a whole number of at least 2" = is.numeric(
      max_leaves
    ) &&
      length(max_leaves) == 1L &&
      !is.na(max_leaves) &&
      ((is.infinite(max_leaves) && max_leaves > 0) ||
        (is.finite(max_leaves) &&
          max_leaves == as.integer(max_leaves) &&
          max_leaves >= 2L)),
    "`n_bins` must be one whole number of at least 2" = is.numeric(n_bins) &&
      length(n_bins) == 1L &&
      is.finite(n_bins) &&
      n_bins == as.integer(n_bins) &&
      n_bins >= 2L,
    "`min_samples_leaf` must be one positive whole number" = is.numeric(
      min_samples_leaf
    ) &&
      length(min_samples_leaf) == 1L &&
      is.finite(min_samples_leaf) &&
      min_samples_leaf == as.integer(min_samples_leaf) &&
      min_samples_leaf >= 1L,
    "`min_impurity_decrease` must be one non-negative finite number" = is.numeric(
      min_impurity_decrease
    ) &&
      length(min_impurity_decrease) == 1L &&
      is.finite(min_impurity_decrease) &&
      min_impurity_decrease >= 0,
    "`max_batch_size` must be one positive whole number" = is.numeric(
      max_batch_size
    ) &&
      length(max_batch_size) == 1L &&
      is.finite(max_batch_size) &&
      max_batch_size == as.integer(max_batch_size) &&
      max_batch_size >= 1L,
    "`n_streams` must be one positive whole number" = is.numeric(n_streams) &&
      length(n_streams) == 1L &&
      is.finite(n_streams) &&
      n_streams == as.integer(n_streams) &&
      n_streams >= 1L,
    "`seed` must be NULL or one non-negative whole number" = is.null(seed) ||
      (is.numeric(seed) &&
        length(seed) == 1L &&
        is.finite(seed) &&
        seed == as.integer(seed) &&
        seed >= 0)
  )

  seed <- seed %||% (sample.int(.Machine$integer.max, 1L) - 1L)
  split_criterion <- decision_tree_match_split_criterion(
    split_criterion,
    classification
  )
  max_leaves <- if (is.infinite(max_leaves)) -1L else as.integer(max_leaves)
  # cuML recovers mtry by truncating this single-precision fraction times p.
  max_features <- if (mtry == ncol(x)) {
    1
  } else {
    (mtry + 0.5) / ncol(x)
  }
  common <- list(
    n_trees = as.integer(trees),
    bootstrap = as.logical(bootstrap),
    max_samples = as.numeric(sample_fraction),
    n_streams = as.integer(n_streams),
    max_depth = as.integer(max_depth),
    max_leaves = max_leaves,
    max_features = as.numeric(max_features),
    n_bins = as.integer(n_bins),
    min_samples_leaf = as.integer(min_samples_leaf),
    min_samples_split = as.integer(min_n),
    split_criterion = split_criterion,
    min_impurity_decrease = as.numeric(min_impurity_decrease),
    max_batch_size = as.integer(max_batch_size),
    seed = as.integer(seed)
  )

  if (classification) {
    xptr <- rlang::exec(
      .rf_classifier_fit,
      input = x,
      labels = as.integer(outcome) - 1L,
      !!!common
    )
    class_levels <- levels(outcome)
  } else {
    xptr <- rlang::exec(
      .rf_regressor_fit,
      input = x,
      responses = as.numeric(outcome),
      !!!common
    )
    class_levels <- NULL
  }

  inference <- list(
    device = 1L,
    device_id = -1L,
    layout = 0L,
    precision = -1L,
    default_chunk_size = 0L,
    align_bytes = 0L
  )
  new_nvforest_model(
    xptr,
    class_levels,
    inference,
    cls = c("cuda_ml_rand_forest", "cuda_ml_nvforest"),
    blueprint = processed$blueprint
  )
}

#' @export
cuda_ml_get_state.cuda_ml_rand_forest <- function(model) {
  new_model_state(
    nvforest_model_payload(model),
    "cuda_ml_rand_forest_model_state"
  )
}

#' @export
cuda_ml_set_state.cuda_ml_rand_forest_model_state <- function(model_state) {
  payload <- cuda_ml_state_payload(
    model_state,
    "cuda_ml_rand_forest_model_state"
  )
  nvforest_unserialize_payload(
    payload,
    c("cuda_ml_rand_forest", "cuda_ml_nvforest")
  )
}

register_rand_forest_model <- function(pkgname) {
  for (mode in c("classification", "regression")) {
    parsnip::set_model_engine(
      model = "rand_forest",
      mode = mode,
      eng = pkgname
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
        defaults = list()
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
        allow_sparse_x = FALSE
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
          object = quote(object$fit),
          new_data = quote(new_data),
          type = type
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
        object = quote(object$fit),
        new_data = quote(new_data),
        type = "numeric"
      )
    )
  )

  invisible()
}
