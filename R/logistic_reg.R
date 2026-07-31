logistic_reg_loss_type <- list(sigmoid = 0L, softmax = 2L)

logistic_reg_build_qn_params <- function(penalty, mixture) {
  if (is.null(penalty)) {
    return(list(l1 = 0, l2 = 0))
  }

  stopifnot(
    "`penalty` must be one non-negative finite number" = is.numeric(penalty) &&
      length(penalty) == 1L &&
      is.finite(penalty) &&
      penalty >= 0,
    "`mixture` must be one finite number between 0 and 1" = is.numeric(
      mixture
    ) &&
      length(mixture) == 1L &&
      is.finite(mixture) &&
      mixture >= 0 &&
      mixture <= 1
  )

  list(
    l1 = penalty * mixture,
    l2 = penalty * (1 - mixture)
  )
}

logistic_reg_validate_class_weight <- function(class_weight, processed) {
  if (identical(class_weight, "balanced")) {
    return(invisible())
  }

  outcome_levels <- levels(processed$outcomes[[1L]])
  stopifnot(
    "`class_weight` must be a named numeric vector with one value per class" = is.numeric(
      class_weight
    ) &&
      identical(sort(names(class_weight)), sort(outcome_levels)),
    "`class_weight` values must be finite and non-negative" = all(is.finite(
      class_weight
    )) &&
      all(class_weight >= 0)
  )

  invisible()
}

logistic_reg_validate_sample_weight <- function(sample_weight, processed) {
  stopifnot(
    "`sample_weight` must be a numeric vector with one value per observation" = is.numeric(
      sample_weight
    ) &&
      length(sample_weight) == nrow(processed$predictors),
    "`sample_weight` values must be finite and non-negative" = all(is.finite(
      sample_weight
    )) &&
      all(sample_weight >= 0)
  )

  invisible()
}

logistic_reg_build_sample_weight <- function(
  sample_weight,
  class_weight,
  processed
) {
  outcomes <- processed$outcomes[[1L]]

  if (is.null(sample_weight)) {
    sample_weight <- rep(1, length(outcomes))
  }
  if (identical(class_weight, "balanced")) {
    class_weight <- length(outcomes) / (nlevels(outcomes) * table(outcomes))
  }
  if (!is.null(class_weight)) {
    sample_weight <- sample_weight *
      unname(class_weight[as.character(outcomes)])
  }

  as.numeric(sample_weight)
}

#' Train a logistic or multinomial regression model
#'
#' Fits a factor outcome with cuML's quasi-Newton solver. Regularization follows
#' tidymodels conventions: \code{penalty} is the total regularization strength
#' and \code{mixture} is the proportion assigned to the L1 penalty. Set
#' \code{mixture = 0} for ridge, \code{mixture = 1} for lasso, or use an
#' intermediate value for an elastic-net penalty. Normalize predictors with a
#' recipe before fitting when scaling is required.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @template fit-intercept
#' @param penalty A non-negative regularization strength, or \code{NULL} for no
#'   regularization. Default: \code{NULL}.
#' @param mixture The proportion of regularization assigned to the L1 penalty,
#'   between 0 and 1. Default: 0.
#' @param tol Stopping tolerance. Default: 1e-4.
#' @param class_weight \code{NULL}, \code{"balanced"}, or a named numeric vector
#'   with one non-negative weight per outcome level.
#' @param sample_weight A numeric vector with one non-negative weight per
#'   training observation, or \code{NULL}.
#' @param max_iter Maximum solver iterations. Default: 1000L.
#' @param linesearch_max_iter Maximum line-search iterations per solver
#'   iteration. Default: 50L.
#' @param lbfgs_memory Number of vectors retained by the L-BFGS approximation.
#'   Default: 5L.
#' @param penalty_normalized Whether to normalize regularization by the number
#'   of observations. Default: TRUE.
#'
#' @return A classification model for use with \code{predict()}.
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_logistic_reg <- function(x, ...) {
  UseMethod("cuda_ml_logistic_reg")
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_logistic_reg", x)
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.data.frame <- function(
  x,
  y,
  fit_intercept = TRUE,
  penalty = NULL,
  mixture = 0,
  tol = 1e-4,
  class_weight = NULL,
  sample_weight = NULL,
  max_iter = 1000L,
  linesearch_max_iter = 50L,
  lbfgs_memory = 5L,
  penalty_normalized = TRUE,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)
  cuda_ml_logistic_reg_bridge(
    processed,
    fit_intercept,
    penalty,
    mixture,
    tol,
    class_weight,
    sample_weight,
    max_iter,
    linesearch_max_iter,
    lbfgs_memory,
    penalty_normalized
  )
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.matrix <- function(
  x,
  y,
  fit_intercept = TRUE,
  penalty = NULL,
  mixture = 0,
  tol = 1e-4,
  class_weight = NULL,
  sample_weight = NULL,
  max_iter = 1000L,
  linesearch_max_iter = 50L,
  lbfgs_memory = 5L,
  penalty_normalized = TRUE,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)
  cuda_ml_logistic_reg_bridge(
    processed,
    fit_intercept,
    penalty,
    mixture,
    tol,
    class_weight,
    sample_weight,
    max_iter,
    linesearch_max_iter,
    lbfgs_memory,
    penalty_normalized
  )
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.formula <- function(
  formula,
  data,
  fit_intercept = TRUE,
  penalty = NULL,
  mixture = 0,
  tol = 1e-4,
  class_weight = NULL,
  sample_weight = NULL,
  max_iter = 1000L,
  linesearch_max_iter = 50L,
  lbfgs_memory = 5L,
  penalty_normalized = TRUE,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(formula, data)
  cuda_ml_logistic_reg_bridge(
    processed,
    fit_intercept,
    penalty,
    mixture,
    tol,
    class_weight,
    sample_weight,
    max_iter,
    linesearch_max_iter,
    lbfgs_memory,
    penalty_normalized
  )
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.recipe <- function(
  x,
  data,
  fit_intercept = TRUE,
  penalty = NULL,
  mixture = 0,
  tol = 1e-4,
  class_weight = NULL,
  sample_weight = NULL,
  max_iter = 1000L,
  linesearch_max_iter = 50L,
  lbfgs_memory = 5L,
  penalty_normalized = TRUE,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, data)
  cuda_ml_logistic_reg_bridge(
    processed,
    fit_intercept,
    penalty,
    mixture,
    tol,
    class_weight,
    sample_weight,
    max_iter,
    linesearch_max_iter,
    lbfgs_memory,
    penalty_normalized
  )
}

cuda_ml_logistic_reg_bridge <- function(
  processed,
  fit_intercept,
  penalty,
  mixture,
  tol,
  class_weight,
  sample_weight,
  max_iter,
  linesearch_max_iter,
  lbfgs_memory,
  penalty_normalized
) {
  stopifnot(
    "`fit_intercept` must be TRUE or FALSE" = is.logical(fit_intercept) &&
      length(fit_intercept) == 1L &&
      !is.na(fit_intercept),
    "`tol` must be one positive finite number" = is.numeric(tol) &&
      length(tol) == 1L &&
      is.finite(tol) &&
      tol > 0,
    "`max_iter` must be one positive whole number" = is.numeric(max_iter) &&
      length(max_iter) == 1L &&
      is.finite(max_iter) &&
      max_iter >= 1L &&
      max_iter == as.integer(max_iter),
    "`linesearch_max_iter` must be one positive whole number" = is.numeric(
      linesearch_max_iter
    ) &&
      length(linesearch_max_iter) == 1L &&
      is.finite(linesearch_max_iter) &&
      linesearch_max_iter >= 1L &&
      linesearch_max_iter == as.integer(linesearch_max_iter),
    "`lbfgs_memory` must be one positive whole number" = is.numeric(
      lbfgs_memory
    ) &&
      length(lbfgs_memory) == 1L &&
      is.finite(lbfgs_memory) &&
      lbfgs_memory >= 1 &&
      lbfgs_memory == as.integer(lbfgs_memory),
    "`penalty_normalized` must be TRUE or FALSE" = is.logical(
      penalty_normalized
    ) &&
      length(penalty_normalized) == 1L &&
      !is.na(penalty_normalized)
  )
  hardhat::validate_predictors_are_numeric(processed$predictors)
  hardhat::validate_outcomes_are_univariate(processed$outcomes)
  hardhat::validate_outcomes_are_factors(processed$outcomes)

  qn_params <- logistic_reg_build_qn_params(penalty, mixture)
  if (!is.null(class_weight)) {
    logistic_reg_validate_class_weight(class_weight, processed)
  }
  if (!is.null(sample_weight)) {
    logistic_reg_validate_sample_weight(sample_weight, processed)
  }

  x <- as.matrix(processed$predictors)
  outcome <- processed$outcomes[[1L]]
  validate_classification_outcome(outcome)
  n_classes <- nlevels(outcome)

  loss_type <- if (n_classes == 2L) {
    logistic_reg_loss_type$sigmoid
  } else {
    logistic_reg_loss_type$softmax
  }

  model_xptr <- .qn_fit(
    X = x,
    y = as.integer(outcome) - 1L,
    n_classes = n_classes,
    loss_type = loss_type,
    fit_intercept = fit_intercept,
    l1 = qn_params$l1,
    l2 = qn_params$l2,
    max_iters = as.integer(max_iter),
    tol = as.numeric(tol),
    delta = as.numeric(tol) * 0.01,
    linesearch_max_iters = as.integer(linesearch_max_iter),
    lbfgs_memory = as.integer(lbfgs_memory),
    penalty_normalized = as.logical(penalty_normalized),
    sample_weight = logistic_reg_build_sample_weight(
      sample_weight,
      class_weight,
      processed
    )
  )

  new_model(
    cls = "cuda_ml_logistic_reg",
    mode = "classification",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}

#' Predict from a logistic or multinomial regression model
#'
#' @param object A fitted \code{cuda_ml_logistic_reg} model.
#' @param new_data New predictor data.
#' @param type Either \code{"class"} or \code{"prob"}.
#' @param ... Unused.
#'
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_logistic_reg <- function(
  object,
  new_data,
  type = c("class", "prob"),
  ...
) {
  check_dots_used()
  type <- match.arg(type)
  processed <- hardhat::forge(new_data, object$blueprint)
  model <- object$xptr

  if (identical(type, "prob")) {
    predictions <- .qn_predict_probabilities(
      X = as.matrix(processed$predictors),
      n_classes = model$n_classes,
      coefs = model$coefs,
      loss_type = model$loss_type,
      fit_intercept = model$fit_intercept
    )
    out <- postprocess_class_probabilities(predictions, object)
  } else {
    predictions <- .qn_predict(
      X = as.matrix(processed$predictors),
      n_classes = model$n_classes,
      coefs = model$coefs,
      loss_type = model$loss_type,
      fit_intercept = model$fit_intercept
    )
    out <- postprocess_classification_results(predictions + 1L, object)
  }

  hardhat::validate_prediction_size(out, processed$predictors)
  out
}

#' @export
cuda_ml_get_state.cuda_ml_logistic_reg <- function(model) {
  payload <- list(
    model = list(
      n_classes = model$xptr$n_classes,
      coefs = model$xptr$coefs,
      loss_type = model$xptr$loss_type,
      fit_intercept = model$xptr$fit_intercept
    ),
    blueprint = model$blueprint
  )
  new_model_state(payload, "cuda_ml_logistic_reg_model_state")
}

#' @export
cuda_ml_set_state.cuda_ml_logistic_reg_model_state <- function(model_state) {
  payload <- cuda_ml_state_payload(
    model_state,
    "cuda_ml_logistic_reg_model_state"
  )
  new_model(
    cls = "cuda_ml_logistic_reg",
    mode = "classification",
    xptr = payload$model,
    blueprint = payload$blueprint
  )
}

register_logistic_reg_models <- function(pkgname) {
  for (model in c("logistic_reg", "multinom_reg")) {
    parsnip::set_model_engine(
      model = model,
      mode = "classification",
      eng = pkgname
    )
    parsnip::set_dependency(model = model, eng = pkgname, pkg = pkgname)

    parsnip::set_model_arg(
      model = model,
      eng = pkgname,
      parsnip = "penalty",
      original = "penalty",
      func = list(pkg = "dials", fun = "penalty"),
      has_submodel = FALSE
    )
    parsnip::set_model_arg(
      model = model,
      eng = pkgname,
      parsnip = "mixture",
      original = "mixture",
      func = list(pkg = "dials", fun = "mixture"),
      has_submodel = FALSE
    )

    parsnip::set_fit(
      model = model,
      eng = pkgname,
      mode = "classification",
      value = list(
        interface = "formula",
        protect = c("formula", "data"),
        func = c(pkg = pkgname, fun = "cuda_ml_logistic_reg"),
        defaults = list()
      )
    )
    parsnip::set_encoding(
      model = model,
      eng = pkgname,
      mode = "classification",
      options = list(
        predictor_indicators = "none",
        compute_intercept = FALSE,
        remove_intercept = FALSE,
        allow_sparse_x = FALSE
      )
    )

    for (type in c("class", "prob")) {
      parsnip::set_pred(
        model = model,
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
  }

  invisible()
}
