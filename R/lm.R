validate_lm_input <- function(processed) {
  hardhat::validate_outcomes_are_univariate(processed$outcomes)
  hardhat::validate_outcomes_are_numeric(processed$outcomes)
  hardhat::validate_predictors_are_numeric(processed$predictors)

  predictors <- as.matrix(processed$predictors)
  if (ncol(predictors) < 1) {
    stop("Predictors must contain at least 1 feature.")
  }
  if (nrow(predictors) < 2) stop("At least 2 samples are required.")
}

new_linear_model <- function(cls, xptr, ...) {
  new_model(
    cls = c(cls, "cuda_ml_linear_model"),
    mode = "regression",
    xptr = xptr,
    ...
  )
}

#' Make predictions on new data points.
#'
#' Make predictions on new data points using a linear model.
#'
#' @template predict
#'
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_linear_model <- function(object, new_data, ...) {
  check_dots_used()

  processed <- hardhat::forge(new_data, object$blueprint)

  preds <- .lm_predict(
    input = as.matrix(processed$predictors),
    coef = object$xptr$coef,
    intercept = object$xptr$intercept
  )

  out <- postprocess_regression_results(preds)
  hardhat::validate_prediction_size(out, processed$predictors)
  out
}

#' @export
cuda_ml_get_state.cuda_ml_linear_model <- function(model) {
  payload <- list(
    model_class = class(model)[[1L]],
    coefficients = as.numeric(model$xptr$coef),
    intercept = as.numeric(model$xptr$intercept),
    blueprint = model$blueprint
  )
  new_model_state(payload, "cuda_ml_linear_model_state")
}

#' @export
cuda_ml_set_state.cuda_ml_linear_model_state <- function(model_state) {
  payload <- cuda_ml_state_payload(
    model_state,
    "cuda_ml_linear_model_state"
  )
  supported <- c(
    "cuda_ml_ols",
    "cuda_ml_ridge",
    "cuda_ml_lasso",
    "cuda_ml_elastic_net",
    "cuda_ml_sgd"
  )
  stopifnot(
    "The serialized linear-model class is unsupported" = payload$model_class %in%
      supported
  )

  new_linear_model(
    cls = payload$model_class,
    xptr = list(
      coef = payload$coefficients,
      intercept = payload$intercept
    ),
    blueprint = payload$blueprint
  )
}

#' Train a regularized linear regression model
#'
#' This is the tidymodels-style linear regression interface. It dispatches to
#' \code{cuda_ml_ols()}, \code{cuda_ml_ridge()}, \code{cuda_ml_lasso()}, or
#' \code{cuda_ml_elastic_net()} according to \code{penalty} and \code{mixture}.
#' The named model functions remain available when direct control over their
#' solver arguments is needed.
#'
#' @param formula A model formula.
#' @param data A data frame containing predictors and outcome.
#' @param penalty A non-negative regularization strength, or \code{NULL} for no
#'   regularization.
#' @param mixture The proportion of regularization assigned to the L1 penalty,
#'   between 0 and 1. When \code{NULL}, a lasso penalty is used.
#' @param ... Arguments passed to the selected named model function.
#'
#' @return A fitted cuda.ml linear model.
#' @export
cuda_ml_linear_reg <- function(
  formula,
  data,
  penalty = NULL,
  mixture = NULL,
  ...
) {
  if (is.null(penalty)) {
    return(cuda_ml_ols(formula, data, ...))
  }

  mixture <- mixture %||% 1
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

  if (penalty == 0) {
    cuda_ml_ols(formula, data, ...)
  } else if (mixture == 0) {
    cuda_ml_ridge(formula, data, alpha = penalty, ...)
  } else if (mixture == 1) {
    cuda_ml_lasso(formula, data, alpha = penalty, ...)
  } else {
    cuda_ml_elastic_net(
      formula,
      data,
      alpha = penalty,
      l1_ratio = mixture,
      ...
    )
  }
}

register_linear_reg_model <- function(pkgname) {
  parsnip::set_model_engine(
    model = "linear_reg",
    mode = "regression",
    eng = pkgname
  )
  parsnip::set_dependency(model = "linear_reg", eng = pkgname, pkg = pkgname)

  parsnip::set_model_arg(
    model = "linear_reg",
    eng = pkgname,
    parsnip = "penalty",
    original = "penalty",
    func = list(pkg = "dials", fun = "penalty"),
    has_submodel = FALSE
  )
  parsnip::set_model_arg(
    model = "linear_reg",
    eng = pkgname,
    parsnip = "mixture",
    original = "mixture",
    func = list(pkg = "dials", fun = "mixture"),
    has_submodel = FALSE
  )

  parsnip::set_fit(
    model = "linear_reg",
    eng = pkgname,
    mode = "regression",
    value = list(
      interface = "formula",
      protect = c("formula", "data"),
      func = c(pkg = pkgname, fun = "cuda_ml_linear_reg"),
      defaults = list()
    )
  )
  parsnip::set_encoding(
    model = "linear_reg",
    eng = pkgname,
    mode = "regression",
    options = list(
      predictor_indicators = "none",
      compute_intercept = FALSE,
      remove_intercept = FALSE,
      allow_sparse_x = FALSE
    )
  )
  parsnip::set_pred(
    model = "linear_reg",
    eng = pkgname,
    mode = "regression",
    type = "numeric",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = quote(object$fit),
        new_data = quote(new_data)
      )
    )
  )

  invisible()
}
