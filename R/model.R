#' Model with numeric input.
#'
#' Generic ML model with one numeric input provided by a matrix or dataframe.
#'
#' @param x The input matrix or dataframe. Each data point should be a row
#'   and should consist of numeric values only.
#'
#' @name model-with-numeric-input
NULL

#' Supervised ML model with numeric output.
#'
#' Supervised ML model outputting a numeric value for each train/test sample.
#'
#' @param y A numeric vector of desired responses.
#'
#' @name supervised-model-with-numeric-output
NULL

#' Supervised ML algorithm with formula support.
#'
#' Supervised ML algorithm that supports specifying predictor(s) and response
#' variable using the formula syntax.
#'
#' @param formula If 'x' is a dataframe, then a R formula syntax of the form
#'   '<response col> ~ .' or
#'   '<response col> ~ <predictor 1> + <predictor 2> + ...'
#'   may be used to specify the response column and the predictor column(s).
#'
#' @name supervised-model-formula-spec
NULL

#' Supervised ML model with "classification" or "regression" modes.
#'
#' Supervised ML model that needs to be trained with different cuML routines
#' depending on whether "classification" or "regression" mode is required.
#'
#' @param mode Type of task to perform. Should be either "classification" or
#'   "regression".
#'
#' @name supervised-model-classification-or-regression-mode
NULL

#' Parameter for CUML log level.
#'
#' Log level parameter whose value must map to one of the CUML log levels.
#'
#' @param cuml_log_level Log level within cuML library functions. Must be one of
#'   {"off", "critical", "error", "warn", "info", "debug", "trace"}.
#'   Default: off.
#'
#' @name cuml-log-level
NULL

process_input_specs <- function(x, model) {
  if (!is.null(model$formula)) {
    predictor_cols <- labels(terms(model$formula, data = x))
    x <- x[, which(names(x) %in% predictor_cols)]
  }

  x
}

process_input_and_label_specs <- function(x, y, formula) {
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

  list(x, y)
}

new_model <- function(cls, mode, xptr, formula = NULL, resp_var = NULL, ...) {
  structure(
    c(
      list(
        mode = mode,
        xptr = xptr,
        formula = formula,
        resp_var_cls = class(resp_var),
        resp_var_attrs = attributes(resp_var)
      ),
      rlang::dots_list(...)
    ),
    class = cls
  )
}

postprocess_classification_results <- function(predictions, model) {
  if (!is.null(model$resp_var_cls)) {
    class(predictions) <- model$resp_var_cls
  }
  if (!is.null(model$resp_var_attrs)) {
    attributes(predictions) <- model$resp_var_attrs
  }

  predictions
}
