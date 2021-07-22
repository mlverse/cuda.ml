#' @importFrom stats terms
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
      stop(
        "'x' must be a data.frame when predictor column(s) and response ",
        "column are specified using the formula syntax."
      )
    }
    response_col <- all.vars(formula)[[1]]
    predictor_cols <- labels(terms(formula, data = x))
    y <- x[, response_col]
    x <- x[, which(names(x) %in% predictor_cols)]
  } else if (!is.numeric(y)) {
    stop(
      "'y' must be a numeric vector if predictor(s) and responses are not",
      " specified using the formula syntax."
    )
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
