validate_lm_input <- function(processed) {
  hardhat::validate_outcomes_are_univariate(processed$outcomes)
  hardhat::validate_outcomes_are_numeric(processed$outcomes)
  hardhat::validate_predictors_are_numeric(processed$predictors)

  predictors <- as.matrix(processed$predictors)
  if (ncol(predictors) < 1) stop("Predictors must contain at least 1 feature.")
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
predict.cuda_ml_linear_model <- function(object, x, ...) {
  check_dots_used()

  processed <- hardhat::forge(x, object$blueprint)

  preds <- .lm_predict(
    input = as.matrix(processed$predictors),
    coef = object$xptr$coef,
    intercept = object$xptr$intercept
  )

  postprocess_regression_results(preds)
}
