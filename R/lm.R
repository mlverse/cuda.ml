#' Make predictions on new data points.
#'
#' Make predictions on new data points using a linear model.
#' See \code{\link{cuda_ml_predict}} for full documentation of parameters.
#'
#' @template predict
#'
#' @seealso cuda_ml_predict
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_linear_model <- function(object, ...) {
  check_dots_used()

  x <- ..1

  processed <- hardhat::forge(x, object$blueprint)

  preds <- .glm_predict(
    input = as.matrix(processed$predictors),
    coef = object$xptr$coef,
    intercept = object$xptr$intercept
  )

  postprocess_regression_results(preds)
}
