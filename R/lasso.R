lasso_validate_alpha <- function(alpha) {
  stopifnot(
    "`alpha` must be one positive finite number" = is.numeric(alpha) &&
      length(alpha) == 1L &&
      is.finite(alpha) &&
      alpha > 0
  )
}

#' Train a linear model using LASSO regression.
#'
#' Train a linear model using LASSO (Least Absolute Shrinkage and Selection
#' Operator) regression.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @template fit-intercept
#' @template coordinate-descend
#' @param alpha Positive multiplier of the L1 penalty term. Use
#'   \code{cuda_ml_ols()} for an unpenalized linear model. Default: 1.
#'
#' @return A LASSO regressor that can be used with the 'predict' S3 generic to
#'   make predictions on new data points.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (interactive() && cuda_ml_backend_info()$runtime_installed) {
#'   model <- cuda_ml_lasso(formula = mpg ~ ., data = mtcars, alpha = 1e-3)
#'   predictors <- subset(mtcars, select = -mpg)
#'   cuda_ml_predictions <- predict(model, predictors)
#'
#'   # predictions will be comparable to those from a `glmnet` model with
#'   # `lambda` set to 1e-3 and `alpha` set to 1
#'   # (in `glmnet`, `lambda` is the weight of the penalty term, and `alpha` is
#'   #  the elastic mixing parameter between L1 and L2 penalties.
#'
#'   if (requireNamespace("glmnet", quietly = TRUE)) {
#'     glmnet_model <- glmnet::glmnet(
#'       x = as.matrix(predictors), y = mtcars$mpg,
#'       alpha = 1, lambda = 1e-3, nlambda = 1, standardize = FALSE
#'     )
#'
#'     glm_predictions <- predict(
#'       glmnet_model, as.matrix(predictors),
#'       s = 0
#'     )
#'
#'     print(
#'       all.equal(
#'         as.numeric(glm_predictions),
#'         cuda_ml_predictions$.pred,
#'         tolerance = 1e-2
#'       )
#'     )
#'   }
#' }
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_lasso <- function(x, ...) {
  UseMethod("cuda_ml_lasso")
}

#' @rdname cuda_ml_lasso
#' @export
cuda_ml_lasso.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_lasso", x)
}

#' @rdname cuda_ml_lasso
#' @export
cuda_ml_lasso.data.frame <- function(
  x,
  y,
  alpha = 1,
  max_iter = 1000L,
  tol = 1e-3,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)

  cuda_ml_lasso_bridge(
    processed = processed,
    alpha = alpha,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    selection = selection
  )
}

#' @rdname cuda_ml_lasso
#' @export
cuda_ml_lasso.matrix <- function(
  x,
  y,
  alpha = 1,
  max_iter = 1000L,
  tol = 1e-3,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)

  cuda_ml_lasso_bridge(
    processed = processed,
    alpha = alpha,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    selection = selection
  )
}

#' @rdname cuda_ml_lasso
#' @export
cuda_ml_lasso.formula <- function(
  formula,
  data,
  alpha = 1,
  max_iter = 1000L,
  tol = 1e-3,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(formula, data)

  cuda_ml_lasso_bridge(
    processed = processed,
    alpha = alpha,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    selection = selection
  )
}

#' @rdname cuda_ml_lasso
#' @export
cuda_ml_lasso.recipe <- function(
  x,
  data,
  alpha = 1,
  max_iter = 1000L,
  tol = 1e-3,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, data)

  cuda_ml_lasso_bridge(
    processed = processed,
    alpha = alpha,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    selection = selection
  )
}

cuda_ml_lasso_bridge <- function(
  processed,
  alpha,
  max_iter,
  tol,
  fit_intercept,
  selection = c("cyclic", "random")
) {
  validate_lm_input(processed)
  lasso_validate_alpha(alpha)
  selection <- match.arg(selection)
  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  model_xptr <- .cd_fit(
    x = x,
    y = y,
    fit_intercept = fit_intercept,
    epochs = as.integer(max_iter),
    loss = 0L, # squared loss
    alpha = as.numeric(alpha),
    l1_ratio = 1,
    shuffle = identical(selection, "random"),
    tol = as.numeric(tol)
  )

  new_linear_model(
    cls = "cuda_ml_lasso",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}
