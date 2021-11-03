lasso_validate_alpha <- function(alpha) {
  if (alpha <= 0) {
    stop("`alpha` (multiplier of the L1 penalty term) must be positive!")
  }
}

#' Train a linear model using LASSO regression.
#'
#' Train a linear model using LASSO (Least Absolute Shrinkage and Selection
#' Operator) regression.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @template lm
#' @param alpha Multiplier of the L1 penalty term (i.e., the result would become
#'   and Ordinary Least Square model if \code{alpha} were set to 0). Default: 1.
#' @param max_iter The maximum number of coordinate descent iterations.
#'   Default: 1000L.
#' @param tol Stop the coordinate descent when the duality gap is below this
#'   threshold. Default: 1e-3.
#' @param selection If "random", then instead of updating coefficients in cyclic
#'   order, a random coefficient is updated in each iteration. Default: "cyclic".
#'
#' @return A LASSO regressor that can be used with the 'predict' S3 generic to
#'   make predictions on new data points.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' model <- cuda_ml_lasso(formula = mpg ~ ., data = mtcars, alpha = 1e-3)
#' predictions <- predict(model, mtcars)
#'
#' # predictions will be comparable to those from a `glmnet` model with `lambda`
#' # set to 1e-3 and `alpha` set to 1
#' # (in `glmnet`, `lambda` is the weight of the penalty term, and `alpha` is
#' #  the elastic mixing parameter between L1 and L2 penalties.
#'
#' library(glmnet)
#'
#' glmnet_model <- glmnet(
#'   x = as.matrix(mtcars[names(mtcars) != "mpg"]), y = mtcars$mpg,
#'   alpha = 1, lambda = 1e-3, nlambda = 1, standardize = FALSE
#' )
#'
#' glm_predictions <- predict(
#'   glmnet_model, as.matrix(mtcars[names(mtcars) != "mpg"]), s = 0
#' )
#'
#' print(max(abs(glm_predictions - predictions$.pred)))
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
cuda_ml_lasso.data.frame <- function(x, y,
                                     alpha = 1,
                                     max_iter = 1000L, tol = 1e-3,
                                     fit_intercept = TRUE,
                                     normalize_input = FALSE,
                                     selection = c("cyclic", "random"),
                                     ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_lasso_bridge(
    processed = processed,
    alpha = alpha,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input,
    selection = selection
  )
}

#' @rdname cuda_ml_lasso
#' @export
cuda_ml_lasso.matrix <- function(x, y,
                                 alpha = 1,
                                 max_iter = 1000L, tol = 1e-3,
                                 fit_intercept = TRUE,
                                 normalize_input = FALSE,
                                 selection = c("cyclic", "random"),
                                 ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_lasso_bridge(
    processed = processed,
    alpha = alpha,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input,
    selection = selection
  )
}

#' @rdname cuda_ml_lasso
#' @export
cuda_ml_lasso.formula <- function(formula, data,
                                  alpha = 1,
                                  max_iter = 1000L, tol = 1e-3,
                                  fit_intercept = TRUE,
                                  normalize_input = FALSE,
                                  selection = c("cyclic", "random"),
                                  ...) {
  processed <- hardhat::mold(formula, data)

  cuda_ml_lasso_bridge(
    processed = processed,
    alpha = alpha,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input,
    selection = selection
  )
}

#' @rdname cuda_ml_lasso
#' @export
cuda_ml_lasso.recipe <- function(x, data,
                                 alpha = 1,
                                 max_iter = 1000L, tol = 1e-3,
                                 fit_intercept = TRUE,
                                 normalize_input = FALSE,
                                 selection = c("cyclic", "random"),
                                 ...) {
  processed <- hardhat::mold(x, data)

  cuda_ml_lasso_bridge(
    processed = processed,
    alpha = alpha,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input,
    selection = selection
  )
}

cuda_ml_lasso_bridge <- function(processed,
                                 alpha = 1,
                                 max_iter = 1000L, tol = 1e-3,
                                 fit_intercept = TRUE,
                                 normalize_input = FALSE,
                                 selection = c("cyclic", "random"),
                                 ...) {
  validate_lm_input(processed)
  lasso_validate_alpha(alpha)
  selection <- match.arg(selection)
  if (!fit_intercept && normalize_input) {
    stop(
      "fit_intercept=FALSE, normalize_input=TRUE is unsupported for LASSO ",
      "regression"
    )
  }

  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  model_xptr <- .cd_fit(
    x = x,
    y = y,
    fit_intercept = fit_intercept,
    normalize_input = normalize_input,
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
