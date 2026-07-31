elastic_net_validate_alpha <- function(alpha) {
  stopifnot(
    "`alpha` must be one positive finite number" = is.numeric(alpha) &&
      length(alpha) == 1L &&
      is.finite(alpha) &&
      alpha > 0
  )
}

elastic_net_validate_l1_ratio <- function(l1_ratio) {
  stopifnot(
    "`l1_ratio` must be one finite number between 0 and 1" = is.numeric(
      l1_ratio
    ) &&
      length(l1_ratio) == 1L &&
      is.finite(l1_ratio) &&
      l1_ratio >= 0 &&
      l1_ratio <= 1
  )
}

#' Train a linear model using elastic regression.
#'
#' Train a linear model with combined L1 and L2 priors as the regularizer.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @template fit-intercept
#' @template coordinate-descend
#' @template l1_ratio
#' @param alpha Positive multiplier of the penalty term. Use
#'   \code{cuda_ml_ols()} for an unpenalized linear model. Default: 1.
#'
#' @return An elastic net regressor that can be used with the 'predict' S3
#'   generic to make predictions on new data points.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (interactive() && cuda_ml_backend_info()$runtime_installed) {
#'   model <- cuda_ml_elastic_net(
#'     formula = mpg ~ ., data = mtcars, alpha = 1e-3, l1_ratio = 0.6
#'   )
#'   cuda_ml_predictions <- predict(model, mtcars)
#'
#'   # predictions will be comparable to those from a `glmnet` model with
#'   # `lambda` set to 1e-3 and `alpha` set to 0.6
#'   # (in `glmnet`, `lambda` is the weight of the penalty term, and `alpha` is
#'   #  the elastic mixing parameter between L1 and L2 penalties.
#'
#'   if (requireNamespace("glmnet", quietly = TRUE)) {
#'     glmnet_model <- glmnet::glmnet(
#'       x = as.matrix(mtcars[names(mtcars) != "mpg"]), y = mtcars$mpg,
#'       alpha = 0.6, lambda = 1e-3, nlambda = 1, standardize = FALSE
#'     )
#'
#'     glm_predictions <- predict(
#'       glmnet_model, as.matrix(mtcars[names(mtcars) != "mpg"]),
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
cuda_ml_elastic_net <- function(x, ...) {
  UseMethod("cuda_ml_elastic_net")
}

#' @rdname cuda_ml_elastic_net
#' @export
cuda_ml_elastic_net.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_elastic_net", x)
}

#' @rdname cuda_ml_elastic_net
#' @export
cuda_ml_elastic_net.data.frame <- function(
  x,
  y,
  alpha = 1,
  l1_ratio = 0.5,
  max_iter = 1000L,
  tol = 1e-3,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)

  cuda_ml_elastic_net_bridge(
    processed = processed,
    alpha = alpha,
    l1_ratio = l1_ratio,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    selection = selection
  )
}

#' @rdname cuda_ml_elastic_net
#' @export
cuda_ml_elastic_net.matrix <- function(
  x,
  y,
  alpha = 1,
  l1_ratio = 0.5,
  max_iter = 1000L,
  tol = 1e-3,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)

  cuda_ml_elastic_net_bridge(
    processed = processed,
    alpha = alpha,
    l1_ratio = l1_ratio,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    selection = selection
  )
}

#' @rdname cuda_ml_elastic_net
#' @export
cuda_ml_elastic_net.formula <- function(
  formula,
  data,
  alpha = 1,
  l1_ratio = 0.5,
  max_iter = 1000L,
  tol = 1e-3,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(formula, data)

  cuda_ml_elastic_net_bridge(
    processed = processed,
    alpha = alpha,
    l1_ratio = l1_ratio,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    selection = selection
  )
}

#' @rdname cuda_ml_elastic_net
#' @export
cuda_ml_elastic_net.recipe <- function(
  x,
  data,
  alpha = 1,
  l1_ratio = 0.5,
  max_iter = 1000L,
  tol = 1e-3,
  fit_intercept = TRUE,
  selection = c("cyclic", "random"),
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, data)

  cuda_ml_elastic_net_bridge(
    processed = processed,
    alpha = alpha,
    l1_ratio = l1_ratio,
    max_iter = max_iter,
    tol = tol,
    fit_intercept = fit_intercept,
    selection = selection
  )
}

cuda_ml_elastic_net_bridge <- function(
  processed,
  alpha,
  l1_ratio,
  max_iter,
  tol,
  fit_intercept,
  selection = c("cyclic", "random")
) {
  validate_lm_input(processed)
  elastic_net_validate_alpha(alpha)
  elastic_net_validate_l1_ratio(l1_ratio)
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
    l1_ratio = as.numeric(l1_ratio),
    shuffle = identical(selection, "random"),
    tol = as.numeric(tol)
  )

  new_linear_model(
    cls = "cuda_ml_elastic_net",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}
