sgd_match_loss <- function(loss = c("squared_loss", "log", "hinge")) {
  loss <- match.arg(loss)

  switch(loss,
    squared_loss = 0L,
    log = 1L,
    hinge = 2L
  )
}

sgd_match_penalty <- function(penalty = c("none", "l1", "l2", "elasticnet")) {
  penalty <- match.arg(penalty)

  switch(penalty,
    none = 0L,
    l1 = 1L,
    l2 = 2L,
    elasticnet = 3L
  )
}

sgd_match_learning_rate <- function(learning_rate = c("constant", "invscaling", "adaptive")) {
  learning_rate <- match.arg(learning_rate)

  switch(learning_rate,
    constant = 1L,
    invscaling = 2L,
    adaptive = 3L
  )
}

#' Train a MBSGD linear model.
#'
#' Train a linear model using mini-batch stochastic gradient descent.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @template fit-intercept
#' @template l1_ratio
#' @param loss Loss function, must be one of {"squared_loss", "log", "hinge"}.
#' @param penalty Type of regularization to perform, must be one of
#'   {"none", "l1", "l2", "elasticnet"}.
#'
#'   - "none": no regularization.
#'   - "l1": perform regularization based on the L1-norm (LASSO) which tries to
#'           minimize the sum of the absolute values of the coefficients.
#'   - "l2": perform regularization based on the L2 norm (Ridge) which tries to
#'           minimize the sum of the square of the coefficients.
#'   - "elasticnet": perform the Elastic Net regularization which is based on
#'                   the weighted averable of L1 and L2 norms.
#'   Default: "none".
#' @param alpha Multiplier of the penalty term. Default: 1e-4.
#' @param batch_size The number of samples that will be included in each batch.
#'   Default: 32L.
#' @param epochs The number of times the model should iterate through the entire
#'   dataset during training. Default: 1000L.
#' @param tol Threshold for stopping training. Training will stop if
#'   (loss in current epoch) > (loss in previous epoch) - \code{tol}.
#'   Default: 1e-3.
#' @param shuffle Whether to shuffles the training data after each epoch.
#'   Default: True.
#' @param eta0 The initial learning rate. Default: 1e-3.
#' @param power_t The exponent used for calculating the invscaling learning
#'   rate. Default: 0.5.
#' @param learning_rate Must be one of {"constant", "invscaling", "adaptive"}.
#'
#'   - "constant": the learning rate will be kept constant.
#'   - "invscaling": (learning rate) = (initial learning rate) / pow(t, power_t)
#'                   where \code{t} is the number of epochs and
#'                   \code{power_t} is a tunable parameter of this model.
#'   - "adaptive": (learning rate) = (initial learning rate) as long as the
#'                 training loss keeps decreasing. Each time the last
#'                 \code{n_iter_no_change} consecutive epochs fail to decrease
#'                 the training loss by \code{tol}, the current learning rate is
#'                 divided by 5.
#'   Default: "constant".
#' @param eta0 The initial learning rate. Default: 1e-3.
#' @param power_t The exponent used in the invscaling learning rate
#'   calculations.
#' @param n_iters_no_change The maximum number of epochs to train if there is no
#'   imporvement in the model. Default: 5.
#'
#' @return A linear model that can be used with the 'predict' S3 generic to make
#'   predictions on new data points.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' model <- cuda_ml_sgd(
#'   mpg ~ ., mtcars,
#'   batch_size = 4L, epochs = 50000L,
#'   learning_rate = "adaptive", eta0 = 1e-5,
#'   penalty = "l2", alpha = 1e-5, tol = 1e-6,
#'   n_iters_no_change = 10L
#' )
#'
#' preds <- predict(model, mtcars[names(mtcars) != "mpg"])
#' print(all.equal(preds$.pred, mtcars$mpg, tolerance = 0.09))
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_sgd <- function(x, ...) {
  check_dots_used()
  UseMethod("cuda_ml_sgd")
}

#' @rdname cuda_ml_sgd
#' @export
cuda_ml_sgd.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_sgd", x)
}

#' @rdname cuda_ml_sgd
#' @export
cuda_ml_sgd.data.frame <- function(x, y,
                                   fit_intercept = TRUE,
                                   loss = c("squared_loss", "log", "hinge"),
                                   penalty = c("none", "l1", "l2", "elasticnet"),
                                   alpha = 1e-4, l1_ratio = 0.5,
                                   epochs = 1000L, tol = 1e-3, shuffle = TRUE,
                                   learning_rate = c("constant", "invscaling", "adaptive"),
                                   eta0 = 1e-3, power_t = 0.5, batch_size = 32L,
                                   n_iters_no_change = 5L,
                                   ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_sgd_bridge(
    processed = processed,
    fit_intercept = fit_intercept,
    loss = loss,
    penalty = penalty,
    alpha = alpha,
    l1_ratio = l1_ratio,
    epochs = epochs,
    tol = tol,
    shuffle = shuffle,
    learning_rate = learning_rate,
    eta0 = eta0,
    power_t = power_t,
    batch_size = batch_size,
    n_iters_no_change = n_iters_no_change
  )
}

#' @rdname cuda_ml_sgd
#' @export
cuda_ml_sgd.matrix <- function(x, y,
                               fit_intercept = TRUE,
                               loss = c("squared_loss", "log", "hinge"),
                               penalty = c("none", "l1", "l2", "elasticnet"),
                               alpha = 1e-4, l1_ratio = 0.5,
                               epochs = 1000L, tol = 1e-3, shuffle = TRUE,
                               learning_rate = c("constant", "invscaling", "adaptive"),
                               eta0 = 1e-3, power_t = 0.5, batch_size = 32L,
                               n_iters_no_change = 5L,
                               ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_sgd_bridge(
    processed = processed,
    fit_intercept = fit_intercept,
    loss = loss,
    penalty = penalty,
    alpha = alpha,
    l1_ratio = l1_ratio,
    epochs = epochs,
    tol = tol,
    shuffle = shuffle,
    learning_rate = learning_rate,
    eta0 = eta0,
    power_t = power_t,
    batch_size = batch_size,
    n_iters_no_change = n_iters_no_change
  )
}

#' @rdname cuda_ml_sgd
#' @export
cuda_ml_sgd.formula <- function(formula, data,
                                fit_intercept = TRUE,
                                loss = c("squared_loss", "log", "hinge"),
                                penalty = c("none", "l1", "l2", "elasticnet"),
                                alpha = 1e-4, l1_ratio = 0.5,
                                epochs = 1000L, tol = 1e-3, shuffle = TRUE,
                                learning_rate = c("constant", "invscaling", "adaptive"),
                                eta0 = 1e-3, power_t = 0.5, batch_size = 32L,
                                n_iters_no_change = 5L,
                                ...) {
  processed <- hardhat::mold(formula, data)

  cuda_ml_sgd_bridge(
    processed = processed,
    fit_intercept = fit_intercept,
    loss = loss,
    penalty = penalty,
    alpha = alpha,
    l1_ratio = l1_ratio,
    epochs = epochs,
    tol = tol,
    shuffle = shuffle,
    learning_rate = learning_rate,
    eta0 = eta0,
    power_t = power_t,
    batch_size = batch_size,
    n_iters_no_change = n_iters_no_change
  )
}

#' @rdname cuda_ml_sgd
#' @export
cuda_ml_sgd.recipe <- function(x, data,
                               fit_intercept = TRUE,
                               loss = c("squared_loss", "log", "hinge"),
                               penalty = c("none", "l1", "l2", "elasticnet"),
                               alpha = 1e-4, l1_ratio = 0.5,
                               epochs = 1000L, tol = 1e-3, shuffle = TRUE,
                               learning_rate = c("constant", "invscaling", "adaptive"),
                               eta0 = 1e-3, power_t = 0.5, batch_size = 32L,
                               n_iters_no_change = 5L,
                               ...) {
  processed <- hardhat::mold(x, data)

  cuda_ml_sgd_bridge(
    processed = processed,
    fit_intercept = fit_intercept,
    loss = loss,
    penalty = penalty,
    alpha = alpha,
    l1_ratio = l1_ratio,
    epochs = epochs,
    tol = tol,
    shuffle = shuffle,
    learning_rate = learning_rate,
    eta0 = eta0,
    power_t = power_t,
    batch_size = batch_size,
    n_iters_no_change = n_iters_no_change
  )
}

cuda_ml_sgd_bridge <- function(processed,
                               fit_intercept,
                               loss,
                               penalty,
                               alpha, l1_ratio,
                               epochs, tol, shuffle,
                               learning_rate,
                               eta0, power_t, batch_size,
                               n_iters_no_change) {
  validate_lm_input(processed)
  loss <- sgd_match_loss(loss)
  penalty <- sgd_match_penalty(penalty)

  learning_rate <- sgd_match_learning_rate(learning_rate)
  if (eta0 <= 0) {
    stop("eta0 must be > 0")
  }

  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  model_xptr <- .sgd_fit(
    x = x,
    y = y,
    fit_intercept = fit_intercept,
    batch_size = as.integer(batch_size),
    epochs = as.integer(epochs),
    lr_type = learning_rate,
    eta0 = as.numeric(eta0),
    power_t = as.numeric(power_t),
    loss = loss,
    penalty = penalty,
    alpha = as.numeric(alpha),
    l1_ratio = as.numeric(l1_ratio),
    shuffle = shuffle,
    tol = as.numeric(tol),
    n_iter_no_change = as.integer(n_iters_no_change)
  )

  new_linear_model(
    cls = "cuda_ml_sgd",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}
