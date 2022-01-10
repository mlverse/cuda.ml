logistic_reg_loss_type <- list(sigmoid = 0L, softmax = 2L)

logistic_reg_build_qn_params <- function(C,
                                         l1_ratio,
                                         penalty = c("l2", "l1", "elasticnet", "none")) {
  penalty <- match.arg(penalty)

  switch(penalty,
    "none" = {
      list(l1 = 0, l2 = 0)
    },
    "l1" = {
      list(l1 = 1 / C, l2 = 0)
    },
    "l2" = {
      list(l1 = 0, l2 = 1 / C)
    },
    "elasticnet" = {
      strength <- 1 / C
      if (is.null(l1_ratio) || !is.numeric(l1_ratio) || l1_ratio < 0 ||
          l1_ratio > 1) {
        stop(
          "`l1_ratio` must be non-NULL and be within [0, 1] for elasticnet ",
          "regularization."
        )
      }
      list(l1 = l1_ratio * strength, l2 = (1 - l1_ratio) * strength)
    }
  )
}

logistic_reg_validate_class_weight <- function(class_weight, processed) {
  if (identical(class_weight, "balanced")) {
    return()
  }
  lvls <- levels(processed$outcomes[[1]])
  n_lvls <- length(lvls)
  if (!setequal(names(class_weight), lvls) || length(class_weight) != n_lvls) {
    stop(
      "Expected `class_weight` to specify weights for the following possible ",
      "outcomes: {", paste(lvls, collapse = ", "), "}."
    )
  }

  for (v in class_weight) {
    if (!is.numeric(v) || v < 0 || !is.finite(v)) {
      stop(
        "Expected `class_weight` to be a numeric vector of finite, non-",
        "negative and non-NaN values."
      )
    }
  }
}

logistic_reg_validate_sample_weight <- function(sample_weight, processed) {
  n_samples <- length(processed$outcomes[[1]])
  if (!is.numeric(sample_weight) || length(sample_weight) != n_samples) {
    stop(
      "Expected `sample_weight` to be a numeric vector with length equal to ",
      "the number of training samples (", length(sample_weight), ")."
    )
  }
  if (any(sample_weight < 0) || any(!is.finite(sample_weight))) {
    stop(
      "Expected `sample_weight` to only contain finite, non-negative, and ",
      "non-NaN numeric values."
    )
  }
}

logistic_reg_build_sample_weight <- function(sample_weight,
                                             class_weight,
                                             processed) {
  outcomes <- processed$outcomes[[1]]
  n_samples <- length(outcomes)
  if (identical(class_weight, "balanced")) {
    class_weight <- n_samples / table(outcomes)
  }
  if (!is.null(class_weight)) {
    # update each sample weight value by multiplying it with its corresponding
    # class weight
    sample_weight <- sample_weight *
      as.numeric(class_weight[levels(outcomes)[outcomes]])
  }

  sample_weight
}

#' Train a logistic regression model.
#'
#' Train a logistic regression model using Quasi-Newton (QN) algorithms (i.e.,
#' Orthant-Wise Limited Memory Quasi-Newton (OWL-QN) if there is L1
#' regularization, Limited Memory BFGS (L-BFGS) otherwise).
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @template fit-intercept
#' @param penalty The penalty type, must be one of
#'   {"none", "l1", "l2", "elasticnet"}.
#'   If "none" or "l2" is selected, then L-BFGS solver will be used.
#'   If "l1" is selected, solver OWL-QN will be used.
#'   If "elasticnet" is selected, OWL-QN will be used if l1_ratio > 0, otherwise
#'   L-BFGS will be used. Default: "l2".
#' @param tol Tolerance for stopping criteria. Default: 1e-4.
#' @param C Inverse of regularization strength; must be a positive float.
#'   Default: 1.0.
#' @param class_weight If \code{NULL}, then each class has equal weight of
#'   \code{1}.
#'   If \code{class_weight} is set to \code{"balanced"}, then weights will be
#'   inversely proportional to class frequencies in the input data.
#'   If otherwise, then \code{class_weight} must be a named numeric vector of
#'   weight values, with names being class labels.
#'   If \code{class_weight} is not \code{NULL}, then each entry in
#'   \code{sample_weight} will be adjusted by multiplying its original value
#'   with the class weight of the corresponding sample's class.
#'   Default: NULL.
#' @param sample_weight Array of weights assigned to individual samples.
#'   If \code{NULL}, then each sample has an equal weight of 1. Default: NULL.
#' @param max_iters Maximum number of solver iterations. Default: 1000L.
#' @param linesearch_max_iters Max number of linesearch iterations per outer
#'   iteration used in the LBFGS- and OWL- QN solvers. Default: 50L.
#' @param l1_ratio The Elastic-Net mixing parameter, must \code{NULL} or be
#'   within the range of [0, 1]. Default: NULL.
#'
#' @examples
#' library(cuda.ml)
#'
#' X <- scale(as.matrix(iris[names(iris) != "Species"]))
#' y <- iris$Species
#'
#' model <- cuda_ml_logistic_reg(X, y, max_iters = 100)
#' predictions <- predict(model, X)
#'
#' # NOTE: if we were only performing binary classifications (e.g., by having
#' # `iris_data <- iris %>% mutate(Species = (Species == "setosa"))`), then the
#' # above would be conceptually equivalent to the following:
#' #
#' # iris_data <- iris %>% mutate(Species = (Species == "setosa"))
#' # model <- glm(
#' #   Species ~ ., data = iris_data, family = binomial(link = "logit"),
#' #   control = glm.control(epsilon = 1e-8, maxit = 100)
#' # )
#' #
#' # predict(model, iris_data, type = "response")
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_logistic_reg <- function(x, ...) {
  check_dots_used()
  UseMethod("cuda_ml_logistic_reg")
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_logistic_reg", x)
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.data.frame <- function(x, y,
                                            fit_intercept = TRUE,
                                            penalty = c("l2", "l1", "elasticnet", "none"),
                                            tol = 1e-4,
                                            C = 1.0,
                                            class_weight = NULL,
                                            sample_weight = NULL,
                                            max_iters = 1000L,
                                            linesearch_max_iters = 50L,
                                            l1_ratio = NULL,
                                            ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_logistic_reg_bridge(
    processed = processed,
    fit_intercept = fit_intercept,
    penalty = penalty,
    tol = tol,
    C = C,
    class_weight = class_weight,
    sample_weight = sample_weight,
    max_iters = max_iters,
    linesearch_max_iters = linesearch_max_iters,
    l1_ratio = l1_ratio
  )
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.matrix <- function(x, y,
                                        fit_intercept = TRUE,
                                        penalty = c("l2", "l1", "elasticnet", "none"),
                                        tol = 1e-4,
                                        C = 1.0,
                                        class_weight = NULL,
                                        sample_weight = NULL,
                                        max_iters = 1000L,
                                        linesearch_max_iters = 50L,
                                        l1_ratio = NULL,
                                        ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_logistic_reg_bridge(
    processed = processed,
    fit_intercept = fit_intercept,
    penalty = penalty,
    tol = tol,
    C = C,
    class_weight = class_weight,
    sample_weight = sample_weight,
    max_iters = max_iters,
    linesearch_max_iters = linesearch_max_iters,
    l1_ratio = l1_ratio
  )
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.formula <- function(formula, data,
                                         fit_intercept = TRUE,
                                         penalty = c("l2", "l1", "elasticnet", "none"),
                                         tol = 1e-4,
                                         C = 1.0,
                                         class_weight = NULL,
                                         sample_weight = NULL,
                                         max_iters = 1000L,
                                         linesearch_max_iters = 50L,
                                         l1_ratio = NULL,
                                         ...) {
  processed <- hardhat::mold(formula, data)

  cuda_ml_logistic_reg_bridge(
    processed = processed,
    fit_intercept = fit_intercept,
    penalty = penalty,
    tol = tol,
    C = C,
    class_weight = class_weight,
    sample_weight = sample_weight,
    max_iters = max_iters,
    linesearch_max_iters = linesearch_max_iters,
    l1_ratio = l1_ratio
  )
}

#' @rdname cuda_ml_logistic_reg
#' @export
cuda_ml_logistic_reg.recipe <- function(x, data,
                                        fit_intercept = TRUE,
                                        penalty = c("l2", "l1", "elasticnet", "none"),
                                        tol = 1e-4,
                                        C = 1.0,
                                        class_weight = NULL,
                                        sample_weight = NULL,
                                        max_iters = 1000L,
                                        linesearch_max_iters = 50L,
                                        l1_ratio = NULL,
                                        ...) {
  processed <- hardhat::mold(x, data)

  cuda_ml_logistic_reg_bridge(
    processed = processed,
    fit_intercept = fit_intercept,
    penalty = penalty,
    tol = tol,
    C = C,
    class_weight = class_weight,
    sample_weight = sample_weight,
    max_iters = max_iters,
    linesearch_max_iters = linesearch_max_iters,
    l1_ratio = l1_ratio
  )
}

cuda_ml_logistic_reg_bridge <- function(processed,
                                        fit_intercept,
                                        penalty, tol, C,
                                        class_weight, sample_weight,
                                        max_iters, linesearch_max_iters,
                                        l1_ratio) {
  hardhat::validate_predictors_are_numeric(processed$predictors)
  hardhat::validate_outcomes_are_univariate(processed$outcomes)
  hardhat::validate_outcomes_are_factors(processed$outcomes)

  x <- as.matrix(processed$predictors)
  # convert outcomes to 0-based enum values
  y <- as.integer(processed$outcomes[[1]]) - 1L

  n_classes <- nlevels(processed$outcomes[[1]])
  loss_type <- ifelse(
    n_classes > 2,
    logistic_reg_loss_type$softmax,
    logistic_reg_loss_type$sigmoid
  )

  qn_params <- logistic_reg_build_qn_params(C, l1_ratio, penalty)

  if (!is.null(class_weight)) {
    logistic_reg_validate_class_weight(class_weight, processed)
  }

  if (!is.null(sample_weight)) {
    logistic_reg_validate_sample_weight(sample_weight, processed)
  } else {
    sample_weight <- rep(1, nrow(x))
  }
  sample_weight <- logistic_reg_build_sample_weight(
    sample_weight, class_weight, processed
  )

  model_xptr <- .qn_fit(
    X = x, y = y,
    n_classes = n_classes,
    loss_type = loss_type,
    fit_intercept = fit_intercept,
    l1 = qn_params$l1, l2 = qn_params$l2,
    max_iters = as.integer(max_iters), tol = as.numeric(tol),
    delta = as.numeric(tol) * 0.01,
    linesearch_max_iters = as.integer(linesearch_max_iters),
    lbfgs_memory = 5L,
    sample_weight = as.numeric(sample_weight)
  )

  new_model(
    cls = "cuda_ml_logistic_reg",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}

#' Make predictions on new data points.
#'
#' Make predictions on new data points using a CuML logistic regression model.
#'
#' @template predict
#'
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_logistic_reg <- function(object, x, ...) {
  check_dots_used()

  processed <- hardhat::forge(x, object$blueprint)
  model <- object$xptr

  preds <- .qn_predict(
    X = as.matrix(processed$predictors),
    n_classes = model$n_classes,
    coefs = model$coefs,
    loss_type = model$loss_type,
    fit_intercept = model$fit_intercept
  )
  # convert from 0-based internal representation to 1-based representation which
  # will correspond to the factor levels of all possible outcomes in R
  preds <- preds + 1L

  postprocess_classification_results(preds, object)
}
