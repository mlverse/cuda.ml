match_kernel_type <- function(kernel = c("rbf", "tanh", "polynomial", "linear")) {
  kernel <- match.arg(kernel)

  switch(
    kernel,
    linear = 0L,
    polynomial = 1L,
    rbf = 2L,
    tanh = 3L
  )
}

#' Train a SVM model.
#'
#' Train a Support Vector Machine model for classification or regression tasks.
#' Please note only binary classification is implemented by cuML at the moment.
#'
#' @inheritParams model-with-numeric-input
#' @inheritParams supervised-model-with-numeric-output
#' @inheritParams supervised-model-formula-spec
#' @inheritParams supervised-model-classification-or-regression-mode
#' @inheritParams cuml-log-level
#' @param cost A positive number for the cost of predicting a sample within or
#'   on the wrong side of the margin. Default: 1.
#' @param kernel Type of the SVM kernel function (must be one of "rbf", "tanh",
#'   "polynomial", or "linear"). Default: "rbf".
#' @param gamma The gamma coefficient (only relevant to polynomial, RBF, and
#'   tanh kernel functions, see explanations below).
#'   Default: 1 / (num features).
#'
#'   The following kernels are implemented:
#'     - RBF K(x_1, x_2) = exp(-gamma |x_1-x_2|^2)
#'     - TANH K(x_1, x_2) = tanh(gamma <x_1,x_2> + coef0)
#'     - POLYNOMIAL K(x_1, x_2) = (gamma <x_1,x_2> + coef0)^degree
#'     - LINEAR K(x_1,x_2) = <x_1,x_2>,
#'   where < , > denotes the dot product.
#' @param coef0 The 0th coefficient (only applicable to polynomial and tanh
#'   kernel functions, see explanations below). Default: 0.
#'
#'   The following kernels are implemented:
#'     - RBF K(x_1, x_2) = exp(-gamma |x_1-x_2|^2)
#'     - TANH K(x_1, x_2) = tanh(gamma <x_1,x_2> + coef0)
#'     - POLYNOMIAL K(x_1, x_2) = (gamma <x_1,x_2> + coef0)^degree
#'     - LINEAR K(x_1,x_2) = <x_1,x_2>,
#'   where < , > denotes the dot product.
#' @param degree Degree of the polynomial kernel function (note: not applicable
#'   to other kernel types, see explanations below). Default: 3.
#'
#'   The following kernels are implemented:
#'     - RBF K(x_1, x_2) = exp(-gamma |x_1-x_2|^2)
#'     - TANH K(x_1, x_2) = tanh(gamma <x_1,x_2> + coef0)
#'     - POLYNOMIAL K(x_1, x_2) = (gamma <x_1,x_2> + coef0)^degree
#'     - LINEAR K(x_1,x_2) = <x_1,x_2>,
#'   where < , > denotes the dot product.
#' @param tol Tolerance to stop fitting. Default: 1e-3.
#' @param max_iter Maximum number of outer iterations in SmoSolver.
#'   Default: 100 * (num samples).
#' @param nochange_steps Number of steps with no change w.r.t convergence.
#'   Default: 1000.
#' @param cache_size Size of kernel cache (MiB) in device memory. Default: 1024.
#' @param sample_weights Optional weight assigned to each input data point.
#'
#' @examples
#'
#' library(cuml4r)
#' library(dplyr)
#'
#' samples <- iris
#' samples[,"isSetosa"] <- (iris[,"Species"] == "setosa")
#' samples <- samples[, names(samples) != "Species"]
#'
#' model <- cuml_svm(
#'   samples,
#'   formula = isSetosa ~ .,
#'   mode = "classification",
#'   kernel = "rbf"
#' )
#'
#' predictions <- predict(model, samples)
#'
#' cat(
#'   "Number of correct predictions: ",
#'   sum(predictions == samples[, "isSetosa"]),
#'   "\n"
#' )
#' @export
cuml_svm <- function(x, y = NULL, formula = NULL,
                     mode = c("classification", "regression"),
                     cost = 1,
                     kernel = c("rbf", "tanh", "polynomial", "linear"),
                     gamma = 1 / ncol(x),
                     coef0 = 0,
                     degree = 3L,
                     tol = 1e-3,
                     max_iter = 100L * nrow(x),
                     nochange_steps = 1000L,
                     cache_size = 1024,
                     sample_weights = NULL,
                     cuml_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  mode <- match.arg(mode)
  kernel <- match_kernel_type(kernel)
  cuml_log_level <- match_cuml_log_level(cuml_log_level)
  c(x, y) %<-% process_input_and_label_specs(x, y, formula)

  switch(
    mode,
    classification = {
      new_model(
        cls = "cuml_svm",
        mode = mode,
        xptr = .svc_fit(
          input = as.matrix(x),
          labels = as.integer(y),
          cost = as.numeric(cost),
          kernel = kernel,
          gamma = as.numeric(gamma),
          coef0 = as.numeric(coef0),
          degree = as.integer(degree),
          tol = as.numeric(tol),
          max_iter = as.integer(max_iter),
          nochange_steps = as.integer(nochange_steps),
          cache_size = as.numeric(cache_size),
          sample_weights = as.numeric(sample_weights),
          verbosity = cuml_log_level
        ),
        formula = formula,
        resp_var = y
      )
    },
    regression = {
      stop("Not implemented")
    }
  )
}

#' Predict using a SVM model.
#'
#' Predict using a Support Vector Machine model.
#'
#' @inheritParams model-with-numeric-input
#' @param model A Support Vector Machine model.
#'
#' @export
predict.cuml_svm <- function(model, x) {
  x <- process_input_specs(x, model)

  switch (
    model$mode,
    classification = {
      .svc_predict(
        model_xptr = model$xptr,
        input = as.matrix(x)
      ) %>%
        postprocess_classification_results(model)
    },
    regression = {
      .rf_regressor_predict(
        model_xptr = model$xptr,
        input = as.matrix(x),
        verbosity = cuml_log_level
      )
    }
  )
}
