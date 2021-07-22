match_kernel_type <- function(kernel = c("rbf", "tanh", "polynomial", "linear")) {
  kernel <- match.arg(kernel)

  switch(kernel,
    linear = 0L,
    polynomial = 1L,
    rbf = 2L,
    tanh = 3L
  )
}

#' Train a SVM model.
#'
#' Train a Support Vector Machine model for classification or regression tasks.
#'
#' @template model-with-numeric-input
#' @template supervised-model-with-numeric-output
#' @template supervised-model-formula-spec
#' @template supervised-model-classification-or-regression-mode
#' @template cuml-log-level
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
#' @param epsilon Espsilon parameter of the epsilon-SVR model. There is no
#'   penalty for points that are predicted within the epsilon-tube around the
#'   target values. Please note this parameter is only relevant for regression
#'   tasks. Default: 0.1.
#' @param sample_weights Optional weight assigned to each input data point.
#'
#' @return A Support Vector Machine classifier / regressor object that can be
#'   used with the 'predict' S3 generic to make predictions on new data points.
#'
#' @examples
#'
#' library(cuml4r)
#'
#' model <- cuml_svm(
#'   iris[1:100,],
#'   formula = Species ~ .,
#'   mode = "classification",
#'   kernel = "rbf"
#' )
#'
#' predictions <- predict(model, iris[1:100,])
#'
#' cat("Iris species predictions: ", predictions, "\n")
#'
#' model <- cuml_svm(
#'   mtcars,
#'   formula = mpg ~ .,
#'   mode = "regression",
#'   kernel = "rbf"
#' )
#'
#' predictions <- predict(model, mtcars)
#'
#' cat("MPG predictions:", predictions, "\n")
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
                     epsilon = 0.1,
                     sample_weights = NULL,
                     cuml_log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  mode <- match.arg(mode)
  kernel <- match_kernel_type(kernel)
  cuml_log_level <- match_cuml_log_level(cuml_log_level)
  c(x, y) %<-% process_input_and_label_specs(x, y, formula)

  switch(mode,
    classification = {
      unique_labels <- unique(y)
      if (length(unique_labels) > 2) {
        # implement a one-vs-rest strategy for multi-class classification
        models <- list()
        for (label in unique_labels) {
          ovr_labels <- as.integer(y == label)
          models[[label]] <- new_model(
            cls = "cuml_svm",
            mode = "classification",
            xptr = .svc_fit(
              input = as.matrix(x),
              labels = ovr_labels,
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
            )
          )
        }

        new_model(
          cls = "cuml_svm_multi_class",
          mode = mode,
          xptr = models,
          formula = formula,
          unique_labels = unique_labels
        )
      } else {
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
      }
    },
    regression = {
      new_model(
        cls = "cuml_svm",
        mode = mode,
        xptr = .svr_fit(
          X = as.matrix(x),
          y = as.numeric(y),
          cost = as.numeric(cost),
          kernel = kernel,
          gamma = as.numeric(gamma),
          coef0 = as.numeric(coef0),
          degree = as.integer(degree),
          tol = as.numeric(tol),
          max_iter = as.integer(max_iter),
          nochange_steps = as.integer(nochange_steps),
          cache_size = as.numeric(cache_size),
          epsilon = as.numeric(epsilon),
          sample_weights = as.numeric(sample_weights),
          verbosity = cuml_log_level
        ),
        formula = formula,
        resp_var = y
      )
    }
  )
}

#' @export
predict.cuml_svm <- function(object, ...) {
  model <- object
  x <- process_input_specs(rlang::dots_list(...)[[1]], model)

  switch(model$mode,
    classification = {
      .svc_predict(
        model_xptr = model$xptr,
        input = as.matrix(x),
        predict_class = TRUE
      ) %>%
        postprocess_classification_results(model)
    },
    regression = {
      .svr_predict(
        svr_xptr = model$xptr,
        X = as.matrix(x)
      )
    }
  )
}

#' @export
predict.cuml_svm_multi_class <- function(object, ...) {
  model <- object
  x <- process_input_specs(rlang::dots_list(...)[[1]], model)

  scores <- seq_along(model$unique_labels) %>%
    lapply(
      function(label_idx) {
        .svc_predict(
          model_xptr = model$xptr[[model$unique_labels[[label_idx]]]]$xptr,
          input = as.matrix(x),
          predict_class = FALSE
        )
      }
    )

  seq_len(nrow(x)) %>%
    sapply(
      function(input_idx) {
        optimal_label_idx <- seq_along(model$unique_labels) %>%
          lapply(function(label_idx) scores[[label_idx]][[input_idx]]) %>%
          which.max()

        model$unique_labels[[optimal_label_idx]]
      }
    )
}
