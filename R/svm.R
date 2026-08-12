svm_match_kernel_type <- function(
  kernel = c("rbf", "tanh", "polynomial", "linear")
) {
  kernel <- match.arg(kernel)

  switch(kernel, linear = 0L, polynomial = 1L, rbf = 2L, tanh = 3L)
}

#' Train a SVM model.
#'
#' Train a Support Vector Machine model for classification or regression tasks.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @param cost A positive number for the cost of predicting a sample within or
#'   on the wrong side of the margin. Default: 1.
#' @param kernel Type of the SVM kernel function (must be one of "rbf", "tanh",
#'   "polynomial", or "linear"). Default: "rbf".
#' @param gamma The gamma coefficient (only relevant to polynomial, RBF, and
#'   tanh kernel functions, see explanations below).
#'   Default: 1 / (num features).
#'
#'   The following kernels are implemented:
#'   - RBF K(x_1, x_2) = exp(-gamma |x_1-x_2|^2)
#'   - TANH K(x_1, x_2) = tanh(gamma <x_1,x_2> + coef0)
#'   - POLYNOMIAL K(x_1, x_2) = (gamma <x_1,x_2> + coef0)^degree
#'   - LINEAR K(x_1,x_2) = <x_1,x_2>,
#'
#'   where < , > denotes the dot product.
#' @param coef0 The 0th coefficient (only applicable to polynomial and tanh
#'   kernel functions, see explanations below). Default: 0.
#'
#'   The following kernels are implemented:
#'   - RBF K(x_1, x_2) = exp(-gamma |x_1-x_2|^2)
#'   - TANH K(x_1, x_2) = tanh(gamma <x_1,x_2> + coef0)
#'   - POLYNOMIAL K(x_1, x_2) = (gamma <x_1,x_2> + coef0)^degree
#'   - LINEAR K(x_1,x_2) = <x_1,x_2>,
#'
#'   where < , > denotes the dot product.
#' @param degree Degree of the polynomial kernel function (note: not applicable
#'   to other kernel types, see explanations below). Default: 3.
#'
#'   The following kernels are implemented:
#'   - RBF K(x_1, x_2) = exp(-gamma |x_1-x_2|^2)
#'   - TANH K(x_1, x_2) = tanh(gamma <x_1,x_2> + coef0)
#'   - POLYNOMIAL K(x_1, x_2) = (gamma <x_1,x_2> + coef0)^degree
#'   - LINEAR K(x_1,x_2) = <x_1,x_2>,
#'
#'   where < , > denotes the dot product.
#' @param tol Tolerance to stop fitting. Default: 1e-3.
#' @param max_iter Maximum number of outer iterations in SmoSolver.
#'   Default: 100 * (num samples).
#' @param nochange_steps Number of steps with no change w.r.t convergence.
#'   Default: 1000.
#' @param cache_size Size of kernel cache (MiB) in device memory. Default: 1024.
#' @param epsilon Epsilon parameter of the epsilon-SVR model. There is no
#'   penalty for points that are predicted within the epsilon-tube around the
#'   target values. Please note this parameter is only relevant for regression
#'   tasks. Default: 0.1.
#' @param sample_weights Optional weight assigned to each input data point.
#'
#' @return A SVM classifier / regressor object that can be used with the
#'   'predict' S3 generic to make predictions on new data points.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (interactive() && cuda_ml_backend_info()$runtime_installed) {
#'   # Classification
#'
#'   model <- cuda_ml_svm(
#'     formula = Class ~ .,
#'     data = modeldata::two_class_dat,
#'     kernel = "rbf"
#'   )
#'
#'   predictions <- predict(
#'     model,
#'     modeldata::two_class_dat[names(modeldata::two_class_dat) != "Class"]
#'   )
#'
#'   # Regression
#'
#'   model <- cuda_ml_svm(
#'     formula = mpg ~ .,
#'     data = mtcars,
#'     kernel = "rbf"
#'   )
#'
#'   predictions <- predict(model, mtcars)
#' }
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_svm <- function(x, ...) {
  UseMethod("cuda_ml_svm")
}

#' @rdname cuda_ml_svm
#' @export
cuda_ml_svm.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_svm", x)
}

#' @rdname cuda_ml_svm
#' @export
cuda_ml_svm.data.frame <- function(
  x,
  y,
  cost = 1,
  kernel = c("rbf", "tanh", "polynomial", "linear"),
  gamma = NULL,
  coef0 = 0,
  degree = 3L,
  tol = 1e-3,
  max_iter = NULL,
  nochange_steps = 1000L,
  cache_size = 1024,
  epsilon = 0.1,
  sample_weights = NULL,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)

  cuda_ml_svm_bridge(
    processed = processed,
    cost = cost,
    kernel = kernel,
    gamma = gamma,
    coef0 = coef0,
    degree = degree,
    tol = tol,
    max_iter = max_iter,
    nochange_steps = nochange_steps,
    cache_size = cache_size,
    epsilon = epsilon,
    sample_weights = sample_weights
  )
}

#' @rdname cuda_ml_svm
#' @export
cuda_ml_svm.matrix <- function(
  x,
  y,
  cost = 1,
  kernel = c("rbf", "tanh", "polynomial", "linear"),
  gamma = NULL,
  coef0 = 0,
  degree = 3L,
  tol = 1e-3,
  max_iter = NULL,
  nochange_steps = 1000L,
  cache_size = 1024,
  epsilon = 0.1,
  sample_weights = NULL,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, y)

  cuda_ml_svm_bridge(
    processed,
    cost = cost,
    kernel = kernel,
    gamma = gamma,
    coef0 = coef0,
    degree = degree,
    tol = tol,
    max_iter = max_iter,
    nochange_steps = nochange_steps,
    cache_size = cache_size,
    epsilon = epsilon,
    sample_weights = sample_weights
  )
}

#' @rdname cuda_ml_svm
#' @export
cuda_ml_svm.formula <- function(
  formula,
  data,
  cost = 1,
  kernel = c("rbf", "tanh", "polynomial", "linear"),
  gamma = NULL,
  coef0 = 0,
  degree = 3L,
  tol = 1e-3,
  max_iter = NULL,
  nochange_steps = 1000L,
  cache_size = 1024,
  epsilon = 0.1,
  sample_weights = NULL,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(formula, data)

  cuda_ml_svm_bridge(
    processed,
    cost = cost,
    kernel = kernel,
    gamma = gamma,
    coef0 = coef0,
    degree = degree,
    tol = tol,
    max_iter = max_iter,
    nochange_steps = nochange_steps,
    cache_size = cache_size,
    epsilon = epsilon,
    sample_weights = sample_weights
  )
}

#' @rdname cuda_ml_svm
#' @export
cuda_ml_svm.recipe <- function(
  x,
  data,
  cost = 1,
  kernel = c("rbf", "tanh", "polynomial", "linear"),
  gamma = NULL,
  coef0 = 0,
  degree = 3L,
  tol = 1e-3,
  max_iter = NULL,
  nochange_steps = 1000L,
  cache_size = 1024,
  epsilon = 0.1,
  sample_weights = NULL,
  ...
) {
  check_dots_used()
  processed <- hardhat::mold(x, data)

  cuda_ml_svm_bridge(
    processed,
    cost = cost,
    kernel = kernel,
    gamma = gamma,
    coef0 = coef0,
    degree = degree,
    tol = tol,
    max_iter = max_iter,
    nochange_steps = nochange_steps,
    cache_size = cache_size,
    epsilon = epsilon,
    sample_weights = sample_weights
  )
}

cuda_ml_svm_bridge <- function(
  processed,
  cost,
  kernel,
  gamma,
  coef0,
  degree,
  tol,
  max_iter,
  nochange_steps,
  cache_size,
  epsilon,
  sample_weights
) {
  hardhat::validate_predictors_are_numeric(processed$predictors)
  hardhat::validate_outcomes_are_univariate(processed$outcomes)
  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  if (is.factor(y)) {
    validate_classification_outcome(y)
  }

  gamma <- gamma %||% 1.0 / ncol(x)
  max_iter <- max_iter %||% 100L * nrow(x)
  kernel <- svm_match_kernel_type(kernel)

  svm_fit_impl <- (if (is.factor(y)) {
    # classification
    ylevels <- levels(y)
    if (length(ylevels) > 2) {
      cuda_ml_svm_classification_multiclass_impl
    } else {
      cuda_ml_svm_classification_binary_impl
    }
  } else {
    cuda_ml_svm_regression_impl
  })

  svm_fit_impl(
    processed = processed,
    cost = cost,
    kernel = kernel,
    gamma = gamma,
    coef0 = coef0,
    degree = degree,
    tol = tol,
    max_iter = max_iter,
    nochange_steps = nochange_steps,
    cache_size = cache_size,
    epsilon = epsilon,
    sample_weights = sample_weights
  )
}

cuda_ml_svm_classification_multiclass_impl <- function(
  processed,
  cost,
  kernel,
  gamma,
  coef0,
  degree,
  tol,
  max_iter,
  nochange_steps,
  cache_size,
  epsilon,
  sample_weights
) {
  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]
  ylevels <- levels(y)

  # implement a one-vs-rest strategy for multi-class classification
  models <- list()
  for (idx in seq_along(ylevels)) {
    ovr_labels <- as.integer(y == ylevels[[idx]])
    model_xptr <- .svc_fit(
      input = x,
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
      verbosity = 6L
    )
    model <- new_model(
      cls = c("cuda_ml_svc", "cuda_ml_svm"),
      mode = "classification",
      xptr = model_xptr
    )

    models <- append(models, list(model))
  }

  new_model(
    cls = c("cuda_ml_svc_ovr", "cuda_ml_svm"),
    mode = "classification",
    xptr = models,
    multiclass = TRUE,
    blueprint = processed$blueprint
  )
}

#' @export
cuda_ml_get_state.cuda_ml_svc_ovr <- function(model) {
  model_state <- list(
    ovr_model_states = lapply(model$xptr, function(x) cuda_ml_get_state(x)),
    blueprint = model$blueprint
  )

  new_model_state(model_state, "cuda_ml_svc_ovr_model_state")
}

#' @export
cuda_ml_set_state.cuda_ml_svc_ovr_model_state <- function(model_state) {
  payload <- cuda_ml_state_payload(model_state, "cuda_ml_svc_ovr_model_state")
  new_model(
    cls = c("cuda_ml_svc_ovr", "cuda_ml_svm"),
    mode = "classification",
    xptr = lapply(payload$ovr_model_states, function(x) cuda_ml_set_state(x)),
    multiclass = TRUE,
    blueprint = payload$blueprint
  )
}

cuda_ml_svm_classification_binary_impl <- function(
  processed,
  cost,
  kernel,
  gamma,
  coef0,
  degree,
  tol,
  max_iter,
  nochange_steps,
  cache_size,
  epsilon,
  sample_weights
) {
  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  model_xptr <- .svc_fit(
    input = x,
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
    verbosity = 6L
  )

  new_model(
    cls = c("cuda_ml_svc", "cuda_ml_svm"),
    mode = "classification",
    xptr = model_xptr,
    multiclass = FALSE,
    blueprint = processed$blueprint
  )
}

#' @export
cuda_ml_get_state.cuda_ml_svc <- function(model) {
  model_state <- list(
    model_state = .svc_get_state(model$xptr),
    blueprint = model$blueprint
  )

  new_model_state(model_state, "cuda_ml_svc_model_state")
}

#' @export
cuda_ml_set_state.cuda_ml_svc_model_state <- function(model_state) {
  payload <- cuda_ml_state_payload(model_state, "cuda_ml_svc_model_state")
  new_model(
    cls = c("cuda_ml_svc", "cuda_ml_svm"),
    mode = "classification",
    xptr = .svc_set_state(payload$model_state),
    multiclass = FALSE,
    blueprint = payload$blueprint
  )
}

cuda_ml_svm_regression_impl <- function(
  processed,
  cost,
  kernel,
  gamma,
  coef0,
  degree,
  tol,
  max_iter,
  nochange_steps,
  cache_size,
  epsilon,
  sample_weights
) {
  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  model_xptr <- .svr_fit(
    X = x,
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
    verbosity = 6L
  )

  new_model(
    cls = c("cuda_ml_svr", "cuda_ml_svm"),
    mode = "regression",
    xptr = model_xptr,
    blueprint = processed$blueprint
  )
}

#' @export
cuda_ml_get_state.cuda_ml_svr <- function(model) {
  model_state <- list(
    model_state = .svr_get_state(model$xptr),
    blueprint = model$blueprint
  )

  new_model_state(model_state, "cuda_ml_svr_model_state")
}

#' @export
cuda_ml_set_state.cuda_ml_svr_model_state <- function(model_state) {
  payload <- cuda_ml_state_payload(model_state, "cuda_ml_svr_model_state")
  new_model(
    cls = c("cuda_ml_svr", "cuda_ml_svm"),
    mode = "regression",
    xptr = .svr_set_state(payload$model_state),
    blueprint = payload$blueprint
  )
}

#' Make predictions on new data points.
#'
#' Make predictions on new data points using a cuML SVM model.
#'
#' @template predict
#'
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_svm <- function(object, new_data, ...) {
  check_dots_used()

  processed <- hardhat::forge(new_data, object$blueprint)

  predict_cuda_ml_svm_bridge(model = object, processed = processed)
}

predict_cuda_ml_svm_bridge <- function(model, processed) {
  svm_predict_impl <- switch(
    model$mode,
    classification = (if (model$multiclass) {
      predict_cuda_ml_svm_classification_multiclass_impl
    } else {
      predict_cuda_ml_svm_classification_binary_impl
    }),
    regression = (predict_cuda_ml_svm_regression_impl)
  )

  out <- svm_predict_impl(model = model, processed = processed)
  hardhat::validate_prediction_size(out, processed$predictors)

  out
}

predict_cuda_ml_svm_classification_multiclass_impl <- function(
  model,
  processed
) {
  pred_levels <- get_pred_levels(model)

  scores <- lapply(
    seq_along(pred_levels),
    function(label_idx) {
      .svc_predict(
        model_xptr = model$xptr[[label_idx]]$xptr,
        input = as.matrix(processed$predictors),
        predict_class = FALSE
      )
    }
  )

  preds <- sapply(
    seq_len(nrow(processed$predictors)),
    function(row_idx) {
      row_scores <- lapply(
        seq_along(pred_levels),
        function(label_idx) scores[[label_idx]][[row_idx]]
      )

      which.max(row_scores)
    }
  )

  postprocess_classification_results(preds, model)
}

predict_cuda_ml_svm_classification_binary_impl <- function(model, processed) {
  preds <- .svc_predict(
    model_xptr = model$xptr,
    input = as.matrix(processed$predictors),
    predict_class = TRUE
  )

  postprocess_classification_results(preds, model)
}

predict_cuda_ml_svm_regression_impl <- function(model, processed) {
  preds <- .svr_predict(
    svr_xptr = model$xptr,
    X = as.matrix(processed$predictors)
  )

  postprocess_regression_results(preds)
}

# register the CuML-based rand_forest model for parsnip
register_svm_model <- function(pkgname) {
  for (model in c(paste0("svm_", c("rbf", "poly", "linear")))) {
    for (mode in c("classification", "regression")) {
      parsnip::set_model_engine(model = model, mode = mode, eng = pkgname)
    }
    parsnip::set_dependency(model = model, eng = pkgname, pkg = pkgname)

    parsnip::set_model_arg(
      model = model,
      eng = pkgname,
      parsnip = "cost",
      original = "cost",
      func = list(pkg = "dials", fun = "cost", range = c(-10, 5)),
      has_submodel = FALSE
    )

    parsnip::set_model_arg(
      model = model,
      eng = pkgname,
      parsnip = "margin",
      original = "epsilon",
      func = list(pkg = "dials", fun = "svm_margin"),
      has_submodel = FALSE
    )
  }

  parsnip::set_model_arg(
    model = "svm_rbf",
    eng = pkgname,
    parsnip = "rbf_sigma",
    original = "gamma",
    func = list(pkg = "dials", fun = "rbf_sigma"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "svm_poly",
    eng = pkgname,
    parsnip = "degree",
    original = "degree",
    func = list(pkg = "dials", fun = "degree"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "svm_poly",
    eng = pkgname,
    parsnip = "scale_factor",
    original = "gamma",
    func = list(pkg = "dials", fun = "scale_factor"),
    has_submodel = FALSE
  )

  for (kernel in c("rbf", "poly", "linear")) {
    model <- paste0("svm_", kernel)

    for (mode in c("classification", "regression")) {
      parsnip::set_fit(
        model = model,
        eng = pkgname,
        mode = mode,
        value = list(
          interface = "formula",
          protect = c("formula", "data"),
          func = c(pkg = pkgname, fun = "cuda_ml_svm"),
          defaults = list(kernel = kernel)
        )
      )

      parsnip::set_encoding(
        model = model,
        eng = pkgname,
        mode = mode,
        options = list(
          predictor_indicators = "none",
          compute_intercept = FALSE,
          remove_intercept = FALSE,
          allow_sparse_x = FALSE
        )
      )
    }

    parsnip::set_pred(
      model = model,
      eng = pkgname,
      mode = "classification",
      type = "class",
      value = list(
        pre = NULL,
        post = NULL,
        func = c(fun = "predict"),
        args = list(
          quote(object$fit),
          quote(new_data)
        )
      )
    )

    parsnip::set_pred(
      model = model,
      eng = pkgname,
      mode = "regression",
      type = "numeric",
      value = list(
        pre = NULL,
        post = NULL,
        func = c(fun = "predict"),
        args = list(
          quote(object$fit),
          quote(new_data)
        )
      )
    )
  }
}
