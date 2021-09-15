#' @importFrom stats predict
NULL

match_eig_algo <- function(eig_algo = c("dq", "jacobi")) {
  eig_algo <- match.arg(eig_algo)

  switch(eig_algo,
    dq = 0L,
    jacobi = 1L
  )
}

new_model <- function(cls,
                      mode = c("classification", "regression"),
                      xptr = NULL, ...) {
  mode <- match.arg(mode)
  do.call(
    hardhat::new_model,
    c(
      list(class = c(cls, "cuml_model"), mode = mode, xptr = xptr),
      rlang::dots_list(...)
    )
  )
}

get_pred_levels <- function(model) {
  levels(model$blueprint$ptypes$outcomes[[1]])
}

postprocess_classification_results <- function(predictions, model) {
  predictions <- as.integer(predictions)
  pred_levels <- get_pred_levels(model)
  predictions <- pred_levels[predictions]
  predictions <- factor(predictions, levels = pred_levels)

  hardhat::spruce_class(predictions)
}

postprocess_class_probabilities <- function(predictions, model) {
  pred_levels <- get_pred_levels(model)

  hardhat::spruce_prob(pred_levels, predictions)
}

postprocess_regression_results <- function(predictions) {
  hardhat::spruce_numeric(predictions)
}

report_undefined_fn <- function(fn_name, x) {
  stop(
    "`", fn_name, "()` is undefined for object of class ",
    paste(class(x), sep = " "),
    call. = FALSE
  )
}

#' Transform data using a trained cuML model.
#'
#' Given a trained cuML model, transform an input dataset using that model.
#'
#' @template cuml-transform
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuml_transform <- function(model, x, ...) {
  check_dots_used()
  UseMethod("cuml_transform")
}

#' Apply the inverse transformation defined by a trained cuML model.
#'
#' Given a trained cuML model, apply the inverse transformation defined by that
#' model to an input dataset.
#'
#' @template cuml-transform
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuml_inverse_transform <- function(model, x, ...) {
  check_dots_used()
  UseMethod("cuml_inverse_transform")
}

#' Determine whether a CuML model is a classifier.
#'
#' Given a trained CuML model, return \code{TRUE} if the model is a classifier,
#' otherwise \code{FALSE} (e.g., if the model is a regressor).
#'
#' @param model A trained CuML model.
#'
#' @export
cuml_is_classifier <- function(model) {
  UseMethod("cuml_is_classifier")
}

#' @export
cuml_is_classifier.default <- function(model) {
  report_undefined_fn("cuml_is_classifier", model)
}

#' @export
cuml_is_classifier.cuml_model <- function(model) {
  identical(model$mode, "classification")
}

#' Determine whether a CuML model can predict class probabilities.
#'
#' Given a trained CuML model, return \code{TRUE} if the model is a classifier
#' and is capable of outputting class probabilities as prediction results (e.g.,
#' if the model is a KNN or an ensemble classifier), otherwise return
#' \code{FALSE}.
#'
#' @param model A trained CuML model.
#'
#' @export
cuml_can_predict_class_probabilities <- function(model) {
  UseMethod("cuml_can_predict_class_probabilities")
}

#' @export
cuml_can_predict_class_probabilities.default <- function(model) {
  report_undefined_fn("cuml_can_predict_class_probabilities", model)
}

#' @export
cuml_can_predict_class_probabilities.cuml_model <- function(model) {
  FALSE
}

cuml_can_predict_class_probabilities.cuml_fil <- cuml_is_classifier

cuml_can_predict_class_probabilities.cuml_knn <- cuml_is_classifier

cuml_can_predict_class_probabilities.cuml_rand_forest <- cuml_is_classifier

#' Make predictions on new data points.
#'
#' Use a trained CuML model to make predictions on new data points.
#' Notice calling \code{cuml_predict()} will be identical to calling the
#' \code{predict()} S3 generic, except for \code{cuml_predict()} also comes
#' with proper documentation on all possible predict options (such as
#' \code{output_class_probabilities}) and will emit a sensible warning message
#' when a predict option is not applicable for a given model.
#'
#' @param model A trained CuML model.
#' @param x A matrix or dataframe containing new data points.
#' @param output_class_probabilities Whether to output class probabilities.
#'   NOTE: setting \code{output_class_probabilities} to \code{TRUE} is only
#'   valid when the model being applied is a classification model and supports
#'   class probabilities output. CuML classification models supporting class
#'   probabilities include \code{knn}, \code{fil}, and \code{rand_forest}.
#'   A warning message will be emitted if \code{output_class_probabilities}
#'   is set to \code{TRUE} or \code{FALSE} but the model being applied does
#'   not support class probabilities output.
#' @param ... Additional arguments to \code{predict()}. Currently unused.
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuml_predict <- function(model, x, output_class_probabilities = NULL, ...) {
  check_dots_used()
  UseMethod("cuml_predict")
}

#' @export
cuml_predict.default <- function(model, x, output_class_probabilities = NULL, ...) {
  report_undefined_fn("cuml_predict", model)
}

#' @export
cuml_predict.cuml_model <- function(model, x, output_class_probabilities = NULL, ...) {
  can_predict_class_probabilities <- cuml_can_predict_class_probabilities(model)

  if (!can_predict_class_probabilities &&
    identical(output_class_probabilities, TRUE)) {
    model_cls <- class(model)
    model_cls <- model_cls[which(startsWith(model_cls, "cuml_"))]
    model_type <- ifelse(cuml_is_classifier(model), "Classifier", "Regressor")
    warning(
      model_type,
      " of type '",
      paste(model_cls, collapse = " "),
      "' does not support outputting class probabilities!"
    )
  }

  if (can_predict_class_probabilities && !is.null(output_class_probabilities)) {
    predict(model, x, as.logical(output_class_probabilities), ...)
  } else {
    predict(model, x, ...)
  }
}

#' Serialize a CuML model
#'
#' Given a CuML model, serialize its state into a connection.
#'
#' @param model The model object.
#' @param connection An open connection or \code{NULL}. If \code{NULL}, then the
#'   model state is serialized to a raw vector.
#' @param ... Additional arguments to \code{base::serialize()}.
#'
#' @seealso \code{\link[base]{serialize}}
#'
#' @export
cuml_serialize <- function(model, connection, ...) {
  UseMethod("cuml_serialize")
}

#' @export
cuml_serialize.default <- function(model, connection, ...) {
  report_undefined_fn("cuml_serialize", model)
}

#' @export
cuml_serialize.cuml_model <- function(model, connection, ...) {
  model_type <- class(model)[[1]]
  get_state_impl <- tryCatch(
    get(paste0(".", model_type, "_get_state")),
    error = function(e) {
      stop("Model of type '", model_type, "' does not support serialization.")
    }
  )

  model %>%
    get_state_impl() %>%
    serialize(connection, ...)
}

#' Unserialize a CuML model state
#'
#' Unserialize a CuML model state into a CuML model object.
#'
#' @param connection An open connection or a raw vector.
#' @param ... Additional arguments to \code{base::unserialize()}.
#'
#' @seealso \code{\link[base]{unserialize}}
#'
#' @export
cuml_unserialize <- function(connection, ...) {
  state <- unserialize(connection, ...)
  model_type <- attributes(state)$model_type
  set_state_impl <- tryCatch(
    get(paste0(".", model_type, "_set_state")),
    error = function(e) {
      stop("Model of type '", model_type, "' does not support unserialization.")
    }
  )

  model <- set_state_impl(state)
  class(model) <- c(model_type, "cuml_model", class(model))

  model
}
