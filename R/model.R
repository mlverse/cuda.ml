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
      list(class = c(cls, "cuda_ml_model"), mode = mode, xptr = xptr),
      rlang::dots_list(...)
    )
  )
}

new_model_state <- function(model_state, cls) {
  class(model_state) <- c(cls, "cuda_ml_model_state", class(model_state))

  model_state
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
#' @template cudaml-transform
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_transform <- function(model, x, ...) {
  check_dots_used()
  UseMethod("cuda_ml_transform")
}

#' Apply the inverse transformation defined by a trained cuML model.
#'
#' Given a trained cuML model, apply the inverse transformation defined by that
#' model to an input dataset.
#'
#' @template cudaml-transform
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_inverse_transform <- function(model, x, ...) {
  check_dots_used()
  UseMethod("cuda_ml_inverse_transform")
}

#' Determine whether a CuML model is a classifier.
#'
#' Given a trained CuML model, return \code{TRUE} if the model is a classifier,
#' otherwise \code{FALSE} (e.g., if the model is a regressor).
#'
#' @param model A trained CuML model.
#'
#' @return A logical value indicating whether the model is a classifier.
#'
#' @export
cuda_ml_is_classifier <- function(model) {
  UseMethod("cuda_ml_is_classifier")
}

#' @export
cuda_ml_is_classifier.default <- function(model) {
  report_undefined_fn("cuda_ml_is_classifier", model)
}

#' @export
cuda_ml_is_classifier.cuda_ml_model <- function(model) {
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
#' @return A logical value indicating whether the model supports outputting
#'   class probabilities.
#'
#' @export
cuda_ml_can_predict_class_probabilities <- function(model) {
  UseMethod("cuda_ml_can_predict_class_probabilities")
}

#' @export
cuda_ml_can_predict_class_probabilities.default <- function(model) {
  report_undefined_fn("cuda_ml_can_predict_class_probabilities", model)
}

#' @export
cuda_ml_can_predict_class_probabilities.cuda_ml_model <- function(model) {
  FALSE
}

cuda_ml_can_predict_class_probabilities.cuda_ml_fil <- cuda_ml_is_classifier

cuda_ml_can_predict_class_probabilities.cuda_ml_knn <- cuda_ml_is_classifier

cuda_ml_can_predict_class_probabilities.cuda_ml_rand_forest <- cuda_ml_is_classifier

#' Serialize a CuML model
#'
#' Given a CuML model, serialize its state into a connection.
#'
#' @param model The model object.
#' @param connection An open connection or \code{NULL}. If \code{NULL}, then the
#'   model state is serialized to a raw vector. Default: NULL.
#' @param ... Additional arguments to \code{base::serialize()}.
#'
#' @return \code{NULL} unless \code{connection} is \code{NULL}, in which case
#'   the serialized model state is returned as a raw vector.
#'
#' @seealso \code{\link[base]{serialize}}
#'
#' @export
cuda_ml_serialize <- function(model, connection = NULL, ...) {
  UseMethod("cuda_ml_serialize")
}

#' @rdname cuda_ml_serialize
#'
#' @export
cuda_ml_serialise <- cuda_ml_serialize

#' @export
cuda_ml_serialize.default <- function(model, connection = NULL, ...) {
  report_undefined_fn("cuda_ml_serialize", model)
}

#' @export
cuda_ml_serialize.cuda_ml_model <- function(model, connection = NULL, ...) {
  model_state <- cuda_ml_get_state(model)

  serialize(model_state, connection, ...)
}

cuda_ml_get_state <- function(model) {
  UseMethod("cuda_ml_get_state")
}

cuda_ml_get_state.default <- function(model) {
  stop(
    "Model of type '", paste(class(model), collapse = " "), "' does not ",
    "support serialization."
  )
}

cuda_ml_get_state.cuda_ml_model <- function(model) {
  # Default implementation: assume the entire model object can be serializabled
  # by `base::serialize()`.
  model_state <- list(model = model)

  new_model_state(model_state, cls = NULL)
}

cuda_ml_set_state.cuda_ml_model_state <- function(model_state) {
  # Default implementation: assume the entire model state can be unserialized by
  # `base::unserialize()`.
  model_state$model
}

#' Unserialize a CuML model state
#'
#' Unserialize a CuML model state into a CuML model object.
#'
#' @param connection An open connection or a raw vector.
#' @param ... Additional arguments to \code{base::unserialize()}.
#'
#' @return A unserialized CuML model.
#'
#' @seealso \code{\link[base]{unserialize}}
#'
#' @export
cuda_ml_unserialize <- function(connection, ...) {
  model_state <- unserialize(connection, ...)

  cuda_ml_set_state(model_state)
}

#' @rdname cuda_ml_unserialize
#'
#' @export
cuda_ml_unserialise <- cuda_ml_unserialize


cuda_ml_set_state <- function(model_state) {
  UseMethod("cuda_ml_set_state")
}

cuda_ml_set_state.default <- function(model_state) {
  stop(
    "No unserialization routine found for model state of type '",
    paste(class(model_state), collapse = " "), "'"
  )
}
