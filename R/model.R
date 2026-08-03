#' @importFrom stats predict
NULL

match_eig_algo <- function(eig_algo = c("dq", "jacobi")) {
  eig_algo <- match.arg(eig_algo)

  switch(eig_algo, dq = 0L, jacobi = 1L)
}

new_model <- function(
  cls,
  mode = c("classification", "regression"),
  xptr = NULL,
  ...
) {
  mode <- match.arg(mode)
  do.call(
    hardhat::new_model,
    c(
      list(class = c(cls, "cuda_ml_model"), mode = mode, xptr = xptr),
      rlang::dots_list(...)
    )
  )
}

validate_classification_outcome <- function(outcome) {
  stopifnot(
    "The classification outcome must be a factor" = is.factor(outcome),
    "The outcome must contain at least two factor levels" = nlevels(outcome) >=
      2L,
    "Every outcome factor level must be represented in the training data" = all(
      tabulate(as.integer(outcome), nbins = nlevels(outcome)) > 0L
    )
  )
  invisible(outcome)
}

cuda_ml_state_backend_identity <- function() {
  info <- cuda_ml_backend_info()
  info[c(
    "cuda_version",
    "rapids_version",
    "nvforest_version",
    "treelite_version",
    "platform"
  )]
}

new_model_state <- function(payload, cls) {
  stopifnot(
    "A concrete model-state class is required" = is.character(cls) &&
      length(cls) == 1L &&
      nzchar(cls)
  )

  structure(
    list(
      schema = 1L,
      package_version = as.character(utils::packageVersion("cuda.ml")),
      backend = cuda_ml_state_backend_identity(),
      model_abi = cls,
      payload = payload
    ),
    class = c(cls, "cuda_ml_model_state")
  )
}

cuda_ml_validate_model_state <- function(model_state) {
  stopifnot(
    "Unversioned cuda.ml model states are unsupported" = is.list(model_state) &&
      identical(model_state$schema, 1L),
    "The cuda.ml model-state package version is incompatible" = identical(
      model_state$package_version,
      as.character(utils::packageVersion("cuda.ml"))
    ),
    "The cuda.ml model-state backend is incompatible" = identical(
      model_state$backend,
      cuda_ml_state_backend_identity()
    ),
    "The cuda.ml model-state ABI is missing" = is.character(
      model_state$model_abi
    ) &&
      length(model_state$model_abi) == 1L &&
      nzchar(model_state$model_abi),
    "The cuda.ml model-state payload is missing" = "payload" %in%
      names(model_state)
  )

  invisible(model_state)
}

cuda_ml_state_payload <- function(model_state, model_abi) {
  stopifnot(
    "The cuda.ml model-state ABI is incompatible" = identical(
      model_state$model_abi,
      model_abi
    )
  )
  model_state$payload
}

get_pred_levels <- function(model) {
  if (!is.null(model$class_levels)) {
    return(model$class_levels)
  }

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
    "`",
    fn_name,
    "()` is undefined for object of class ",
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

#' @export
cuda_ml_get_state.default <- function(model) {
  stop(
    "Model of type '",
    paste(class(model), collapse = " "),
    "' does not ",
    "support serialization."
  )
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

cuda_ml_set_state <- function(model_state) {
  cuda_ml_validate_model_state(model_state)
  UseMethod("cuda_ml_set_state")
}

#' @export
cuda_ml_set_state.default <- function(model_state) {
  stop(
    "No unserialization routine found for model state of type '",
    paste(class(model_state), collapse = " "),
    "'"
  )
}

#' Bundle a cuda.ml model
#'
#' Converts a model with an explicit portable state into a
#' \code{bundle::bundle()} object. Models without an explicit state fail rather
#' than serializing native pointers.
#'
#' @param x A fitted cuda.ml model.
#' @param ... Unused.
#'
#' @exportS3Method bundle::bundle
bundle.cuda_ml_model <- function(x, ...) {
  ellipsis::check_dots_empty()

  bundle::bundle_constr(
    object = cuda_ml_serialize(x),
    situate = bundle::situate_constr(function(object) {
      cuda.ml::cuda_ml_unserialize(object)
    }),
    desc_class = class(x)[[1L]]
  )
}
