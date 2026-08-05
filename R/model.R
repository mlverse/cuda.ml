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

cuda_ml_state_backend_requirements <- function(model_abi) {
  requirements <- list(
    cuda_ml_linear_model_state = character(),
    cuda_ml_logistic_reg_model_state = character(),
    cuda_ml_pca_model_state = "rapids_version",
    cuda_ml_svc_model_state = "rapids_version",
    cuda_ml_svc_ovr_model_state = "rapids_version",
    cuda_ml_svr_model_state = "rapids_version",
    cuda_ml_umap_model_state = "rapids_version",
    cuda_ml_nvforest_model_state = "treelite_version",
    cuda_ml_rand_forest_model_state = "treelite_version"
  )
  required <- requirements[[model_abi]]
  if (is.null(required)) {
    stop(
      "The cuda.ml model-state ABI `",
      model_abi,
      "` is not supported by this cuda.ml version.",
      call. = FALSE
    )
  }
  required
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
  if (!is.list(model_state)) {
    stop("A cuda.ml model state must be a list.", call. = FALSE)
  }
  if (is.null(model_state$schema)) {
    stop("Unversioned cuda.ml model states are unsupported.", call. = FALSE)
  }
  if (!identical(model_state$schema, 1L)) {
    stop(
      "cuda.ml model-state schema ",
      paste(model_state$schema, collapse = ", "),
      " is unsupported; this cuda.ml version supports schema 1.",
      call. = FALSE
    )
  }
  if (
    !is.character(model_state$package_version) ||
      length(model_state$package_version) != 1L ||
      !nzchar(model_state$package_version)
  ) {
    stop(
      "The cuda.ml model-state package-version provenance is missing.",
      call. = FALSE
    )
  }
  if (
    !is.character(model_state$model_abi) ||
      length(model_state$model_abi) != 1L ||
      !nzchar(model_state$model_abi)
  ) {
    stop("The cuda.ml model-state ABI is missing.", call. = FALSE)
  }
  if (!"payload" %in% names(model_state)) {
    stop("The cuda.ml model-state payload is missing.", call. = FALSE)
  }

  required <- cuda_ml_state_backend_requirements(model_state$model_abi)
  if (length(required) > 0L && !is.list(model_state$backend)) {
    stop(
      "The cuda.ml model-state ABI `",
      model_state$model_abi,
      "` is missing its required backend identity.",
      call. = FALSE
    )
  }
  installed <- if (length(required) > 0L) {
    cuda_ml_state_backend_identity()
  } else {
    list()
  }
  for (field in required) {
    state_value <- model_state$backend[[field]]
    if (is.null(state_value)) {
      stop(
        "The cuda.ml model-state ABI `",
        model_state$model_abi,
        "` is missing required backend identity `",
        field,
        "`.",
        call. = FALSE
      )
    }
    if (!identical(state_value, installed[[field]])) {
      stop(
        "The cuda.ml model-state ABI `",
        model_state$model_abi,
        "` requires backend `",
        field,
        "` `",
        paste(state_value, collapse = ", "),
        "`, but this installation provides `",
        paste(installed[[field]], collapse = ", "),
        "`.",
        call. = FALSE
      )
    }
  }

  invisible(model_state)
}

cuda_ml_state_payload <- function(model_state, model_abi) {
  if (!identical(model_state$model_abi, model_abi)) {
    stop(
      "The cuda.ml model-state ABI `",
      model_state$model_abi,
      "` is incompatible with this model; expected `",
      model_abi,
      "`.",
      call. = FALSE
    )
  }
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

#' Transform data with a dimensionality-reduction model
#'
#' These generics apply a fitted dimensionality-reduction mapping. They are
#' distinct from \code{predict()}, which produces outcomes from supervised
#' models and returns tidymodels-style prediction columns.
#'
#' @section Supported methods:
#' \itemize{
#'   \item \code{cuda_ml_transform()} maps predictors into a learned
#'     lower-dimensional representation. It supports fitted TSVD and UMAP
#'     models.
#'   \item \code{cuda_ml_inverse_transform()} maps component coordinates back
#'     toward the original feature space. It supports fitted PCA and TSVD
#'     models.
#' }
#' PCA stores the transformed training input when \code{transform_input = TRUE},
#' but it does not currently provide a method for transforming new data.
#'
#' @template cudaml-transform
#'
#' @seealso \code{\link[stats]{predict}}, \code{\link{cuda_ml_pca}},
#'   \code{\link{cuda_ml_tsvd}}, and \code{\link{cuda_ml_umap}}
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_transform <- function(model, x, ...) {
  check_dots_used()
  UseMethod("cuda_ml_transform")
}

#' @rdname cuda_ml_transform
#' @export
cuda_ml_inverse_transform <- function(model, x, ...) {
  check_dots_used()
  UseMethod("cuda_ml_inverse_transform")
}

#' Save and restore supported cuda.ml models
#'
#' \code{cuda_ml_serialize()} saves the explicit state of a fitted cuda.ml
#' model. \code{cuda_ml_unserialize()} restores that state as a fitted model.
#'
#' @param model The model object.
#' @param connection For \code{cuda_ml_serialize()}, an open connection or
#'   \code{NULL}; \code{NULL} returns the state as a raw vector. For
#'   \code{cuda_ml_unserialize()}, an open connection or a raw vector.
#' @param ... Additional arguments passed to \code{base::serialize()} or
#'   \code{base::unserialize()}.
#' @param device,device_id,layout,precision,default_chunk_size,align_bytes Named
#'   nvForest inference options. They are supported only for nvForest and
#'   random-forest states. When \code{device} is omitted, those states restore
#'   for GPU inference. When \code{precision} is omitted, the saved prediction
#'   precision is used. The remaining omitted options use nvForest defaults.
#'
#' @return \code{cuda_ml_serialize()} returns \code{NULL} when writing to a
#'   connection and otherwise returns a raw vector.
#'   \code{cuda_ml_unserialize()} returns the restored fitted model.
#'
#' @section Supported models:
#' Explicit state is supported for:
#' \itemize{
#'   \item OLS, ridge, lasso, elastic-net, and SGD linear models;
#'   \item logistic and multinomial regression;
#'   \item PCA;
#'   \item binary and one-vs-rest SVC models and SVR models;
#'   \item UMAP;
#'   \item random forests and other nvForest-backed models.
#' }
#' Other cuda.ml models fail during \code{cuda_ml_serialize()} instead of
#' saving native pointers that cannot be used in another R process.
#'
#' @section Compatibility:
#' The cuda.ml package version that created a state is recorded as provenance;
#' a different package version does not by itself prevent restoration. Backend
#' compatibility depends on the model family:
#' \itemize{
#'   \item Linear and logistic-regression states do not require a matching
#'     backend version.
#'   \item PCA, SVC, one-vs-rest SVC, SVR, and UMAP states require the same
#'     RAPIDS version.
#'   \item Random-forest and nvForest states require the same Treelite version.
#' }
#' Other recorded backend details are provenance and do not gate restoration.
#' The package checks the saved model type and required metadata before loading
#' its payload, and rejects unsupported or incompatible states.
#'
#' Random-forest and nvForest states contain device-neutral Treelite model
#' bytes. They retain prediction precision, class labels, preprocessing, and
#' model semantics, but not the inference device, device identifier, tree
#' layout, chunk size, or memory alignment. Select those settings while
#' restoring; omitting \code{device} selects GPU inference.
#'
#' Saving a state to a file connection and restoring it in another R process
#' uses this same contract. The target process must have a compatible cuda.ml
#' installation and must prepare the corresponding backend before prediction:
#' \code{cuda_ml_install()} for GPU operation or
#' \code{cuda_ml_install(device = "cpu")} for CPU-only nvForest inference.
#' \code{bundle::bundle()} stores the same explicit state, so saving a bundle
#' with \code{saveRDS()} and restoring it with \code{readRDS()} and
#' \code{bundle::unbundle()} has the same compatibility requirements. For an
#' nvForest-backed model, the bundle also stores its chosen deployment device
#' separately from the device-neutral state. A bundle is not required for
#' deployment; \code{cuda_ml_serialize()} returns the complete state artifact
#' directly.
#'
#' A restored fit supports the same prediction or transformation operations as
#' the original fit. cuda.ml does not provide warm-start, incremental-training,
#' or fine-tuning operations for either live or restored fits. Refit a model by
#' calling its fitting function again with training data.
#'
#' @seealso \code{\link[base]{serialize}},
#'   \code{\link[base]{unserialize}}, and \code{\link[bundle]{bundle}}
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

#' @rdname cuda_ml_serialize
#' @export
cuda_ml_unserialize <- function(
  connection,
  ...,
  device = NULL,
  device_id = NULL,
  layout = NULL,
  precision = NULL,
  default_chunk_size = NULL,
  align_bytes = NULL
) {
  model_state <- unserialize(connection, ...)

  restore_options <- list()
  if (!missing(device)) {
    restore_options["device"] <- list(device)
  }
  if (!missing(device_id)) {
    restore_options["device_id"] <- list(device_id)
  }
  if (!missing(layout)) {
    restore_options["layout"] <- list(layout)
  }
  if (!missing(precision)) {
    restore_options["precision"] <- list(precision)
  }
  if (!missing(default_chunk_size)) {
    restore_options["default_chunk_size"] <- list(default_chunk_size)
  }
  if (!missing(align_bytes)) {
    restore_options["align_bytes"] <- list(align_bytes)
  }

  if (length(restore_options) == 0L) {
    return(cuda_ml_set_state(model_state))
  }
  cuda_ml_set_state_with_options(model_state, restore_options)
}

cuda_ml_set_state <- function(model_state) {
  cuda_ml_validate_model_state(model_state)
  UseMethod("cuda_ml_set_state")
}

cuda_ml_set_state_with_options <- function(model_state, options) {
  stopifnot(
    "Restore options must be a named list" = is.list(options) &&
      length(options) > 0L &&
      !is.null(names(options)) &&
      all(nzchar(names(options)))
  )
  cuda_ml_validate_model_state(model_state)
  UseMethod("cuda_ml_set_state_with_options")
}

#' @export
cuda_ml_set_state_with_options.default <- function(model_state, options) {
  stop(
    "Restore-time inference options are only supported for nvForest ",
    "and random-forest model states.",
    call. = FALSE
  )
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
#' Converts a model with explicit state into a
#' \code{bundle::bundle()} object. Models without an explicit state fail rather
#' than serializing native pointers.
#'
#' @param x A fitted cuda.ml model.
#' @param ... Unused.
#' @param device For an nvForest-backed model, the device on which the bundle
#'   will restore. \code{NULL} preserves the model's current device. Use
#'   \code{"cpu"} when bundling a GPU-trained random forest for CPU-only
#'   deployment. Other cuda.ml model types do not accept this argument.
#'
#' @inheritSection cuda_ml_serialize Compatibility
#'
#' @seealso \code{\link{cuda_ml_serialize}},
#'   \code{\link{cuda_ml_unserialize}}
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

#' @rdname bundle.cuda_ml_model
#' @exportS3Method bundle::bundle
bundle.cuda_ml_nvforest <- function(x, device = NULL, ...) {
  ellipsis::check_dots_empty()
  device <- match.arg(
    device %||% cuda_ml_nvforest_info(x)$device,
    c("gpu", "cpu")
  )

  bundle::bundle_constr(
    object = list(
      state = cuda_ml_serialize(x),
      device = device
    ),
    situate = bundle::situate_constr(function(object) {
      cuda.ml::cuda_ml_unserialize(
        object$state,
        device = object$device
      )
    }),
    desc_class = class(x)[[1L]]
  )
}
