nvforest_match_model_type <- function(model_type) {
  if (is.null(model_type)) {
    return(-1L)
  }

  model_type <- match.arg(
    model_type,
    c(
      "xgboost_ubj",
      "xgboost_json",
      "xgboost_legacy",
      "lightgbm",
      "treelite_checkpoint"
    )
  )

  switch(
    model_type,
    xgboost_ubj = 0L,
    xgboost_json = 1L,
    xgboost_legacy = 2L,
    lightgbm = 3L,
    treelite_checkpoint = 4L
  )
}

nvforest_match_device <- function(device = c("gpu", "cpu")) {
  device <- match.arg(device)
  switch(device, cpu = 0L, gpu = 1L)
}

nvforest_match_layout <- function(
  layout = c("depth_first", "breadth_first", "layered")
) {
  layout <- match.arg(layout)
  switch(layout, depth_first = 0L, breadth_first = 1L, layered = 2L)
}

nvforest_match_precision <- function(
  precision = c("native", "single", "double")
) {
  precision <- match.arg(precision)
  switch(precision, native = -1L, single = 0L, double = 1L)
}

nvforest_task_mode <- function(task_type) {
  switch(
    as.character(task_type),
    "0" = "classification",
    "1" = "regression",
    "2" = "classification",
    stop("nvForest returned an unsupported task type.", call. = FALSE)
  )
}

nvforest_task_name <- function(task_type) {
  switch(
    as.character(task_type),
    "0" = "binary_classification",
    "1" = "regression",
    "2" = "multiclass_classification",
    stop("nvForest returned an unsupported task type.", call. = FALSE)
  )
}

nvforest_code_name <- function(value, names, label) {
  result <- names[[as.character(value)]]
  if (is.null(result)) {
    stop("nvForest returned an unsupported ", label, ".", call. = FALSE)
  }
  result
}

nvforest_inference_options <- function(
  device,
  device_id,
  layout,
  precision,
  default_chunk_size,
  align_bytes
) {
  device <- nvforest_match_device(device)
  stopifnot(
    "`device_id` must be NULL or one non-negative whole number" = is.null(
      device_id
    ) ||
      (is.numeric(device_id) &&
        length(device_id) == 1L &&
        is.finite(device_id) &&
        device_id >= 0 &&
        device_id == as.integer(device_id)),
    "`default_chunk_size` must be NULL or one positive whole number" = is.null(
      default_chunk_size
    ) ||
      (is.numeric(default_chunk_size) &&
        length(default_chunk_size) == 1L &&
        is.finite(default_chunk_size) &&
        default_chunk_size >= 1 &&
        default_chunk_size == as.integer(default_chunk_size)),
    "`align_bytes` must be NULL or one non-negative whole number" = is.null(
      align_bytes
    ) ||
      (is.numeric(align_bytes) &&
        length(align_bytes) == 1L &&
        is.finite(align_bytes) &&
        align_bytes >= 0 &&
        align_bytes == as.integer(align_bytes))
  )

  stopifnot(
    "`device_id` must be NULL for CPU inference" = device == 1L ||
      is.null(device_id),
    "GPU `default_chunk_size` must be a power of two between 1 and 32" = device ==
      0L ||
      is.null(default_chunk_size) ||
      (default_chunk_size <= 32L &&
        bitwAnd(
          as.integer(default_chunk_size),
          as.integer(default_chunk_size) - 1L
        ) ==
          0L)
  )

  list(
    device = device,
    device_id = as.integer(device_id %||% -1L),
    layout = nvforest_match_layout(layout),
    precision = nvforest_match_precision(precision),
    default_chunk_size = as.integer(default_chunk_size %||% 0L),
    align_bytes = as.integer(align_bytes %||% if (device == 0L) 64L else 0L)
  )
}

new_nvforest_model <- function(
  xptr,
  class_levels = NULL,
  inference,
  cls = "cuda_ml_nvforest",
  blueprint = hardhat::default_xy_blueprint()
) {
  info <- .nvforest_model_info(xptr)
  mode <- nvforest_task_mode(info$task_type)

  if (identical(mode, "classification")) {
    class_levels <- class_levels %||%
      as.character(seq_len(info$num_classes) - 1L)
    stopifnot(
      "`class_levels` must contain one unique label per model class" = is.character(
        class_levels
      ) &&
        length(class_levels) == info$num_classes &&
        !anyDuplicated(class_levels)
    )
  } else {
    stopifnot(
      "`class_levels` is only valid for classifiers" = is.null(class_levels)
    )
  }

  new_model(
    cls = cls,
    mode = mode,
    xptr = xptr,
    class_levels = class_levels,
    inference = inference,
    blueprint = blueprint
  )
}

#' Load a tree ensemble with nvForest
#'
#' Loads an XGBoost, LightGBM, or Treelite model with the current nvForest API.
#' The model's classification or regression task is read from Treelite
#' metadata rather than supplied separately.
#'
#' @param model_file Path to a model file.
#' @param model_type File format, or \code{NULL} to infer it. Supported values
#'   are \code{"xgboost_ubj"}, \code{"xgboost_json"},
#'   \code{"xgboost_legacy"}, \code{"lightgbm"}, and
#'   \code{"treelite_checkpoint"}.
#' @param class_levels Optional class labels in model-output order. When omitted,
#'   classifiers use \code{"0"}, \code{"1"}, and so on.
#' @param device Inference device: \code{"gpu"} or \code{"cpu"}.
#' @param device_id GPU device identifier, or \code{NULL} for the current device.
#' @param layout Tree layout.
#' @param precision Native, single, or double precision.
#' @param default_chunk_size Default prediction chunk size, or \code{NULL} to
#'   use nvForest's heuristic.
#' @param align_bytes Memory alignment, or \code{NULL} for the device default.
#'
#' @return An nvForest model for use with \code{predict()}.
#' @export
cuda_ml_nvforest_load_model <- function(
  model_file,
  model_type = NULL,
  class_levels = NULL,
  device = c("gpu", "cpu"),
  device_id = NULL,
  layout = c("depth_first", "breadth_first", "layered"),
  precision = c("native", "single", "double"),
  default_chunk_size = NULL,
  align_bytes = NULL
) {
  stopifnot(
    "`model_file` must name one existing file" = is.character(model_file) &&
      length(model_file) == 1L &&
      file.exists(model_file) &&
      !dir.exists(model_file)
  )
  inference <- nvforest_inference_options(
    device,
    device_id,
    layout,
    precision,
    default_chunk_size,
    align_bytes
  )
  xptr <- .nvforest_load_model(
    filename = normalizePath(model_file, mustWork = TRUE),
    model_type = nvforest_match_model_type(model_type),
    device = inference$device,
    device_id = inference$device_id,
    layout = inference$layout,
    precision = inference$precision,
    default_chunk_size = inference$default_chunk_size,
    align_bytes = inference$align_bytes
  )

  new_nvforest_model(xptr, class_levels, inference)
}

nvforest_predict_matrix <- function(
  object,
  x,
  type = NULL,
  threshold = NULL,
  chunk_size = NULL
) {
  stopifnot(
    "`threshold` must be NULL or one finite number between 0 and 1" = is.null(
      threshold
    ) ||
      (is.numeric(threshold) &&
        length(threshold) == 1L &&
        is.finite(threshold) &&
        threshold >= 0 &&
        threshold <= 1),
    "`chunk_size` must be NULL or one positive whole number" = is.null(
      chunk_size
    ) ||
      (is.numeric(chunk_size) &&
        length(chunk_size) == 1L &&
        is.finite(chunk_size) &&
        chunk_size >= 1 &&
        chunk_size == as.integer(chunk_size))
  )

  info <- .nvforest_model_info(object$xptr)
  if (identical(object$mode, "classification")) {
    stopifnot(
      "`threshold` is only valid for binary classifiers" = is.null(threshold) ||
        info$num_classes == 2L,
      "`threshold` requires probability-valued classifier output" = is.null(
        threshold
      ) ||
        isTRUE(info$has_probability_output)
    )
    type <- match.arg(type %||% "class", c("class", "prob"))
    stopifnot(
      "`threshold` is only valid for class predictions" = is.null(threshold) ||
        identical(type, "class")
    )
    prediction_type <- if (identical(type, "prob")) 1L else 0L
  } else {
    stopifnot(
      "`threshold` is only valid for binary classifiers" = is.null(threshold)
    )
    type <- match.arg(type %||% "numeric", "numeric")
    prediction_type <- 0L
  }

  predictions <- .nvforest_predict(
    model = object$xptr,
    input = x,
    prediction_type = prediction_type,
    threshold = as.numeric(threshold %||% 0.5),
    chunk_size = nvforest_chunk_size(info, chunk_size)
  )

  if (identical(type, "class")) {
    predictions <- as.integer(predictions) + 1L
    out <- postprocess_classification_results(predictions, object)
  } else if (identical(type, "prob")) {
    out <- postprocess_class_probabilities(predictions, object)
  } else {
    out <- postprocess_regression_results(as.numeric(predictions))
  }
  hardhat::validate_prediction_size(out, x)
  out
}

nvforest_predictors <- function(object, new_data) {
  if (is.null(object$blueprint$ptypes)) {
    x <- as.matrix(new_data)
  } else {
    processed <- hardhat::forge(new_data, object$blueprint)
    x <- as.matrix(processed$predictors)
  }
  stopifnot("`new_data` must produce numeric predictors" = is.numeric(x))
  x
}

nvforest_validate_model <- function(object) {
  stopifnot(
    "`object` must be an nvForest-backed model" = inherits(
      object,
      "cuda_ml_nvforest"
    )
  )
  invisible(object)
}

#' Predict with an nvForest model
#'
#' @param object An nvForest-backed model.
#' @param new_data Numeric predictor data.
#' @param type Classification models support \code{"class"} and \code{"prob"};
#'   regression models support \code{"numeric"}.
#' @param threshold Binary classification threshold, or \code{NULL} for 0.5.
#' @param chunk_size Prediction chunk size, or \code{NULL} for the model default.
#' @param ... Unused.
#'
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_nvforest <- function(
  object,
  new_data,
  type = NULL,
  threshold = NULL,
  chunk_size = NULL,
  ...
) {
  check_dots_used()
  x <- nvforest_predictors(object, new_data)
  nvforest_predict_matrix(object, x, type, threshold, chunk_size)
}

#' Inspect an nvForest model
#'
#' @param object An nvForest-backed model.
#'
#' @return A named list of current nvForest model properties.
#' @export
cuda_ml_nvforest_info <- function(object) {
  nvforest_validate_model(object)
  info <- .nvforest_model_info(object$xptr)
  info$task_type <- nvforest_task_name(info$task_type)
  info$device <- nvforest_code_name(
    info$device,
    c("0" = "cpu", "1" = "gpu"),
    "device"
  )
  info$layout <- nvforest_code_name(
    info$layout,
    c("0" = "depth_first", "1" = "breadth_first", "2" = "layered"),
    "layout"
  )
  info$precision <- nvforest_code_name(
    info$precision,
    c("-1" = "native", "0" = "single", "1" = "double"),
    "precision"
  )
  info
}

nvforest_chunk_size <- function(info, chunk_size) {
  if (is.null(chunk_size)) {
    return(0L)
  }

  stopifnot(
    "`chunk_size` must be one positive whole number" = is.numeric(chunk_size) &&
      length(chunk_size) == 1L &&
      is.finite(chunk_size) &&
      chunk_size >= 1 &&
      chunk_size == as.integer(chunk_size),
    "GPU `chunk_size` must be a power of two between 1 and 32" = info$device ==
      0L ||
      (chunk_size <= 32L &&
        bitwAnd(as.integer(chunk_size), as.integer(chunk_size) - 1L) == 0L)
  )
  as.integer(chunk_size)
}

#' Return terminal leaf identifiers
#'
#' @param object An nvForest-backed model.
#' @param new_data Numeric predictor data.
#' @param chunk_size Prediction chunk size, or \code{NULL} for the model default.
#'
#' @return An integer matrix with one row per observation and one column per
#'   tree.
#' @export
cuda_ml_nvforest_leaf_ids <- function(object, new_data, chunk_size = NULL) {
  nvforest_validate_model(object)
  x <- nvforest_predictors(object, new_data)
  predictions <- .nvforest_predict(
    model = object$xptr,
    input = x,
    prediction_type = 2L,
    threshold = 0.5,
    chunk_size = nvforest_chunk_size(
      .nvforest_model_info(object$xptr),
      chunk_size
    )
  )
  storage.mode(predictions) <- "integer"
  predictions
}

#' Return individual-tree predictions
#'
#' @inheritParams cuda_ml_nvforest_leaf_ids
#'
#' @return For scalar-leaf models, a numeric matrix with one column per tree.
#'   For vector-leaf models, a numeric array indexed by observation, tree, and
#'   output.
#' @export
cuda_ml_nvforest_predict_per_tree <- function(
  object,
  new_data,
  chunk_size = NULL
) {
  nvforest_validate_model(object)
  x <- nvforest_predictors(object, new_data)
  predictions <- .nvforest_predict(
    model = object$xptr,
    input = x,
    prediction_type = 3L,
    threshold = 0.5,
    chunk_size = nvforest_chunk_size(
      .nvforest_model_info(object$xptr),
      chunk_size
    )
  )
  info <- cuda_ml_nvforest_info(object)
  if (!info$has_vector_leaves) {
    return(predictions)
  }

  tree_output <- array(
    predictions,
    dim = c(nrow(x), info$num_outputs, info$num_trees)
  )
  aperm(tree_output, c(1L, 3L, 2L))
}

nvforest_model_payload <- function(model) {
  list(
    model = .nvforest_serialize(model$xptr),
    class_levels = model$class_levels,
    inference = model$inference,
    averaged_vector_leaf_probabilities = inherits(
      model,
      "cuda_ml_rand_forest"
    ) &&
      identical(model$mode, "classification"),
    blueprint = model$blueprint
  )
}

nvforest_unserialize_payload <- function(payload, cls) {
  inference <- payload$inference
  xptr <- .nvforest_unserialize(
    bytes = payload$model,
    device = inference$device,
    device_id = inference$device_id,
    layout = inference$layout,
    precision = inference$precision,
    default_chunk_size = inference$default_chunk_size,
    align_bytes = inference$align_bytes,
    averaged_vector_leaf_probabilities = payload$averaged_vector_leaf_probabilities
  )
  new_nvforest_model(
    xptr,
    payload$class_levels,
    inference,
    cls = cls,
    blueprint = payload$blueprint
  )
}

#' @export
cuda_ml_get_state.cuda_ml_nvforest <- function(model) {
  new_model_state(
    nvforest_model_payload(model),
    "cuda_ml_nvforest_model_state"
  )
}

#' @export
cuda_ml_set_state.cuda_ml_nvforest_model_state <- function(model_state) {
  payload <- cuda_ml_state_payload(
    model_state,
    "cuda_ml_nvforest_model_state"
  )
  nvforest_unserialize_payload(payload, "cuda_ml_nvforest")
}
