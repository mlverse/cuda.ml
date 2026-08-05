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
  info <- nvforest_native_model_info(xptr)
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
#' @param model_type File format, or \code{NULL} to infer it from a recognized
#'   filename suffix. See \strong{Model formats}.
#' @param class_levels Optional class labels in model-output order. When omitted,
#'   classifiers use \code{"0"}, \code{"1"}, and so on.
#' @param device Inference device: \code{"gpu"} or \code{"cpu"}. The default is
#'   \code{"gpu"}.
#' @param device_id GPU device identifier, or \code{NULL} for the current device.
#' @param layout Tree layout.
#' @param precision Native, single, or double precision.
#' @param default_chunk_size Default prediction chunk size, or \code{NULL} to
#'   use nvForest's heuristic.
#' @param align_bytes Memory alignment, or \code{NULL} for the device default.
#'
#' @return An nvForest model for use with \code{predict()}.
#'
#' @section Model formats:
#' The supported \code{model_type} values are:
#' \itemize{
#'   \item \code{"xgboost_ubj"} for XGBoost UBJSON;
#'   \item \code{"xgboost_json"} for XGBoost JSON;
#'   \item \code{"xgboost_legacy"} for the legacy XGBoost binary format;
#'   \item \code{"lightgbm"} for LightGBM text models; and
#'   \item \code{"treelite_checkpoint"} for Treelite checkpoints.
#' }
#' When \code{model_type = NULL}, the format is inferred only from the
#' case-insensitive filename suffix: \file{.ubj}, \file{.json}, \file{.model},
#' and \file{.txt} map to \code{"xgboost_ubj"}, \code{"xgboost_json"},
#' \code{"xgboost_legacy"}, and \code{"lightgbm"}, respectively. Treelite
#' checkpoints have no inferred suffix and require
#' \code{model_type = "treelite_checkpoint"}. Inference does not inspect file
#' contents; use an explicit type when the suffix does not identify the format.
#'
#' @section Runtime requirements:
#' GPU inference requires the complete, roughly 1.6 GiB runtime installed by
#' \code{\link{cuda_ml_install}()} and a supported NVIDIA GPU and driver.
#' For CPU-only deployment, install the separate, roughly 3 MiB backend with
#' \code{cuda_ml_install(device = "cpu")}. It does not install cuML or the
#' complete managed CUDA and RAPIDS runtime, and it requires neither an NVIDIA
#' GPU nor an NVIDIA driver. An existing complete backend installation can also
#' execute nvForest models on CPU; the separate backend avoids that runtime in
#' CPU-only environments.
#'
#' @section Persistence:
#' Persist nvForest models with \code{\link{cuda_ml_serialize}()} and restore
#' them with \code{\link{cuda_ml_unserialize}()}. Current states do not record
#' CPU or GPU placement. Select the deployment device when restoring, for
#' example \code{cuda_ml_unserialize(state, device = "cpu")}; GPU is the
#' default. Tree layout, chunk size, memory alignment, and GPU device identifier
#' are likewise restore-time settings. Prediction precision is retained unless
#' explicitly overridden. cuda.ml validates the saved state and selected
#' backend before restoration.
#'
#' To create a standard Treelite checkpoint together with the metadata needed
#' for a complete cuda.ml round-trip, use
#' \code{\link{cuda_ml_nvforest_export}()} and restore the pair with
#' \code{\link{cuda_ml_nvforest_import}()}.
#'
#' @seealso \code{\link{cuda_ml_nvforest_info}()},
#'   \code{\link{cuda_ml_nvforest_leaf_ids}()},
#'   \code{\link{cuda_ml_nvforest_predict_per_tree}()}, and
#'   \code{vignette("nvforest")}
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
  xptr <- nvforest_native_load_model(
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

  info <- nvforest_native_model_info(object$xptr)
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

  predictions <- nvforest_native_predict(
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
#'   regression models support \code{"numeric"}. Probability prediction is
#'   available only when
#'   \code{cuda_ml_nvforest_info(object)$has_probability_output} is true.
#'   Unsupported Treelite postprocessors fail explicitly.
#' @param threshold Binary classification threshold, or \code{NULL} for 0.5.
#' @param chunk_size Native prediction chunk size, or \code{NULL} for the model
#'   default. It controls native batching and does not limit the size of the
#'   returned R object.
#' @param ... Unused.
#'
#' @return A tibble with \code{.pred} for regression,
#'   \code{.pred_class} for class prediction, or one probability column named
#'   \code{.pred_<level>} for each class.
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
#' @return A named list with:
#' \describe{
#'   \item{\code{task_type}}{One of \code{"binary_classification"},
#'     \code{"multiclass_classification"}, or \code{"regression"}.}
#'   \item{\code{num_classes}, \code{num_features}, \code{num_outputs}, and
#'     \code{num_trees}}{Model dimensions.}
#'   \item{\code{has_vector_leaves}, \code{average_tree_output}, and
#'     \code{has_probability_output}}{Logical model properties.}
#'   \item{\code{device}, \code{device_id}, \code{layout}, and
#'     \code{precision}}{Resolved inference configuration.}
#'   \item{\code{default_chunk_size} and \code{align_bytes}}{Native chunk and
#'     memory-alignment settings.}
#'   \item{\code{treelite_postprocessor}}{The model's Treelite
#'     postprocessor.}
#' }
#' @export
cuda_ml_nvforest_info <- function(object) {
  nvforest_validate_model(object)
  info <- nvforest_native_model_info(object$xptr)
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
#' @param chunk_size Native prediction chunk size, or \code{NULL} for the model
#'   default. It controls native batching and does not limit the size of the
#'   returned R object.
#'
#' @return An integer matrix with one row per observation and one column per
#'   tree.
#' @export
cuda_ml_nvforest_leaf_ids <- function(object, new_data, chunk_size = NULL) {
  nvforest_validate_model(object)
  x <- nvforest_predictors(object, new_data)
  predictions <- nvforest_native_predict(
    model = object$xptr,
    input = x,
    prediction_type = 2L,
    threshold = 0.5,
    chunk_size = nvforest_chunk_size(
      nvforest_native_model_info(object$xptr),
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
#'
#' @section Memory use:
#' The complete result is materialized in R: rows by trees for scalar-leaf
#' models and rows by trees by outputs for vector-leaf models. The
#' \code{chunk_size} argument controls native prediction work but does not bound
#' the memory required by the R result.
#'
#' @seealso \code{\link{cuda_ml_nvforest_leaf_ids}()}
#' @export
cuda_ml_nvforest_predict_per_tree <- function(
  object,
  new_data,
  chunk_size = NULL
) {
  nvforest_validate_model(object)
  x <- nvforest_predictors(object, new_data)
  predictions <- nvforest_native_predict(
    model = object$xptr,
    input = x,
    prediction_type = 3L,
    threshold = 0.5,
    chunk_size = nvforest_chunk_size(
      nvforest_native_model_info(object$xptr),
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
    model = nvforest_native_serialize(model$xptr),
    class_levels = model$class_levels,
    precision = nvforest_code_name(
      model$inference$precision,
      c("-1" = "native", "0" = "single", "1" = "double"),
      "precision"
    ),
    averaged_vector_leaf_probabilities = inherits(
      model,
      "cuda_ml_rand_forest"
    ) &&
      identical(model$mode, "classification"),
    blueprint = model$blueprint
  )
}

nvforest_export_paths <- function(directory, prefix) {
  stopifnot(
    "`directory` must name one existing directory" = is.character(
      directory
    ) &&
      length(directory) == 1L &&
      !is.na(directory) &&
      dir.exists(directory),
    "`prefix` must be one non-empty basename" = is.character(prefix) &&
      length(prefix) == 1L &&
      !is.na(prefix) &&
      nzchar(prefix) &&
      !prefix %in% c(".", "..") &&
      !grepl("[/\\\\]", prefix)
  )

  directory <- normalizePath(directory, mustWork = TRUE)
  c(
    checkpoint = file.path(
      directory,
      paste0(prefix, ".treelite.checkpoint")
    ),
    metadata = file.path(directory, paste0(prefix, ".cuda-ml.json"))
  )
}

nvforest_json_scalar <- function(x) {
  jsonlite::unbox(x)
}

nvforest_export_model_info <- function(object) {
  info <- cuda_ml_nvforest_info(object)
  fields <- c(
    "task_type",
    "num_classes",
    "num_features",
    "num_outputs",
    "num_trees",
    "has_vector_leaves",
    "average_tree_output",
    "has_probability_output",
    "treelite_postprocessor"
  )
  lapply(info[fields], nvforest_json_scalar)
}

nvforest_export_feature_names <- function(blueprint, num_features) {
  predictors <- blueprint$ptypes$predictors
  if (is.null(predictors)) {
    return(NULL)
  }

  processed <- hardhat::forge(predictors, blueprint)$predictors
  feature_names <- colnames(processed)
  if (is.null(feature_names) || length(feature_names) != num_features) {
    return(NULL)
  }
  feature_names
}

nvforest_export_manifest <- function(
  object,
  state,
  checkpoint_file,
  checkpoint_path
) {
  info <- cuda_ml_nvforest_info(object)
  list(
    format = nvforest_json_scalar("cuda_ml_nvforest_export"),
    schema = nvforest_json_scalar(1L),
    package_version = nvforest_json_scalar(state$package_version),
    backend = lapply(state$backend, nvforest_json_scalar),
    model_abi = nvforest_json_scalar(state$model_abi),
    checkpoint = list(
      file = nvforest_json_scalar(checkpoint_file),
      format = nvforest_json_scalar("treelite_checkpoint"),
      size = nvforest_json_scalar(unname(file.info(checkpoint_path)$size)),
      sha256 = nvforest_json_scalar(cuda_ml_hash_file(checkpoint_path))
    ),
    payload = list(
      class_levels = state$payload$class_levels,
      precision = nvforest_json_scalar(state$payload$precision),
      averaged_vector_leaf_probabilities = nvforest_json_scalar(
        state$payload$averaged_vector_leaf_probabilities
      ),
      blueprint = list(
        encoding = nvforest_json_scalar("r-serialize-v3-base64"),
        data = nvforest_json_scalar(
          jsonlite::base64_enc(
            serialize(state$payload$blueprint, NULL, version = 3L)
          )
        )
      )
    ),
    model = list(
      mode = nvforest_json_scalar(object$mode),
      feature_names = nvforest_export_feature_names(
        state$payload$blueprint,
        info$num_features
      ),
      info = nvforest_export_model_info(object)
    )
  )
}

#' Export and import an nvForest checkpoint pair
#'
#' \code{cuda_ml_nvforest_export()} writes a standard Treelite checkpoint and a
#' cuda.ml JSON sidecar. The checkpoint contains the device-neutral tree
#' ensemble. The sidecar retains cuda.ml metadata, class labels,
#' prediction precision, random-forest probability semantics, and the R
#' preprocessing blueprint needed for a complete cuda.ml round-trip.
#' \code{cuda_ml_nvforest_import()} restores the pair on a caller-selected
#' inference device.
#'
#' @param object An nvForest-backed model.
#' @param directory An existing output directory.
#' @param prefix A non-empty filename prefix without directory components.
#' @param overwrite Whether to replace both existing output files. The default
#'   is \code{FALSE}.
#'
#' @return \code{cuda_ml_nvforest_export()} invisibly returns a named character
#'   vector containing the absolute \code{checkpoint} and \code{metadata}
#'   paths. \code{cuda_ml_nvforest_import()} returns the restored
#'   nvForest-backed model.
#'
#' @section Files:
#' The function writes exactly \file{<prefix>.treelite.checkpoint} and
#' \file{<prefix>.cuda-ml.json}. The JSON records the checkpoint's relative
#' filename, size, and SHA-256 digest. It does not record inference device,
#' layout, chunk size, memory alignment, or GPU device identifier.
#'
#' Other Treelite consumers can load the checkpoint without the JSON. They must
#' supply numeric predictors in the recorded processed feature order when
#' feature names are available, or in the checkpoint's original positional
#' order otherwise. They must also implement any class-label and postprocessing
#' behavior described by the sidecar.
#'
#' Loading the bare checkpoint with
#' \code{cuda_ml_nvforest_load_model(model_type = "treelite_checkpoint")}
#' likewise omits the sidecar's preprocessing, original class labels, cuda.ml
#' model class, and random-forest probability semantics. Use
#' \code{cuda_ml_nvforest_import()} for an exact cuda.ml round-trip.
#'
#' @section Persistence choices:
#' Use \code{\link{cuda_ml_serialize}()} and
#' \code{\link{cuda_ml_unserialize}()} for one R-native state value. The
#' checkpoint pair is useful when the Treelite model must also be independently
#' available. A bundle is optional wrapping around the R-native state and is
#' not required for either workflow.
#'
#' cuda.ml validates the sidecar and selected backend before import. Prepare the
#' backend first with \code{\link{cuda_ml_install}()} for GPU operation or
#' \code{cuda_ml_install(device = "cpu")} for CPU-only inference. Import never
#' downloads a backend.
#'
#' @section Trust:
#' The JSON embeds an R-serialized hardhat blueprint so that formula and recipe
#' preprocessing round-trip. Import only artifacts from trusted sources, as
#' with \code{readRDS()} and \code{\link{cuda_ml_unserialize}()}. The recorded
#' SHA-256 digest checks integrity, not authenticity.
#'
#' @seealso \code{\link{cuda_ml_nvforest_load_model}()} and
#'   \code{\link{cuda_ml_serialize}()}
#' @export
cuda_ml_nvforest_export <- function(
  object,
  directory,
  prefix,
  overwrite = FALSE
) {
  nvforest_validate_model(object)
  stopifnot(
    "`overwrite` must be one non-missing logical value" = is.logical(
      overwrite
    ) &&
      length(overwrite) == 1L &&
      !is.na(overwrite)
  )
  paths <- nvforest_export_paths(directory, prefix)
  stopifnot(
    "nvForest export targets must not be directories" = !any(
      dir.exists(paths)
    )
  )
  if (!overwrite && any(file.exists(paths))) {
    stop(
      "The nvForest export files already exist: ",
      paste(basename(paths[file.exists(paths)]), collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  staging <- c(
    checkpoint = tempfile(
      paste0(".", prefix, "-checkpoint-"),
      tmpdir = directory
    ),
    metadata = tempfile(
      paste0(".", prefix, "-metadata-"),
      tmpdir = directory
    )
  )
  on.exit(unlink(staging), add = TRUE)

  state <- cuda_ml_get_state(object)
  writeBin(state$payload$model, staging[["checkpoint"]])
  manifest <- nvforest_export_manifest(
    object,
    state,
    basename(paths[["checkpoint"]]),
    staging[["checkpoint"]]
  )
  json <- jsonlite::toJSON(
    manifest,
    auto_unbox = FALSE,
    null = "null",
    digits = NA,
    pretty = TRUE
  )
  writeLines(enc2utf8(json), staging[["metadata"]], useBytes = TRUE)

  stopifnot(
    "Could not publish the nvForest checkpoint" = file.rename(
      staging[["checkpoint"]],
      paths[["checkpoint"]]
    ),
    "Could not publish the nvForest metadata" = file.rename(
      staging[["metadata"]],
      paths[["metadata"]]
    )
  )
  paths <- normalizePath(paths, mustWork = TRUE)
  names(paths) <- c("checkpoint", "metadata")
  invisible(paths)
}

nvforest_validate_export_manifest <- function(metadata, paths) {
  stopifnot(
    "The nvForest metadata format is unsupported" = is.list(metadata) &&
      identical(metadata$format, "cuda_ml_nvforest_export"),
    "The nvForest metadata schema is unsupported" = identical(
      metadata$schema,
      1L
    ),
    "The nvForest metadata model ABI is unsupported" = metadata$model_abi %in%
      c(
        "cuda_ml_nvforest_model_state",
        "cuda_ml_rand_forest_model_state"
      ),
    "The nvForest metadata checkpoint is invalid" = is.list(
      metadata$checkpoint
    ) &&
      identical(metadata$checkpoint$file, basename(paths[["checkpoint"]])) &&
      identical(metadata$checkpoint$format, "treelite_checkpoint") &&
      is.numeric(metadata$checkpoint$size) &&
      length(metadata$checkpoint$size) == 1L &&
      is.character(metadata$checkpoint$sha256) &&
      length(metadata$checkpoint$sha256) == 1L,
    "The nvForest metadata payload is invalid" = is.list(metadata$payload) &&
      is.character(metadata$payload$precision) &&
      length(metadata$payload$precision) == 1L &&
      is.logical(metadata$payload$averaged_vector_leaf_probabilities) &&
      length(metadata$payload$averaged_vector_leaf_probabilities) == 1L &&
      is.list(metadata$payload$blueprint) &&
      identical(
        metadata$payload$blueprint$encoding,
        "r-serialize-v3-base64"
      ) &&
      is.character(metadata$payload$blueprint$data) &&
      length(metadata$payload$blueprint$data) == 1L
  )

  size <- unname(file.info(paths[["checkpoint"]])$size)
  if (!identical(as.numeric(metadata$checkpoint$size), as.numeric(size))) {
    stop(
      "The nvForest checkpoint size does not match its metadata.",
      call. = FALSE
    )
  }
  if (
    !identical(
      metadata$checkpoint$sha256,
      cuda_ml_hash_file(paths[["checkpoint"]])
    )
  ) {
    stop(
      "The nvForest checkpoint SHA-256 does not match its metadata.",
      call. = FALSE
    )
  }
  invisible(metadata)
}

nvforest_read_checkpoint <- function(path) {
  connection <- file(path, open = "rb")
  on.exit(close(connection))
  readBin(connection, what = "raw", n = unname(file.info(path)$size))
}

nvforest_export_state <- function(metadata, checkpoint_path) {
  blueprint <- unserialize(
    jsonlite::base64_dec(metadata$payload$blueprint$data)
  )
  stopifnot(
    "The nvForest metadata blueprint is invalid" = is.list(blueprint)
  )
  structure(
    list(
      schema = metadata$schema,
      package_version = metadata$package_version,
      backend = metadata$backend,
      model_abi = metadata$model_abi,
      payload = list(
        model = nvforest_read_checkpoint(checkpoint_path),
        class_levels = metadata$payload$class_levels,
        precision = metadata$payload$precision,
        averaged_vector_leaf_probabilities = metadata$payload$averaged_vector_leaf_probabilities,
        blueprint = blueprint
      )
    ),
    class = c(metadata$model_abi, "cuda_ml_model_state")
  )
}

nvforest_validate_export_model <- function(object, metadata) {
  info <- cuda_ml_nvforest_info(object)
  expected <- metadata$model$info
  if (
    !identical(object$mode, metadata$model$mode) ||
      !identical(info[names(expected)], expected)
  ) {
    stop(
      "The imported nvForest model does not match its metadata.",
      call. = FALSE
    )
  }
  invisible(object)
}

#' @rdname cuda_ml_nvforest_export
#'
#' @param device Inference device: \code{"gpu"} or \code{"cpu"}. The default
#'   is \code{"gpu"}.
#' @param device_id GPU device identifier, or \code{NULL} for the current device.
#' @param layout Tree layout.
#' @param precision Native, single, or double precision. \code{NULL} retains the
#'   exported model's prediction precision.
#' @param default_chunk_size Default prediction chunk size, or \code{NULL} to
#'   use nvForest's heuristic.
#' @param align_bytes Memory alignment, or \code{NULL} for the device default.
#'
#' @export
cuda_ml_nvforest_import <- function(
  directory,
  prefix,
  device = c("gpu", "cpu"),
  device_id = NULL,
  layout = c("depth_first", "breadth_first", "layered"),
  precision = NULL,
  default_chunk_size = NULL,
  align_bytes = NULL
) {
  paths <- nvforest_export_paths(directory, prefix)
  if (!all(file.exists(paths)) || any(dir.exists(paths))) {
    stop(
      "The nvForest export files do not exist: ",
      paste(
        basename(paths[!file.exists(paths) | dir.exists(paths)]),
        collapse = ", "
      ),
      ".",
      call. = FALSE
    )
  }

  metadata <- jsonlite::read_json(
    paths[["metadata"]],
    simplifyVector = TRUE
  )
  nvforest_validate_export_manifest(metadata, paths)
  state <- nvforest_export_state(metadata, paths[["checkpoint"]])
  object <- cuda_ml_set_state_with_options(
    state,
    list(
      device = device,
      device_id = device_id,
      layout = layout,
      precision = precision,
      default_chunk_size = default_chunk_size,
      align_bytes = align_bytes
    )
  )
  nvforest_validate_export_model(object, metadata)
  object
}

nvforest_unserialize_payload <- function(payload, cls, inference) {
  xptr <- nvforest_native_unserialize(
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

nvforest_unserialize_payload_with_options <- function(
  payload,
  cls,
  device = c("gpu", "cpu"),
  device_id = NULL,
  layout = c("depth_first", "breadth_first", "layered"),
  precision = NULL,
  default_chunk_size = NULL,
  align_bytes = NULL
) {
  inference <- nvforest_inference_options(
    device = device %||% c("gpu", "cpu"),
    device_id = device_id,
    layout = layout %||% c("depth_first", "breadth_first", "layered"),
    precision = precision %||% payload$precision,
    default_chunk_size = default_chunk_size,
    align_bytes = align_bytes
  )
  nvforest_unserialize_payload(payload, cls, inference)
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
  cuda_ml_set_state_with_options.cuda_ml_nvforest_model_state(
    model_state,
    list()
  )
}

#' @export
cuda_ml_set_state_with_options.cuda_ml_nvforest_model_state <- function(
  model_state,
  options
) {
  payload <- cuda_ml_state_payload(
    model_state,
    "cuda_ml_nvforest_model_state"
  )
  do.call(
    nvforest_unserialize_payload_with_options,
    c(
      list(payload = payload, cls = "cuda_ml_nvforest"),
      options
    )
  )
}
