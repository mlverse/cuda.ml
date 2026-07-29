#' Determine whether Forest Inference Library (FIL) functionalities are enabled
#' in the current installation of \{cuda.ml\}.
#'
#' CuML Forest Inference Library (FIL) functionalities (see
#' https://github.com/rapidsai/cuml/tree/main/python/cuml/fil#readme) will
#' require the Treelite C API and a compatible cuML FIL API. If you need FIL to
#' run tree-based model ensembles on GPU, and \code{fil_enabled()} returns
#' FALSE, install compatible Treelite and cuML versions, then re-install
#' \{cuda.ml\}.
#'
#' @return A logical value indicating whether the Forest Inference Library (FIL)
#'   functionalities are enabled.
#'
#' @examples
#' if (interactive()) {
#'   if (cuda_ml_fil_enabled()) {
#'     # run GPU-accelerated Forest Inference Library (FIL) functionalities
#'   } else {
#'     message(
#'       "FIL functionalities are disabled in the current installation of ",
#'       "{cuda.ml}. Install compatible Treelite and cuML versions, then ",
#'       "re-install {cuda.ml} to enable FIL."
#'     )
#'   }
#' }
#' @export
cuda_ml_fil_enabled <- function() {
  if (!has_cuML()) {
    return(FALSE)
  }
  .fil_enabled()
}

fil_match_model_type <- function(filename, model_type = c("xgboost", "lightgbm")) {
  model_type <- match.arg(model_type)

  switch(model_type,
    xgboost = ifelse(grepl("\\.json$", filename), 1L, 0L),
    lightgbm = 2L
  )
}

fil_match_algo <- function(algo = c("auto", "naive", "tree_reorg", "batch_tree_reorg")) {
  algo <- match.arg(algo)

  switch(algo,
    auto = 0L,
    naive = 1L,
    tree_reorg = 2L,
    batch_tree_reorg = 3L
  )
}

file_match_storage_type <- function(storage_type = c("auto", "dense", "sparse")) {
  storage_type <- match.arg(storage_type)

  switch(storage_type,
    auto = 0L,
    dense = 1L,
    sparse = 2L
  )
}

#' Load a XGBoost or LightGBM model file.
#'
#' Load a XGBoost or LightGBM model file using Treelite. The resulting model
#' object can be used to perform high-throughput batch inference on new data
#' points using the GPU acceleration functionality from the CuML Forest
#' Inference Library (FIL).
#'
#' @param filename Path to the saved model file.
#' @param mode Type of task to be performed by the model. Must be one of
#'   \{"classification", "regression"\}.
#' @param model_type Format of the saved model file. Notice if \code{filename}
#'   ends with ".json" and \code{model_type} is "xgboost", then \{cuda.ml\} will
#'   assume the model file is in XGBoost JSON (instead of binary) format.
#'   Default: "xgboost".
#' @param algo Inference algorithm. The current cuML FIL API supports only
#'   \code{"auto"}.
#' @param threshold Class probability threshold for classification. Ignored for
#'   regression tasks. Default: 0.5.
#' @param storage_type In-memory storage format. The current cuML FIL API
#'   supports only \code{"auto"}.
#' @param threads_per_tree Number of threads per tree. The current cuML FIL API
#'   supports only \code{1L}.
#' @param n_items Number of input samples each thread processes. The current
#'   cuML FIL API supports only \code{0L}.
#' @param blocks_per_sm Number of thread blocks per streaming multiprocessor.
#'   The current cuML FIL API supports only \code{0L}.
#'
#' @return A GPU-accelerated FIL model that can be used with the 'predict' S3
#'   generic to make predictions on new data points.
#'
#' @details
#' The current cuML FIL API supports only the default loading controls:
#' \code{algo = "auto"}, \code{storage_type = "auto"},
#' \code{threads_per_tree = 1L}, \code{n_items = 0L}, and
#' \code{blocks_per_sm = 0L}. Other values produce an error.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (
#'   interactive() &&
#'     requireNamespace("xgboost", quietly = TRUE) &&
#'     cuda_ml_fil_enabled()
#' ) {
#'   model_path <- file.path(tempdir(), "xgboost.model")
#'
#'   model <- xgboost::xgboost(
#'     data = as.matrix(mtcars[names(mtcars) != "mpg"]),
#'     label = as.matrix(mtcars["mpg"]),
#'     max.depth = 6,
#'     eta = 1,
#'     nthread = 2,
#'     nrounds = 20,
#'     objective = "reg:squarederror"
#'   )
#'
#'   xgboost::xgb.save(model, model_path)
#'
#'   model <- cuda_ml_fil_load_model(
#'     model_path,
#'     mode = "regression",
#'     model_type = "xgboost"
#'   )
#'
#'   preds <- predict(model, mtcars[names(mtcars) != "mpg"])
#'
#'   print(preds)
#' }
#' @export
cuda_ml_fil_load_model <- function(filename,
                                   mode = c("classification", "regression"),
                                   model_type = c("xgboost", "lightgbm"),
                                   algo = c("auto", "naive", "tree_reorg", "batch_tree_reorg"),
                                   threshold = 0.5,
                                   storage_type = c("auto", "dense", "sparse"),
                                   threads_per_tree = 1L, n_items = 0L,
                                   blocks_per_sm = 0L) {
  mode <- match.arg(mode)
  model_type <- fil_match_model_type(filename, model_type)
  algo <- fil_match_algo(algo)
  storage_type <- file_match_storage_type(storage_type)
  threads_per_tree <- as.integer(threads_per_tree)
  n_items <- as.integer(n_items)
  blocks_per_sm <- as.integer(blocks_per_sm)

  stopifnot(
    "Only the default FIL loading controls are supported" =
      algo == 0L && storage_type == 0L && threads_per_tree == 1L &&
        n_items == 0L && blocks_per_sm == 0L
  )

  xptr <- .fil_load_model(
    model_type = model_type,
    filename = filename,
    algo = algo,
    classification = identical(mode, "classification"),
    threshold = as.numeric(threshold),
    storage_type = storage_type,
    threads_per_tree = threads_per_tree,
    n_items = n_items,
    blocks_per_sm = blocks_per_sm
  )
  model <- list(mode = mode, xptr = xptr)
  class(model) <- c("cuda_ml_fil", "cuda_ml_model", class(model))

  model
}

#' Make predictions on new data points.
#'
#' Make predictions on new data points using a FIL model.
#'
#' @template predict
#' @template output-class-probabilities
#'
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_fil <- function(object, x, output_class_probabilities = FALSE, ...) {
  check_dots_used()

  num_classes <- .fil_get_num_classes(model = object$xptr)
  preds <- .fil_predict(
    model = object$xptr,
    x = as.matrix(x),
    output_class_probabilities = output_class_probabilities
  )

  switch(object$mode,
    classification = {
      if (output_class_probabilities) {
        preds <- hardhat::spruce_prob(
          paste0("class_prob_", seq(num_classes) - 1L), preds
        )
      } else {
        preds <- factor(preds)
        preds <- hardhat::spruce_class(preds)
      }
    },
    regression = {
      preds <- hardhat::spruce_numeric(c(preds))
    }
  )

  preds
}
