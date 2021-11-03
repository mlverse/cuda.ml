#' Determine whether Forest Inference Library (FIL) functionalities are enabled
#' in the current installation of {cuda.ml}.
#'
#' CuML Forest Inference Library (FIL) functionalities (see
#' https://github.com/rapidsai/cuml/tree/main/python/cuml/fil#readme) will
#' require Treelite C API. If you need FIL to run tree-based model ensemble on
#' GPU, and \code{fil_enabled()} returns FALSE, then please consider installing
#' Treelite and then re-installing {cuda.ml}.
#'
#' @return A logical value indicating whether the Forest Inference Library (FIL)
#'   functionalities are enabled.
#'
#' @examples
#' if (cuda_ml_fil_enabled()) {
#'   # run GPU-accelerated Forest Inference Library (FIL) functionalities
#' } else {
#'   message(
#'     "FIL functionalities are disabled in the current installation of ",
#'     "{cuda.ml}. Please reinstall Treelite C library first, and then re-install",
#'     " {cuda.ml} to enable FIL."
#'   )
#' }
#' @export
cuda_ml_fil_enabled <- .fil_enabled

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
#'   {"classification", "regression"}.
#' @param model_type Format of the saved model file. Notice if \code{filename}
#'   ends with ".json" and \code{model_type} is "xgboost", then {cuda.ml} will
#'   assume the model file is in XGBoost JSON (instead of binary) format.
#'   Default: "xgboost".
#' @param algo Type of the algorithm for inference, must be one of the
#'   following.
#'     - "auto":
#'         Choose the algorithm automatically. Currently 'batch_tree_reorg' is
#'         used for dense storage, and 'naive' for sparse storage.
#'     - "naive":
#'         Simple inference using shared memory.
#'     - "tree_reorg":
#'         Similar to naive but with trees rearranged to be more coalescing-
#'         friendly.
#'     - "batch_tree_reorg":
#'         Similar to 'tree_reorg' but predicting multiple rows per thread
#'         block.
#'   Default: "auto".
#' @param threshold Class probability threshold for classification. Ignored for
#'   regression tasks. Default: 0.5.
#' @param storage_type In-memory storage format of the FIL model. Must be one of
#'   the following.
#'   - "auto":
#'       Choose the storage type automatically,
#'   - "dense":
#'       Create a dense forest,
#'   - "sparse":
#'       Create a sparse forest. Requires \code{algo} to be 'naive' or 'auto'.
#' @param threads_per_tree If >1, then have multiple (neighboring) threads infer
#'   on the same tree within a block, which will improve memory bandwith near
#'   tree root (but consuming more shared memory). Default: 1L.
#' @param n_items Number of input samples each thread processes. If 0, then
#'   choose (up to 4) that fit into shared memory. Default: 0L.
#' @param blocks_per_sm Indicates how CuML should determine the number of thread
#'   blocks to lauch for the inference kernel.
#'   - 0:
#'     Launches the number of blocks proportional to the number of data points.
#'   - >= 1:
#'     Attempts to lauch \code{blocks_per_sm} blocks for each streaming
#'     multiprocessor.
#'     This will fail if \code{blocks_per_sm} blocks result in more threads than
#'     the maximum supported number of threads per GPU. Even if successful, it
#'     is not guaranteed that \code{blocks_per_sm} blocks will run on an SM
#'     concurrently.
#'
#' @return A GPU-accelerated FIL model that can be used with the 'predict' S3
#'   generic to make predictions on new data points.
#'
#' @examples
#'
#' library(cuda.ml)
#' library(xgboost)
#'
#' model_path <- file.path(tempdir(), "xgboost.model")
#'
#' model <- xgboost(
#'   data = as.matrix(mtcars[names(mtcars) != "mpg"]),
#'   label = as.matrix(mtcars["mpg"]),
#'   max.depth = 6,
#'   eta = 1,
#'   nthread = 2,
#'   nrounds = 20,
#'   objective = "reg:squarederror"
#' )
#'
#' xgb.save(model, model_path)
#'
#' model <- cuda_ml_fil_load_model(
#'   model_path,
#'   mode = "regression",
#'   model_type = "xgboost"
#' )
#'
#' preds <- predict(model, mtcars[names(mtcars) != "mpg"])
#'
#' print(preds)
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

  xptr <- .fil_load_model(
    model_type = model_type,
    filename = filename,
    algo = algo,
    classification = identical(mode, "classification"),
    threshold = as.numeric(threshold),
    storage_type = storage_type,
    threads_per_tree = as.integer(threads_per_tree),
    n_items = as.integer(n_items),
    blocks_per_sm = as.integer(blocks_per_sm)
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
