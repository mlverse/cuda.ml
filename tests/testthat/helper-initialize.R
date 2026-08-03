backend_info <- cuda_ml_backend_info()
native_platform_supported <- cuda_ml_supported_platform()
nvidia_smi <- unname(Sys.which("nvidia-smi"))
gpu_output <- character()
if (nzchar(nvidia_smi)) {
  gpu_output <- suppressWarnings(tryCatch(
    system2(
      nvidia_smi,
      c("--query-gpu=compute_cap", "--format=csv,noheader"),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(e) character()
  ))
}
gpu_status <- attr(gpu_output, "status", exact = TRUE)
visible_devices <- Sys.getenv("CUDA_VISIBLE_DEVICES", unset = NA_character_)
gpu_visible <- (
  is.na(visible_devices) ||
    (
      nzchar(trimws(visible_devices)) &&
        !grepl("^-[0-9]+", trimws(visible_devices))
    )
) &&
  length(gpu_output) > 0L &&
  (is.null(gpu_status) || gpu_status == 0L) &&
  all(grepl("^[0-9]+[.][0-9]+$", trimws(gpu_output)))
run_gpu_tests <- !identical(Sys.getenv("CUDA_ML_GPU_TESTS"), "false") &&
  gpu_visible &&
  backend_info$backend_available &&
  backend_info$runtime_installed

if (run_gpu_tests) {
  library(magrittr, warn.conflicts = FALSE)
  library(reticulate)
  library(rlang, warn.conflicts = FALSE)
}

if (run_gpu_tests) {
  reticulate::py_require("scikit-learn")
  sklearn <- reticulate::import("sklearn")
  sklearn_iris_dataset <- list(
    data = iris[, names(iris) != "Species"] %>%
      unname() %>%
      as.matrix(),
    target = as.integer(iris[["Species"]])
  )
  sklearn_mtcars_dataset <- list(
    data = mtcars[, names(mtcars) != "mpg"] %>%
      data.frame(row.names = NULL) %>%
      unname() %>%
      as.matrix(),
    target = mtcars[["mpg"]]
  )
}

#' Sort matrix rows by all columns or by a subset of columns.
#'
#' @param cols Indices of columns used for sorting.
sort_mat <- function(m, cols = seq(ncol(m))) {
  m[do.call(order, lapply(cols, function(x) m[, x])), ]
}

#' Attempt to unserialize a CuML model within a sub-process and use the
#' unserialized model to make predictions.
predict_in_sub_proc <- function(
  model_state,
  data,
  expected_mode,
  expected_model_cls = NULL,
  additional_predict_args = list()
) {
  impl <- function(
    model_state,
    data,
    expected_mode,
    expected_model_cls,
    additional_predict_args
  ) {
    suppressPackageStartupMessages(library(cuda.ml))

    model <- cuda_ml_unserialize(model_state)
    for (cls in expected_model_cls) {
      testthat::expect_s3_class(model, cls)
    }
    stopifnot(identical(model$mode, expected_mode))

    do.call(predict, append(list(model, data), additional_predict_args))
  }

  callr::r(
    impl,
    args = list(
      model_state = model_state,
      data = data,
      expected_mode = expected_mode,
      expected_model_cls = expected_model_cls,
      additional_predict_args = additional_predict_args
    ),
    stdout = "",
    stderr = ""
  )
}

predict_saved_models_in_sub_proc <- function(
  model,
  data,
  additional_predict_args = list()
) {
  state_path <- tempfile(fileext = ".cuda-ml-state")
  bundle_path <- tempfile(fileext = ".rds")
  on.exit(unlink(c(state_path, bundle_path)))

  connection <- file(state_path, open = "wb")
  cuda_ml_serialize(model, connection)
  close(connection)
  saveRDS(bundle::bundle(model), bundle_path)

  callr::r(
    function(state_path, bundle_path, data, additional_predict_args) {
      suppressPackageStartupMessages(library(cuda.ml))

      connection <- file(state_path, open = "rb")
      restored <- cuda_ml_unserialize(connection)
      close(connection)
      unbundled <- bundle::unbundle(readRDS(bundle_path))

      list(
        restored = do.call(
          predict,
          c(list(restored, data), additional_predict_args)
        ),
        unbundled = do.call(
          predict,
          c(list(unbundled, data), additional_predict_args)
        )
      )
    },
    args = list(
      state_path = state_path,
      bundle_path = bundle_path,
      data = data,
      additional_predict_args = additional_predict_args
    ),
    stdout = "",
    stderr = ""
  )
}

gen_blobs <- function(blob_sz = 10, centers = NULL) {
  centers <- centers %||% list(c(1000, 1000), c(-1000, -1000), c(-1000, 1000))
  pts <- centers %>%
    purrr::map(~ MASS::mvrnorm(blob_sz, mu = .x, Sigma = diag(length(.))))

  rlang::exec(rbind, !!!pts)
}

verify_iris_embedding <- function(embedding) {
  set.seed(0L)
  k_clust <- kmeans(embedding, centers = embedding[c(1, 51, 101), ])

  # i.e., one should be able to obtain a reasonably good clustering result
  # (as measured by the BSS/TSS ratio) within very few k-means iterations on the
  # embedding.
  expect_lte(k_clust$iter, 3)
  expect_gte(k_clust$betweenss / k_clust$totss, 0.95)

  # Use `iris$Species` to check pairs of data points from the same species are
  # mostly in the same cluster, and those from different species are mostly in
  # different clusters in the resulting clustering.
  expect_gte(
    sklearn$metrics$adjusted_rand_score(
      labels_true = iris$Species,
      labels_pred = k_clust$cluster
    ),
    0.7
  )
}
