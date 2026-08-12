backend_info <- cuda_ml_backend_info()
native_platform_supported <- cuda_ml_supported_platform()
penguin_predictors <- c(
  "bill_length_mm",
  "bill_depth_mm",
  "flipper_length_mm",
  "body_mass_g"
)
penguins <- palmerpenguins::penguins[c(penguin_predictors, "species")]
penguins <- penguins[complete.cases(penguins), , drop = FALSE]
scaled_penguin_predictors <- as.data.frame(scale(penguins[penguin_predictors]))
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
visible_gpu_count <- if (is.na(visible_devices)) {
  length(gpu_output)
} else if (gpu_visible) {
  sum(nzchar(trimws(strsplit(visible_devices, ",", fixed = TRUE)[[1L]])))
} else {
  0L
}
run_gpu_tests <- !identical(Sys.getenv("CUDA_ML_GPU_TESTS"), "false") &&
  gpu_visible &&
  backend_info$backend_available &&
  backend_info$runtime_installed
run_multi_gpu_tests <- run_gpu_tests && visible_gpu_count >= 2L

if (run_gpu_tests) {
  library(reticulate)
  library(rlang, warn.conflicts = FALSE)
}

if (run_gpu_tests) {
  reticulate::py_require("scikit-learn")
  sklearn <- reticulate::import("sklearn")
  sklearn_penguins_dataset <- list(
    data = scaled_penguin_predictors |>
      unname() |>
      as.matrix(),
    target = as.integer(penguins[["species"]])
  )
  sklearn_mtcars_dataset <- list(
    data = mtcars[, names(mtcars) != "mpg"] |>
      data.frame(row.names = NULL) |>
      unname() |>
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
  pts <- centers |>
    purrr::map(\(center) {
      MASS::mvrnorm(blob_sz, mu = center, Sigma = diag(length(center)))
    })

  do.call(rbind, pts)
}

verify_penguins_embedding <- function(embedding) {
  set.seed(0L)
  initial_rows <- match(levels(penguins$species), penguins$species)
  k_clust <- kmeans(embedding, centers = embedding[initial_rows, ])

  # i.e., one should be able to obtain a reasonably good clustering result
  # (as measured by the BSS/TSS ratio) within very few k-means iterations on the
  # embedding.
  expect_lte(k_clust$iter, 3)
  expect_gte(k_clust$betweenss / k_clust$totss, 0.75)

  # Use the species labels to check that pairs from the same species are mostly
  # in the same cluster, and pairs from different species are mostly separate.
  expect_gte(
    sklearn$metrics$adjusted_rand_score(
      labels_true = penguins$species,
      labels_pred = k_clust$cluster
    ),
    0.35
  )
}
