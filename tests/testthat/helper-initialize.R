library(magrittr, warn.conflicts = FALSE)
library(reticulate)
library(rlang, warn.conflicts = FALSE)

expect_libcuml <- function() {
  if (!has_libcuml()) {
    stop(
      "The current installation of {cuda.ml} is not linked with a valid copy of",
      " libcuml!\n",
      ".libPaths:\n",
      paste(.libPaths(), collapse = "\n")
    )
  }
}

expect_libcuml()

sklearn <- tryCatch(reticulate::import("sklearn"),
  error = function(e) {
    reticulate::py_install("sklearn", pip = TRUE)
    reticulate::import("sklearn")
  }
)
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

#' Sort matrix rows by all columns or by a subset of columns.
#'
#' @param cols Indices of columns used for sorting.
sort_mat <- function(m, cols = seq(ncol(m))) {
  m[do.call(order, lapply(cols, function(x) m[, x])), ]
}

#' Attempt to unserialize a CuML model within a sub-process and use the
#' unserialized model to make predictions.
predict_in_sub_proc <- function(model_state, data, expected_mode,
                                expected_model_cls = NULL,
                                additional_predict_args = list()) {
  impl <- function(expect_libcuda_ml_impl, model_state, data, expected_mode,
                   expected_model_cls, additional_predict_args) {
    library(cuda.ml)
    expect_libcuda_ml_impl()

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
      expect_libcuda_ml_impl = expect_libcuml,
      model_state = model_state,
      data = data,
      expected_mode = expected_mode,
      expected_model_cls = expected_model_cls,
      additional_predict_args = additional_predict_args
    ),
    stdout = "", stderr = ""
  )
}

gen_blobs <- function(blob_sz = 10, centers = NULL) {
  centers <- centers %||% list(c(1000, 1000), c(-1000, -1000), c(-1000, 1000))
  pts <- centers %>%
    purrr::map(~ MASS::mvrnorm(blob_sz, mu = .x, Sigma = diag(2)))

  rlang::exec(rbind, !!!pts)
}
