library(cuml)
library(magrittr)
library(reticulate)
library(rlang)

sklearn <- import("sklearn")
sklearn_iris_dataset <- sklearn$datasets$load_iris()

#' Sort matrix rows by all columns or by a subset of columns.
#'
#' @param cols Indices of columns used for sorting.
sort_mat <- function(m, cols = seq(ncol(m))) {
  m[do.call(order, lapply(cols, function(x) m[, x])), ]
}
