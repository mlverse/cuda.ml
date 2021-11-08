#' @param max_iter The maximum number of coordinate descent iterations.
#'   Default: 1000L.
#' @param tol Stop the coordinate descent when the duality gap is below this
#'   threshold. Default: 1e-3.
#' @param selection If "random", then instead of updating coefficients in cyclic
#'   order, a random coefficient is updated in each iteration. Default: "cyclic".
