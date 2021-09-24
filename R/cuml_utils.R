#' Determine whether {cuda.ml} was linked to a valid version of the RAPIDS cuML
#' shared library.
#'
#' @return A logical value indicating whether the current installation {cuda.ml}
#'   was linked to a valid version of the RAPIDS cuML shared library.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (!has_libcuml()) {
#'   warning(
#'     "Please install the RAPIDS cuML shared library first, and then re-",
#'     "install {cuda.ml}."
#'   )
#' }
#' @export
has_libcuml <- .has_libcuml
