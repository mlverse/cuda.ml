#' Determine whether {cuml} was linked to a valid version of the RAPIDS cuML
#' shared library.
#'
#' Return TRUE if the current installation {cuml} was linked to a valid
#' version of the RAPIDS cuML shared library, otherwise FALSE.
#'
#' @examples
#'
#' library(cuml)
#'
#' if (!has_libcuml()) {
#'   warning(
#'     "Please install the RAPIDS cuML shared library first, and then re-",
#'     "install {cuml}."
#'   )
#' }
#' @export
has_libcuml <- .has_libcuml
