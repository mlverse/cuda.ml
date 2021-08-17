#' Determine whether 'cuml4r' was linked to a valid version of the 'cuml' shared
#' library.
#'
#' Return TRUE if the current installation 'cuml4r' was linked to a valid
#' version of the 'cuml' shared library, otherwise FALSE.
#'
#' @examples
#'
#' library(cuml4r)
#'
#' if (!cuml4r_has_cuml()) {
#'   warning(
#'     "`cuML` is missing, and `cuml4r` will not work as expected! ",
#'     "Please install `cuML` first and then re-install `cuml4r`."
#'   )
#' }
#' @export
cuml4r_has_cuml <- .has_cuml
