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
#' if (!has_cuML()) {
#'   warning(
#'     "Please install the RAPIDS cuML shared library first, and then re-",
#'     "install {cuda.ml}."
#'   )
#' }
#' @export
has_cuML <- .has_cuML

#' Get the major version of the RAPIDS cuML shared library {cuda.ml} was linked
#' to.
#'
#' @return The major version of the RAPIDS cuML shared library {cuda.ml} was
#' linked to in a character vector, or \code{NA_character_} if {cuda.ml} was not
#' linked to any version of RAPIDS cuML.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' print(cuML_major_version())
#' @export
cuML_major_version <- .cuML_major_version

#' Get the minor version of the RAPIDS cuML shared library {cuda.ml} was linked
#' to.
#'
#' @return The minor version of the RAPIDS cuML shared library {cuda.ml} was
#' linked to in a character vector, or \code{NA_character_} if {cuda.ml} was not
#' linked to any version of RAPIDS cuML.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' print(cuML_minor_version())
#' @export
cuML_minor_version <- .cuML_minor_version
