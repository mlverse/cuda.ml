#' Determine whether {cuda.ml} was linked to a valid version of the RAPIDS cuML
#' shared library.
#'
#' @return A logical value indicating whether the current installation {cuda.ml}
#'   was linked to a valid version of the RAPIDS cuML shared library.
#'
#' @details
#' If this returns \code{FALSE}, \pkg{cuda.ml} was installed in stub-only mode.
#' On a GPU machine, verify that \code{nvidia-smi} and \code{nvcc --version}
#' both work, then reinstall \pkg{cuda.ml}. During installation, \pkg{cuda.ml}
#' can bootstrap RAPIDS cuML from pip wheels with \code{uv} or Python/pip. If
#' RAPIDS cuML is already installed, set \code{CUML_PREFIX} to a prefix
#' containing \code{include/cuml} and \code{lib/libcuml++.so} before
#' reinstalling.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (!has_cuML()) {
#'   warning(
#'     "This installation was built without RAPIDS cuML. Verify `nvidia-smi` ",
#'     "and `nvcc --version`, then reinstall {cuda.ml}."
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
