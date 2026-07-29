#' Determine whether this \{cuda.ml\} build contains a cuML backend
#'
#' @return A logical value indicating whether the current \{cuda.ml\}
#'   installation contains a functional RAPIDS cuML backend.
#'
#' @details
#' This is a pure build-capability query. It does not inspect or modify the
#' managed runtime cache, contact the network, load native code, or check for
#' a GPU. A functional binary returns \code{TRUE} before
#' \code{\link{cuda_ml_install}()} has prepared its runtime.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (!has_cuML()) {
#'   warning(
#'     "Install the functional cuda.ml binary from the mlverse R-universe."
#'   )
#' }
#' @export
has_cuML <- function() {
  identical(.cuda_ml_state$metadata[["Backend"]], "full")
}

#' Get the major version of the RAPIDS cuML shared library \{cuda.ml\} was linked
#' to.
#'
#' @return The major version of the RAPIDS cuML shared library \{cuda.ml\} was
#' linked to in a character vector, or \code{NA_character_} if \{cuda.ml\} was not
#' linked to any version of RAPIDS cuML.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (interactive()) {
#'   print(cuML_major_version())
#' }
#' @export
cuML_major_version <- function() {
  if (!has_cuML()) {
    return(NA_character_)
  }
  .cuML_major_version()
}

#' Get the minor version of the RAPIDS cuML shared library \{cuda.ml\} was linked
#' to.
#'
#' @return The minor version of the RAPIDS cuML shared library \{cuda.ml\} was
#' linked to in a character vector, or \code{NA_character_} if \{cuda.ml\} was not
#' linked to any version of RAPIDS cuML.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (interactive()) {
#'   print(cuML_minor_version())
#' }
#' @export
cuML_minor_version <- function() {
  if (!has_cuML()) {
    return(NA_character_)
  }
  .cuML_minor_version()
}
