#' cuda.ml
#'
#' This package provides a R interface for the RAPIDS cuML library.
#'
#' @section Installation:
#' A functional GPU installation requires an NVIDIA GPU with a working driver,
#' a CUDA Toolkit installation that provides \code{nvcc}, and normal R package
#' build tools. During installation, \pkg{cuda.ml} first looks for RAPIDS cuML
#' 24.0 or newer through \code{CUML_PREFIX} or \code{CUDA_PATH}. If none is
#' found, it can bootstrap RAPIDS cuML from pip wheels with \code{uv} or
#' Python/pip and link against the resulting local prefix.
#'
#' Builds without a visible GPU can be cross-compiled by setting
#' \code{CUML_CUDA_ARCHITECTURES} explicitly. On machines without \code{nvcc},
#' or without both a usable NVIDIA driver/GPU and an explicit target
#' architecture, including CRAN check machines, \pkg{cuda.ml} may install in
#' stub-only mode. In that mode \code{has_cuML()} returns \code{FALSE}, and
#' cuML-backed algorithms are unavailable until the system prerequisites are
#' installed and \pkg{cuda.ml} is reinstalled.
#'
#' Useful environment variables include \code{CUDA_HOME}, \code{CUML_PREFIX},
#' \code{CUML_BOOTSTRAP}, \code{CUML_BOOTSTRAP_CACHE}, and
#' \code{CUML_CUDA_ARCHITECTURES}.
#'
#' @author Yitao Li <yitao@rstudio.com>
#' @import Rcpp
#' @useDynLib cuda.ml, .registration = TRUE
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  register_rand_forest_model(pkgname)
  register_svm_model(pkgname)
  register_knn_model(pkgname)
}

.onAttach <- function(libname, pkgname) {
  if (!has_cuML()) {
    packageStartupMessage(
      "
      The current installation of {", pkgname, "} was built without a usable
      RAPIDS cuML shared library.

      To fix this, ensure `nvidia-smi` and `nvcc --version` both work, then
      reinstall {", pkgname, "}. During installation, {", pkgname, "} can
      bootstrap RAPIDS cuML from pip wheels with `uv` or Python/pip.

      If RAPIDS cuML 24.0 or newer is already installed, set `CUML_PREFIX` to a
      prefix containing include/cuml and lib/libcuml.so before reinstalling.\n\n
      "
    )
  }
}
