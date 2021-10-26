#' cuda.ml
#'
#' This package provides a R interface for the RAPIDS cuML library.
#'
#' @docType package
#' @author Yitao Li <yitao@rstudio.com>
#' @import Rcpp
#' @name cuda.ml
#' @useDynLib cuda.ml, .registration = TRUE
NULL

.onLoad <- function(libname, pkgname) {
  register_rand_forest_model(pkgname)
  register_svm_model(pkgname)
  register_knn_model(pkgname)
}

.onAttach <- function(libname, pkgname) {
  if (!has_cuML()) {
    packageStartupMessage(
      "
      The current installation of {", pkgname, "} will not function as expected
      because it was not linked with a valid version of the RAPIDS cuML shared
      library.

      To fix this issue, please follow https://rapids.ai/start.html#get-rapids
      to install the RAPIDS cuML shared library from Conda and ensure the
      'CUML_PREFIX' env variable is set to a valid RAPIDS conda env directory
      (e.g., '/home/user/anaconda3/envs/rapids-21.06', '/usr', or similar)
      during the installation of {", pkgname, "} or alternatively, follow
      https://github.com/yitao-li/cuml-installation-notes#build-from-source-without-conda-and-without-multi-gpu-support
      or
      https://github.com/yitao-li/cuml-installation-notes#build-from-source-without-conda-and-with-multi-gpu-support
      or similar to build and install RAPIDS cuML library from source, and
      then re-install {", pkgname, "}.\n\n
      "
    )
  }
}
