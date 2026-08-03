#' cuda.ml
#'
#' This package provides a R interface for the RAPIDS cuML library.
#'
#' @section Installation:
#' Install the portable R package from CRAN, then explicitly download its
#' locked native backend and runtime:
#' \preformatted{
#' install.packages("cuda.ml")
#' cuda.ml::cuda_ml_install()
#' }
#'
#' The CRAN package contains no compiled code. Prebuilt backends support Linux
#' x86_64 with glibc 2.28 or newer and are selected for the current R minor
#' version.
#'
#' Loading \pkg{cuda.ml} does not require a GPU, load native code, create a
#' cache, or contact the network. Call \code{\link{cuda_ml_install}()} to
#' download the exact backend from GitHub Releases and the pinned CUDA 13.2.2
#' and RAPIDS cuML and nvForest 26.06 runtime from upstream wheel hosts.
#' Treelite 4.7.0 is linked statically into the backend. Native operations fail
#' with an installation instruction until setup is complete. GPU operations
#' then require a supported NVIDIA GPU and driver 580 or newer; nvForest CPU
#' inference does not. Set \code{CUDA_ML_CACHE_DIR} to override the default
#' cache and \code{CUDA_ML_BACKEND_MIRROR} to use an exact backend mirror.
#'
#' To compile cuda.ml itself on the host without a prebuilt cuda.ml backend,
#' call \code{cuda_ml_install(source = TRUE)}. The default managed source build
#' downloads the locked CUDA, RAPIDS, CMake, Ninja, and Treelite build inputs.
#' It detects CUDA-visible GPU architectures when available and otherwise uses
#' the package's portable architecture list. Only Linux x86_64 with glibc 2.28
#' or newer and GNU C++ 14 or newer are required on the host. Use
#' \code{dependencies = "host"} with explicit
#' \code{CUDA_HOME}, \code{CUML_PREFIX}, \code{CUML_CUDA_ARCHITECTURES}, and
#' \code{CUDA_ML_CXX} inputs for a fully native, network-free source build.
#'
#' @author Yitao Li <yitao@rstudio.com>
#' @import Rcpp
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  .cuda_ml_state$metadata <- cuda_ml_backend_metadata(pkgname)
  symbols <- cuda_ml_native_symbols(pkgname)
  .cuda_ml_state$native_symbols <- symbols
  .cuda_ml_state$dll <- NULL

  register <- function(...) register_parsnip_models(pkgname)
  if (isNamespaceLoaded("parsnip")) {
    register()
  }
  setHook(packageEvent("parsnip", "onLoad"), register, action = "append")
}

register_parsnip_models <- function(pkgname) {
  register_rand_forest_model(pkgname)
  register_svm_model(pkgname)
  register_knn_model(pkgname)
  register_logistic_reg_models(pkgname)
  register_linear_reg_model(pkgname)
}
