#' cuda.ml
#'
#' This package provides an R interface for the RAPIDS cuML library.
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
#' install the complete backend used for GPU training and inference. It uses
#' the pinned CUDA 13.2.2 and RAPIDS cuML and nvForest 26.06 runtime and is
#' roughly 1.6 GiB. GPU operations then require a supported NVIDIA GPU and
#' driver 580 or newer.
#'
#' For CPU-only nvForest inference, call
#' \code{cuda_ml_install(device = "cpu")} instead. This installs a separate
#' backend that is roughly 1 MiB to download and 3 MiB when installed. It
#' does not install cuML or the complete managed CUDA and RAPIDS runtime and
#' requires neither an NVIDIA GPU nor an NVIDIA driver. It contains no CUDA
#' runtime libraries. Treelite 4.7.0 is linked statically into both backends.
#'
#' Random forests trained on a GPU by \code{\link{cuda_ml_rand_forest}()} can
#' be persisted with \code{\link{cuda_ml_serialize}()} and restored for CPU
#' inference with \code{cuda_ml_unserialize(state, device = "cpu")}. Current
#' nvForest model states do not encode their deployment device. Alternatively,
#' \code{\link{cuda_ml_nvforest_export}()} writes a standard Treelite checkpoint
#' and cuda.ml JSON metadata that can be restored with
#' \code{\link{cuda_ml_nvforest_import}()}. Native operations fail with an
#' installation instruction until their corresponding backend is installed. Set
#' \code{CUDA_ML_CACHE_DIR} to override the default cache and
#' \code{CUDA_ML_BACKEND_MIRROR} to use an exact backend mirror.
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
  .cuda_ml_state$nvforest_cpu_dll <- NULL
  .cuda_ml_state$nvforest_cpu_symbols <- NULL

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
