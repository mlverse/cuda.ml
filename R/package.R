#' cuda.ml
#'
#' This package provides a R interface for the RAPIDS cuML library.
#'
#' @section Installation:
#' Install the functional Ubuntu 26.04 (Resolute) x86_64 binary from the
#' mlverse \href{https://docs.r-universe.dev/install/binaries.html}{R-universe
#' Linux binary repository}:
#' \preformatted{
#' linux_binary_repo <- function(universe) {
#'   r_version <- paste(
#'     R.version$major,
#'     strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][1L],
#'     sep = "."
#'   )
#'   paste0(
#'     "https://", universe,
#'     ".r-universe.dev/bin/linux/resolute-",
#'     R.version$arch,
#'     "/",
#'     r_version,
#'     "/"
#'   )
#' }
#'
#' repos <- c(
#'   mlverse = linux_binary_repo("mlverse"),
#'   CRAN = linux_binary_repo("cran")
#' )
#' stopifnot(
#'   identical(unname(Sys.info()[["sysname"]]), "Linux"),
#'   identical(R.version$arch, "x86_64"),
#'   grepl(
#'     "/bin/linux/resolute-x86_64/[0-9]+[.][0-9]+/$",
#'     repos[["mlverse"]]
#'   )
#' )
#'
#' install.packages(
#'   "cuda.ml",
#'   repos = repos
#' )
#' }
#'
#' The binary repository path selects the prebuilt tarball. Stock Linux R does
#' not support \code{type = "binary"}, so leave \code{type} at its default.
#' This binary also supports WSL2 running Ubuntu 26.04. Other Linux
#' distributions are not yet supported by the managed binary.
#'
#' Loading \pkg{cuda.ml} does not require a GPU, load native code, create a
#' cache, or contact the network. Call \code{\link{cuda_ml_install}()} to
#' download and cache the pinned CUDA 13.2.2, RAPIDS cuML and nvForest 26.06,
#' and Treelite 4.6.1 runtime while preparing a container or machine image.
#' Native operations fail with an installation instruction until that explicit
#' setup step has completed. GPU operations then require a supported NVIDIA GPU
#' and driver 580 or newer; nvForest CPU inference does not.
#'
#' CRAN builds are network-free source stubs. A stub reports
#' \code{backend = "stub"} from \code{\link{cuda_ml_backend_info}()}; install
#' the R-universe binary for a no-compiler setup. Local functional source builds
#' require exact CUDA 13.2.2, GNU C++ 14 or newer, cuML and nvForest 26.06, and
#' Treelite 4.6.1 inputs. Set \code{CUDA_ML_CACHE_DIR} to override the default
#' managed runtime cache.
#'
#' @author Yitao Li <yitao@rstudio.com>
#' @import Rcpp
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  .cuda_ml_state$metadata <- cuda_ml_backend_metadata(pkgname)
  symbols <- cuda_ml_native_symbols(pkgname)
  .cuda_ml_state$native_symbols <- symbols
  .cuda_ml_state$dll <- NULL

  register_rand_forest_model(pkgname)
  register_svm_model(pkgname)
  register_knn_model(pkgname)
  register_logistic_reg_models(pkgname)
  register_linear_reg_model(pkgname)
}
