#' Options:
#'
#' CUML_VERSION: Specifies the version of the `libcuml` library to be downloaded
#'               and installed. This version must be present in the
#'               `libcuml_versions` list below.
#'
#' CUML_URL: Override the URL to download `libcuml` from. If specified, then no
#'           checks for CUDA version will be performed.
#'
#' CUML_PREFIX: If you have a copy of `libcuml` installed already, you can
#'              specify this environment variable to link {cuda.ml} with an
#'              existing installation of `libcuml`.
#'              If a valid copy of `libcuml` is found in '/usr' or in
#'              "${CUML_PREFIX}/", then no pre-built copy of `libcuml` will be
#'              downloaded.
#'
#' DOWNLOAD_CUML: The default is to automatically download a pre-built copy of
#'                `libcuml` if no existing `libcuml` is specified with the
#'                'CUML_PREFIX' env variable. Set DOWNLOAD_CUML=0 to disable
#'                this default behavior.
#'
#' DISABLE_PARALLEL_BUILD: Parallel build using max($(nproc) - 1, 1) cores is
#'                         enabled by default but can be disabled by setting
#'                         this env variable.
#'
#' CUML4R_CMAKE_PARALLEL_LEVEL: If not set and parallel build is enabled, then
#'                              max($(nproc) - 1, 1) cores will be used by the
#'                              build process. If set, then the number of cores
#'                              specified will be used.

pkg_root <- function() {
  # devtools::load_all() might run the config script from the `src` directory.
  for (p in list(".", "..")) {
    if (file.exists(file.path(p, "DESCRIPTION"))) {
      return(normalizePath(p))
    }
  }

  # should never reach here
  pkg_root <- normalizePath(".")
  warning(
    "Unable to locate 'DESCRIPTION' file! Assuming pkg root is '",
    pkg_root, "'."
  )
  return(pkg_root)
}

load_libcuml_versions <- function() {
  wd <- file.path(pkg_root(), "tools", "config")

  source(file.path(wd, "libcuml_versions.R"))
}

load_util_fns <- function() {
  wd <- file.path(pkg_root(), "tools", "config", "utils")

  for (f in c("cuml.R", "cmake.R", "logging.R", "nvcc.R", "platform.R")) {
    source(file.path(wd, f))
  }
}

load_libcuml_versions()
load_util_fns()

run_cmake <- function() {
  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())

  define(R_INCLUDE_DIR = R.home("include"))
  define(RCPP_INCLUDE_DIR = system.file("include", package = "Rcpp"))
  configure_file(file.path("src", "CMakeLists.txt.in"))

  cuml_prefix <- get_cuml_prefix()
  bundle_libcuml <- FALSE
  if (is.na(cuml_prefix)) {
    cuml_prefix <- normalizePath(file.path(pkg_root(), "libcuml"))
    download_libcuml()
    dir.create("inst")
    file.rename(file.path("libcuml", "lib"), file.path("inst", "libs"))
    file.symlink(file.path("..", "inst", "libs"), file.path("libcuml", "lib"))
    libs <- c("libtreelite", "libtreelite_runtime", "libcuml++")
    bundle_libcuml <- TRUE
  }
  cmake_prefix_path <- paste0(
    c(Sys.getenv("CMAKE_PREFIX_PATH", unset = ""), cuml_prefix),
    collapse = ":"
  )
  Sys.setenv(CMAKE_PREFIX_PATH = cmake_prefix_path)

  setwd(file.path(pkg_root(), "src"))

  cmake_args <- c(
    ".",
    "-DCMAKE_CUDA_ARCHITECTURES=NATIVE",
    paste0("-DCUML_INCLUDE_DIR=", file.path(cuml_prefix, "include")),
    paste0("-DCUML_LIB_DIR=", file.path(cuml_prefix, "lib")),
    paste0(
      "-DCUML_STUB_HEADERS_DIR=", normalizePath(file.path(getwd(), "stubs"))
    ),
    paste0("-DCMAKE_CUDA_COMPILER=", find_nvcc()$path),
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE"
  )
  if (bundle_libcuml) {
    cmake_args <- c(
      cmake_args,
      "-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=TRUE",
      "-DCMAKE_INSTALL_RPATH:STRING='$ORIGIN'"
    )
  }
  cmake_bin <- find_or_download_cmake(
    min_version = cuda_ml_min_cmake_version,
    exdir = file.path(pkg_root(), "tools")
  )
  rc <- system2(cmake_bin, args = cmake_args)

  if (rc != 0) {
    stop("Failed to run 'cmake'!")
  }
}

if (is.null(find_nvcc(stop_if_missing = FALSE)) || !has_libcuml()) {
  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())
  define(STUBS_HEADERS_DIR = normalizePath(file.path(getwd(), "src", "stubs")))
  define(CUSTOMIZED_MAKEFLAGS = "")
} else {
  define(STUBS_HEADERS_DIR = "")
  n_jobs <- (
    if (!is.na(Sys.getenv("DISABLE_PARALLEL_BUILD", unset = NA))) {
      1L
    } else {
      user_specified_parallel_level <- Sys.getenv("CUML4R_CMAKE_PARALLEL_LEVEL", unset = NA)
      if (!is.na(user_specified_parallel_level)) {
        as.integer(user_specified_parallel_level)
      } else {
        max(nproc() - 1L, 1L)
      }
    })
  define(CUSTOMIZED_MAKEFLAGS = paste0("MAKEFLAGS += '-j", n_jobs, "'"))

  run_cmake()
}
