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
#' CUML_BOOTSTRAP: The default is to bootstrap RAPIDS cuML from pip wheels if
#'                 no existing `libcuml` is found and a suitable NVIDIA
#'                 GPU/driver, `nvcc`, and Python package installer are
#'                 available. Set CUML_BOOTSTRAP=0 to disable this behavior.
#'
#' CUML_BOOTSTRAP_CACHE: Override the cache directory used for bootstrapped
#'                       RAPIDS headers and shared libraries.
#'
#' CUML_CUDA_ARCHITECTURES: Override CMAKE_CUDA_ARCHITECTURES. Defaults to
#'                          detected GPU architectures supported by nvcc.
#'
#' CUML_RAPIDS_CMAKE_SOURCE_DIR: Override the local rapids-cmake checkout used
#'                               by CMake FetchContent.
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

  for (f in c("logging.R", "platform.R", "nvcc.R", "bootstrap.R", "cuml.R", "cmake.R")) {
    source(file.path(wd, f))
  }
}

load_libcuml_versions()
load_util_fns()

find_rapids_cmake_source_dir <- function(src_dir, build_dir) {
  candidates <- c(
    Sys.getenv("CUML_RAPIDS_CMAKE_SOURCE_DIR", unset = NA),
    file.path(src_dir, "_deps", "rapids-cmake-src"),
    file.path(build_dir, "_deps", "rapids-cmake-src")
  )
  candidates <- candidates[!is.na(candidates)]

  for (candidate in candidates) {
    if (file.exists(file.path(candidate, "rapids-cmake", "rapids-cuda.cmake"))) {
      return(normalizePath(candidate))
    }
  }

  NA_character_
}

run_cmake <- function() {
  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())
  nvcc <- find_nvcc()

  define(R_INCLUDE_DIR = R.home("include"))
  define(RCPP_INCLUDE_DIR = system.file("include", package = "Rcpp"))
  configure_file(file.path("src", "CMakeLists.txt.in"))

  cmake_bin <- find_or_download_cmake(
    min_version = cuda_ml_min_cmake_version,
    exdir = file.path(pkg_root(), "tools")
  )
  src_dir <- normalizePath(file.path(pkg_root(), "src"))
  build_dir <- file.path(src_dir, ".cmake-build")
  dir.create(build_dir, recursive = TRUE, showWarnings = FALSE)

  define(
    CMAKE_BIN = shQuote(cmake_bin),
    CMAKE_BUILD_DIR = shQuote(build_dir),
    CMAKE_BUILD_OUTPUT = shQuote(file.path(build_dir, "cuda.ml.so"))
  )
  configure_file(
    file.path("tools", "config", "Makefile.cmake.in"),
    target = file.path("src", "Makefile")
  )

  cuml_prefix <- get_cuml_prefix()
  bundle_libcuml <- FALSE
  if (is.na(cuml_prefix)) {
    cuml_prefix <- normalizePath(file.path(pkg_root(), "libcuml"), mustWork = FALSE)
    download_libcuml()
    dir.create("inst", showWarnings = FALSE)
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

  cuda_architectures <- Sys.getenv("CUML_CUDA_ARCHITECTURES", unset = NA)
  if (is.na(cuda_architectures)) {
    cuda_architectures <- infer_cuda_architectures(nvcc)
  }
  cmake_args <- c(
    "-S", src_dir,
    "-B", build_dir,
    paste0("-DCMAKE_CUDA_ARCHITECTURES=", cuda_architectures),
    paste0("-DCUML_INCLUDE_DIR=", file.path(cuml_prefix, "include")),
    paste0("-DCUML_LIB_DIR=", file.path(cuml_prefix, "lib")),
    paste0("-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=", build_dir),
    paste0(
      "-DCUML_STUB_HEADERS_DIR=", normalizePath(file.path(src_dir, "stubs"))
    ),
    paste0("-DCMAKE_CUDA_COMPILER=", nvcc$path),
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE"
  )
  rapids_cmake_source_dir <- find_rapids_cmake_source_dir(src_dir, build_dir)
  if (!is.na(rapids_cmake_source_dir)) {
    cmake_args <- c(
      cmake_args,
      paste0("-DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=", rapids_cmake_source_dir)
    )
  }
  if (bundle_libcuml) {
    cmake_args <- c(
      cmake_args,
      "-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=TRUE",
      "-DCMAKE_INSTALL_RPATH:STRING='$ORIGIN'"
    )
  } else if (!identical(Sys.getenv("CUML_SET_RPATH", unset = "1"), "0")) {
    cmake_args <- c(
      cmake_args,
      "-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=TRUE",
      paste0("-DCMAKE_INSTALL_RPATH:STRING=", file.path(cuml_prefix, "lib"))
    )
  }
  rc <- system2(cmake_bin, args = cmake_args)

  if (rc != 0) {
    stop("Failed to run 'cmake'!")
  }
}

nvcc <- find_nvcc(stop_if_missing = FALSE)
if (is.null(nvcc) && !cuml_cran_like()) {
  warn_missing_nvcc()
}

if (is.null(nvcc) || !has_libcuml(nvcc = nvcc)) {
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
  define(CMAKE_BUILD_PARALLEL_ARGS = paste("--parallel", n_jobs))

  run_cmake()
}
