#' Options:
#'
#' CUML_PREFIX: Required for a functional local source build. Set this to the
#'              explicit prefix containing the compatible RAPIDS cuML headers
#'              and libraries. Local builds never provision a toolchain.
#'
#' CUML_BOOTSTRAP_CACHE: Override the temporary build-toolchain cache used by
#'                       the managed build.
#'
#' CUML_CUDA_ARCHITECTURES: Required for a local build. Set this to the explicit
#'                          CMAKE_CUDA_ARCHITECTURES list. Managed builds use the
#'                          package's fixed portable architecture list.
#'
#' CUDA_ML_BUILD_MODE: Select "managed", "local", or "stub". Managed builds
#'                     provision the pinned CUDA 13.2.2 / RAPIDS 26.06
#'                     toolchain. Local builds require all inputs explicitly.
#'                     Stub builds contain no native backend.
#'
#' CUDA_ML_CXX: Required for local functional builds. Path to the GNU C++ 14 or
#'              newer compiler used for both C++ sources and nvcc host
#'              compilation. Managed builds use /usr/bin/g++.

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
    pkg_root,
    "'."
  )
  return(pkg_root)
}

load_util_fns <- function() {
  wd <- file.path(pkg_root(), "tools", "config", "utils")

  for (f in c(
    "logging.R",
    "platform.R",
    "nvcc.R",
    "artifacts.R",
    "bootstrap.R",
    "cuml.R",
    "cmake.R",
    "native-symbols.R"
  )) {
    source(file.path(wd, f))
  }
}

load_util_fns()

clear_build_artifacts <- function() {
  paths <- c(
    "Makevars",
    "Makefile",
    "_deps",
    ".cmake-build",
    "CMakeCache.txt",
    "CMakeFiles",
    "cmake_install.cmake",
    "CMakeLists.txt",
    "symbols.rds",
    "*.o",
    "*.so"
  )
  for (path in paths) {
    unlink(file.path(pkg_root(), "src", path), recursive = TRUE, expand = TRUE)
  }
}

clear_build_artifacts()
generate_cuda_ml_native_symbol_manifest()
cuml_generate_runtime_lock()

write_backend_metadata <- function(
  backend,
  build_mode,
  cuda = "",
  rapids = "",
  nvforest = "",
  treelite = "",
  platform = "",
  minimum_driver = "",
  architectures = ""
) {
  stopifnot(
    backend %in% c("full", "stub"),
    build_mode %in% c("managed", "local", "stub")
  )

  path <- file.path(pkg_root(), "inst", "cuda-ml-backend.dcf")
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(
    c(
      "Schema: 2",
      paste0("Backend: ", backend),
      paste0("Build-Mode: ", build_mode),
      paste0("CUDA: ", cuda),
      paste0("RAPIDS: ", rapids),
      paste0("nvForest: ", nvforest),
      paste0("Treelite: ", treelite),
      paste0("Platform: ", platform),
      paste0("Minimum-Driver: ", minimum_driver),
      paste0("Architectures: ", architectures)
    ),
    path
  )
}

run_cmake <- function(nvcc, cuml_prefix, cuda_architectures, cxx) {
  stopifnot(
    is.list(nvcc),
    is.character(cuml_prefix),
    length(cuml_prefix) == 1L,
    is.character(cuda_architectures),
    length(cuda_architectures) == 1L,
    is.character(cxx),
    length(cxx) == 1L,
    nzchar(cxx)
  )

  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())

  define(R_INCLUDE_DIR = R.home("include"))
  define(RCPP_INCLUDE_DIR = system.file("include", package = "Rcpp"))
  configure_file(file.path("src", "CMakeLists.txt.in"))

  cmake_bin <- find_cmake()
  src_dir <- normalizePath(file.path(pkg_root(), "src"))
  build_dir <- file.path(src_dir, ".cmake-build")
  dir.create(build_dir, recursive = TRUE, showWarnings = FALSE)

  define(
    CMAKE_BIN = shQuote(cmake_bin),
    CMAKE_BUILD_DIR = shQuote(build_dir),
    CMAKE_BUILD_OUTPUT = shQuote(file.path(build_dir, "cuda.ml.so")),
    RSCRIPT_BIN = shQuote(file.path(R.home("bin"), "Rscript"))
  )
  configure_file(
    file.path("tools", "config", "Makefile.cmake.in"),
    target = file.path("src", "Makefile")
  )

  stopifnot(!is.na(cuml_prefix), nzchar(cuml_prefix))
  cmake_prefix_path <- c(
    Sys.getenv("CMAKE_PREFIX_PATH", unset = ""),
    cuml_prefix
  )
  cmake_prefix_path <- paste(
    cmake_prefix_path[nzchar(cmake_prefix_path)],
    collapse = ":"
  )
  Sys.setenv(CMAKE_PREFIX_PATH = cmake_prefix_path)

  cmake_args <- c(
    "-S",
    src_dir,
    "-B",
    build_dir,
    paste0("-DCMAKE_CUDA_ARCHITECTURES=", cuda_architectures),
    paste0("-DCUML_INCLUDE_DIR=", file.path(cuml_prefix, "include")),
    paste0("-DCUML_LIB_DIR=", file.path(cuml_prefix, "lib")),
    paste0("-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=", build_dir),
    paste0("-DCMAKE_CUDA_COMPILER=", nvcc$path),
    paste0("-DCMAKE_CUDA_HOST_COMPILER=", cxx),
    paste0("-DCMAKE_CXX_COMPILER=", cxx),
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE"
  )
  cmake_args <- c(
    cmake_args,
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=TRUE",
    "-DCMAKE_INSTALL_RPATH:STRING=$ORIGIN",
    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=FALSE"
  )
  rc <- system2(cmake_bin, args = shQuote(cmake_args))

  if (rc != 0) {
    stop("Failed to run 'cmake'!")
  }
}

nvcc <- NULL
cuml_prefix <- NA_character_
cuda_architectures <- NA_character_
build_mode <- cuml_build_mode()

if (identical(build_mode, "managed")) {
  if (!cuml_ubuntu_2604_x86_64()) {
    stop2("Managed cuda.ml builds require Ubuntu 26.04 x86_64.")
  }
  managed_build <- bootstrap_managed_build_from_artifacts()
  nvcc <- managed_build$nvcc
  cuml_prefix <- managed_build$prefix
  cuda_architectures <- cuml_managed_cuda_architectures()
  cxx <- find_cuda_ml_cxx("/usr/bin/g++")
} else if (identical(build_mode, "local")) {
  if (!cuml_ubuntu_2604_x86_64()) {
    stop2("Functional local cuda.ml builds require Ubuntu 26.04 x86_64.")
  }
  cuda_home <- Sys.getenv("CUDA_HOME", unset = "")
  cuml_prefix <- Sys.getenv("CUML_PREFIX", unset = "")
  cuda_architectures <- Sys.getenv("CUML_CUDA_ARCHITECTURES", unset = "")
  cxx_path <- Sys.getenv("CUDA_ML_CXX", unset = "")
  if (
    !nzchar(cuda_home) ||
      !nzchar(cuml_prefix) ||
      !nzchar(cuda_architectures) ||
      !nzchar(cxx_path)
  ) {
    stop2(
      "A local functional build requires explicit build inputs.",
      paste0(
        "Set CUDA_HOME, CUML_PREFIX, CUML_CUDA_ARCHITECTURES, ",
        "and CUDA_ML_CXX."
      )
    )
  }
  cuda_home <- normalizePath(cuda_home, mustWork = TRUE)
  cuml_prefix <- normalizePath(cuml_prefix, mustWork = TRUE)
  nvcc_path <- file.path(cuda_home, "bin", "nvcc")
  nvcc_version <- nvcc_version_from_path(nvcc_path)
  if (is.null(nvcc_version)) {
    stop2("CUDA_HOME does not contain a working bin/nvcc.")
  }
  nvcc <- list(path = nvcc_path, version = nvcc_version)
  cxx <- find_cuda_ml_cxx(cxx_path)
  if (!check_functional_prefix(cuml_prefix)) {
    stop2(
      "CUML_PREFIX does not contain the exact cuML, nvForest, and Treelite prefix.",
      "Use CUDA 13.2.2, RAPIDS 26.06, nvForest 26.06.0, and Treelite 4.6.1."
    )
  }
}

full_build <- !identical(build_mode, "stub")

if (!full_build) {
  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())
  write_backend_metadata("stub", build_mode = build_mode)
} else {
  validate_managed_build_versions(nvcc, cuml_prefix)
  run_cmake(nvcc, cuml_prefix, cuda_architectures, cxx)
  write_backend_metadata(
    backend = "full",
    build_mode = build_mode,
    cuda = cuml_managed_cuda_toolkit_version(),
    rapids = cuml_managed_rapids_version(),
    nvforest = cuml_managed_nvforest_version(),
    treelite = cuml_managed_treelite_version(),
    platform = cuml_managed_platform(),
    minimum_driver = cuml_managed_minimum_driver(),
    architectures = cuda_architectures
  )
}
