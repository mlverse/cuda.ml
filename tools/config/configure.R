get_cuml_prefix <- function() {
  cuml_prefix <- Sys.getenv("CUML_PREFIX", unset = NA_character_)
  if (is.na(cuml_prefix)) {
    # Try the 'CUDA_PATH' env variable if it is present.
    cuml_prefix <- Sys.getenv("CUDA_PATH", unset = NA_character_)
  }
  if (is.na(cuml_prefix)) {
    warning(
      "\t**********************************************\n",
      "\t**********************************************\n",
      "\t**                                          **\n",
      "\t** 'CUML_PREFIX' env variable is missing -- **\n",
      "\t** will boldly assume it is '/usr' !        **\n",
      "\t**                                          **\n",
      "\t**********************************************\n",
      "\t**********************************************\n",
      immediate. = TRUE
    )
    cuml_prefix <- "/usr"
  }

  cuml_prefix
}

has_cuml <- function() {
  cuml_prefix <- get_cuml_prefix()
  cuml_headers_dir <- file.path(cuml_prefix, "include", "cuml")

  if (!dir.exists(cuml_headers_dir)) {
    warning(
      "\t'", cuml_headers_dir, "' does not exist or is not a directory!\n\n",
      "\t***************************************************************************\n",
      "\t***************************************************************************\n",
      "\t**                                                                       **\n",
      "\t**  {cuda.ml} requires a valid RAPIDS installation.                      **\n",
      "\t**  Please follow https://rapids.ai/start.html to install RAPIDS first.  **\n",
      "\t**  {cuda.ml} must be installed from an environment containing a valid   **\n",
      "\t**  CUML_PREFIX env variable (e.g.,                                      **\n",
      "\t**  '/home/user/anaconda3/envs/rapids-21.06',                            **\n",
      "\t**  '/home/user/miniconda3/envs/rapids-21.06', '/usr', or similar such   **\n",
      "\t**  such that \"${CUML_PREFIX}/include/cuml\" is the directory of RAPIDS   **\n",
      "\t**  cuML header files and \"${CUML_PREFIX}/lib\" is the directory of       **\n",
      "\t**  RAPIDS cuML shared library files.).                                  **\n",
      "\t**                                                                       **\n",
      "\t***************************************************************************\n",
      "\t***************************************************************************\n"
    )
    FALSE
  } else {
    TRUE
  }
}

cuml_missing <- !has_cuml()

run_cmake <- function() {
  rc <- system2("which", "nvcc")

  if (rc != 0) {
    stop(
      "\n\n\n",
      "*********************************************************\n",
      "*                                                    \t*\n",
      "*    Unable to locate a CUDA compiler (nvcc).      \t*\n",
      "*                                                  \t*\n",
      "*    Please ensure it is present in PATH (e.g., run \t*\n",
      "*    `export PATH=\"${PATH}:/usr/local/cuda/bin\"` or  \t*\n",
      "*    similar) and try again.                       \t*\n",
      "*                                                    \t*\n",
      "*********************************************************\n\n\n"
    )
  }

  define(R_INCLUDE_DIR = R.home("include"))
  define(RCPP_INCLUDE_DIR = system.file("include", package = "Rcpp"))
  configure_file(file.path("src", "CMakeLists.txt.in"))

  wd <- getwd()
  on.exit(setwd(wd))
  setwd("src")

  cuml_prefix <- get_cuml_prefix()

  rc <- system2(
    "cmake",
    args = c(
      paste0("-DCUML_INCLUDE_DIR=", file.path(cuml_prefix, "include")),
      paste0("-DCUML_LIBRARY_DIR=", file.path(cuml_prefix, "lib")),
      paste0(
        "-DCUML_STUB_HEADERS_DIR=", normalizePath(file.path(getwd(), "stubs"))
      ),
      "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
      "."
    )
  )

  if (rc != 0) {
    stop("Failed to run 'cmake'!")
  }
}

if (!cuml_missing) {
  define(PKG_CPPFLAGS = '')
  run_cmake()
} else {
  define(PKG_CPPFLAGS = normalizePath(file.path(getwd(), "src", "stubs")))
}
