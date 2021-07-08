get_cuda_path <- function() {
  cuda_path <- Sys.getenv("CUDA_PATH", unset = NA_character_)
  if (is.na(cuda_path)) {
    warning(
      "\t**********************************************\n",
      "\t**********************************************\n",
      "\t**                                          **\n",
      "\t**  'CUDA_PATH' env variable is missing --  **\n",
      "\t**  will boldly assume it is '/usr' !       **\n",
      "\t**                                          **\n",
      "\t**********************************************\n",
      "\t**********************************************\n",
      immediate. = TRUE
    )
    cuda_path <- "/usr"
  }

  cuda_path
}

has_cuml <- function() {
  cuda_path <- get_cuda_path()
  cuml_headers_dir <- file.path(cuda_path, "include", "cuml")

  if (!dir.exists(cuml_headers_dir)) {
    warning(
      "\t'", cuml_headers_dir, "' does not exist or is not a directory!\n\n",
      "\t***************************************************************************\n",
      "\t***************************************************************************\n",
      "\t**                                                                       **\n",
      "\t**  `cuml4r` requires a valid RAPIDS installation.                       **\n",
      "\t**  Please follow https://rapids.ai/start.html to install RAPIDS first.  **\n",
      "\t**  `cuml4r` must be installed and run from an environment containing    **\n",
      "\t**  a valid CUDA_PATH env variable\n                                     **\n",
      "\t**  (e.g., '/home/user/anaconda3/envs/rapids-21.06' or similar).         **\n",
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
  define(R_INCLUDE_DIR = R.home("include"))
  define(RCPP_INCLUDE_DIR = system.file("include", package = "Rcpp"))
  configure_file(file.path("src", "CMakeLists.txt.in"))

  wd <- getwd()
  on.exit(setwd(wd))
  setwd("src")
  cuda_path <- get_cuda_path()
  system2(
    "cmake",
    args = c(
      paste0("-DCUML_INCLUDE_DIR=", file.path(cuda_path, "include")),
      paste0("-DCUML_LIBRARY_DIR=", file.path(cuda_path, "lib")),
      "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
      "."
    )
  )
}

if (!cuml_missing) {
  run_cmake()
}
