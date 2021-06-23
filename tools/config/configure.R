has_cuml <- function() {
  cuda_path <- Sys.getenv("CUDA_PATH", unset = NA_character_)
  cuml_missing <- (  
    if (is.na(cuda_path)) {
      warning("'CUDA_PATH' env variable is missing.")
      TRUE
    } else {
      cuml_headers_dir <- file.path(cuda_path, "include", "cuml")
      if (!dir.exists(cuml_headers_dir)) {
        warning(cuml_headers_dir, " does not exist or is not a directory.")
        TRUE
      } else {
        FALSE
      }
    }
  )

  if (cuml_missing) {
    warning("`cuml4r` requires a valid RAPIDS installation. ",
      "Please follow https://rapids.ai/start.html to install RAPIDS first.",
      "`cuml4r` must be installed and run from an environment containing ",
      "a valid CUDA_PATH env variable ",
      "(e.g., '/home/user/anaconda3/envs/rapids-21.06' or similar)."
    )
  }

  !cuml_missing
}

cuml_missing <- !has_cuml()

run_cmake <- function() {
  define(R_INCLUDE_DIR = R.home("include"))
  define(RCPP_INCLUDE_DIR = system.file("include", package = "Rcpp"))
  configure_file(file.path("src", "CMakeLists.txt.in"))

  wd <- getwd()
  on.exit(setwd(wd))
  setwd("src")
  cuda_path <- Sys.getenv("CUDA_PATH")
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
