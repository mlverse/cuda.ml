# CMake 3.21 supports explicit `-real` and `-virtual` CUDA architecture
# suffixes, which the portable backend uses to control SASS and PTX output.
cuda_ml_min_cmake_version <- numeric_version("3.21.1")
cuda_ml_min_gnu_cxx_version <- numeric_version("14")

find_cuda_ml_cxx <- function(path) {
  if (!nzchar(path)) {
    stop2(
      "A functional cuda.ml build requires an explicit GNU C++ compiler.",
      "Use GCC 14 or newer."
    )
  }
  path <- normalizePath(path, mustWork = TRUE)
  output <- tryCatch(
    system2(path, "-dumpfullversion", stdout = TRUE, stderr = TRUE),
    error = function(e) character()
  )
  if (length(output) != 1L || !grepl("^[0-9]+([.][0-9]+)*$", output)) {
    stop2("Unable to determine the CUDA host C++ compiler version.")
  }
  version <- numeric_version(output)
  if (version < cuda_ml_min_gnu_cxx_version) {
    stop2(
      paste0("GNU C++ ", version, " is too old for nvForest 26.06."),
      "Use GCC 14 or newer."
    )
  }
  path
}

find_cmake <- function() {
  cmake <- unname(Sys.which("cmake"))
  if (!nzchar(cmake)) {
    stop2(
      "A functional cuda.ml build requires CMake.",
      paste0("Install CMake ", cuda_ml_min_cmake_version, " or newer.")
    )
  }

  output <- tryCatch(
    system2(cmake, "--version", stdout = TRUE, stderr = TRUE),
    error = function(e) character()
  )
  if (!length(output)) {
    stop2("Unable to determine the installed CMake version.")
  }
  match <- regexec("^cmake version ([0-9]+[.][0-9]+[.][0-9]+)$", output[[1L]])
  value <- regmatches(output[[1L]], match)[[1L]]
  if (length(value) != 2L) {
    stop2("Unable to determine the installed CMake version.")
  }
  version <- numeric_version(value[[2L]])
  if (version < cuda_ml_min_cmake_version) {
    stop2(
      paste0("CMake ", version, " is too old."),
      paste0("Install CMake ", cuda_ml_min_cmake_version, " or newer.")
    )
  }
  cmake
}
