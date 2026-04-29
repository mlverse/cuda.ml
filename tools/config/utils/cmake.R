# NOTE: this version threshold may change depending on which branch of
#       https://github.com/rapidsai/rapids-cmake.git is being used
#       (e.g., see
#        https://github.com/rapidsai/rapids-cmake/blob/d5d59dbfaff7feafdd87700e8c2b347188897af7/CMakeLists.txt#L28
#        where the minimum cmake version is specified in the rapids-cmake repo in branch-21.10,
#        and
#        https://github.com/mlverse/cuda.ml/blob/7bad914c729011bcf05edc1c873609c518d9a77d/src/CMakeLists.txt.in#L13
#        where cuda.ml specifies which branch of the rapids-cmake repo to use)
cuda_ml_min_cmake_version <- numeric_version("3.21.1")

has_cmake <- function() {
  rc <- system2("which", "cmake", stdout = NULL, stderr = NULL)

  rc == 0
}

cmake_version <- function() {
  tryCatch(
    if (has_cmake()) {
      cmake_version_line <- system2("cmake", "--version", stdout = TRUE)[[1]]

      m <- regexec(".*cmake version (\\d+\\.\\d+\\.\\d+)", cmake_version_line)
      numeric_version(regmatches(cmake_version_line, m)[[1]][[2]])
    } else {
      NULL
    },
    error = function(e) NULL
  )
}

download_cmake <- function(cmake_version, exdir) {
  dl_timeout <- options("timeout")
  on.exit(options(timeout = dl_timeout), add = TRUE)

  dest <- paste0(tempfile(paste0("cmake-", cmake_version)), ".tar.gz")
  url <- paste0(
    "https://github.com/Kitware/CMake/releases/download/v", cmake_version,
    "/cmake-", cmake_version, "-linux-x86_64.tar.gz"
  )
  download.file(url = url, destfile = dest)

  untar(tarfile = dest, exdir = exdir)

  cmake_bin <- file.path(
    exdir, paste0("cmake-", cmake_version, "-linux-x86_64"), "bin", "cmake"
  )

  cmake_bin
}

# If 'cmake' is present in PATH and 'cmake --version' shows a version that is
# not lower than `cuda_ml_min_cmake_version`, then return 'cmake', otherwise
# download the required cmake binary (x86_64 only at the moment) to a temporary
# location and return the path to the temporary cmake binary.
#
# NOTE: if we want to support aarch64 in future, then this function needs to be
#       modified to download the correct binary on a non-x86_64 platform
find_or_download_cmake <- function(min_version, exdir) {
  existing_cmake_version <- cmake_version()
  if (is.null(existing_cmake_version) || existing_cmake_version < min_version) {
    download_cmake(cmake_version = min_version, exdir = exdir)
  } else {
    "cmake"
  }
}
