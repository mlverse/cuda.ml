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
format_msg <- function(...) {
  msg <- c(...)

  if (require(cli, quietly = TRUE)) {
    msg <- cli::ansi_strwrap(msg)
    msg <- cli::boxx(msg, border_style = "double")
  } else {
    msg <- strwrap(msg)
    msg <- paste("*\t", c("", msg, ""))
    msg <- paste(msg, collapse = "\n")
    starline <- paste(rep("*", 0.9 * getOption("width")), collapse = "")
    msg <- paste(c(starline, msg, starline), collapse = "\n")
  }

  msg
}

stop2 <- function(...) {
  stop("\n", format_msg(...), call. = FALSE)
}

warning2 <- function(...) {
  warning("\n", format_msg(...), immediate. = TRUE, call. = FALSE)
}

check_path <- function(path) {
  cuml_headers_dir <- file.path(path, "include", "cuml")
  dir.exists(cuml_headers_dir)
}

# A list containing libcuml download links for "cuml_versions" and CUDA major versions.
libcuml_versions <- list(
  "21.08" = list(
    "11" = "https://github.com/mlverse/libcuml-builds/releases/download/v21.08-cuda11.2.1/libcuml-21.08-cuda11.2.1.zip"
  ),
  "21.10" = list(
    "11" = "https://github.com/mlverse/libcuml-builds/releases/download/v21.10-cuda11.2.1/libcuml-21.10-cuda11.2.1.zip"
  )
)

download_libcuml <- function(cuml_version = Sys.getenv("CUML_VERSION", unset = "21.08")) {
  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())

  if (Sys.getenv("DOWNLOAD_CUML", unset = 1) == 0) {
    stop2("No `libcuml` installation has been found and downloading has been prevented by `CUML_NO_DOWNLOAD`.")
  }

  old_timeout <- getOption("timeout")
  options(timeout = 1000)
  on.exit(options(timeout = old_timeout), add = TRUE)

  tmp <- tempfile(fileext = ".zip")
  cuda_version <- as.character(find_nvcc()$version$major)

  url <- Sys.getenv("CUML_URL")
  if (!nzchar(url)) {
    url <- libcuml_versions[[cuml_version]][[cuda_version]]
  }

  download.file(url, tmp)
  unzip(tmp, exdir = ".")

  zip_file_name <- basename(url)
  dir_name <- gsub("\\.zip$", "", zip_file_name)
  file.rename(file.path(".", dir_name), file.path(".", "libcuml"))
}

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

get_cuml_prefix <- function() {
  cuml_prefix <- Sys.getenv("CUML_PREFIX", unset = NA_character_)
  if (is.na(cuml_prefix)) {
    # Try the 'CUDA_PATH' env variable if it is present.
    cuml_prefix <- Sys.getenv("CUDA_PATH", unset = NA_character_)
  }
  if (is.na(cuml_prefix)) {
    cuml_prefix <- "/usr"
    if (check_path(cuml_prefix)) {
      warning2(
        "'CUML_PREFIX' env variable is missing",
        "will boldly assume it is '/usr' !"
      )
      return(cuml_prefix)
    } else {

      # devtools::load_all() might run the config script from the `src` directory.
      cuml_prefix <- file.path(pkg_root(), "libcuml")
      if (check_path(cuml_prefix)) {
        return(cuml_prefix)
      }

      # We will download a pre-built copy of `libcuml`
      return(NA_character_)
    }
  }

  return(cuml_prefix)
}

has_libcuml <- function() {

  # this is here to make sure we only proceed to automatically downloading if we
  # find a compatible nvcc version.
  find_nvcc()

  cuml_prefix <- get_cuml_prefix()
  if (is.na(cuml_prefix)) {
    # Skip subsequent checks if we are downloading a pre-built copy of `libcuml`
    TRUE
  } else {
    cuml_headers_dir <- file.path(cuml_prefix, "include", "cuml")

    if (!dir.exists(cuml_headers_dir)) {
      warning2(
        paste0(cuml_headers_dir, " does not exist or is not a directory!"),
        "",
        "{cuda.ml} requires a valid RAPIDS installation.",
        "Please follow https://rapids.ai/start.html to install RAPIDS first"
      )
      warning2(
        "{cuda.ml} must be installed from an environment containing a valid",
        "CUML_PREFIX env variable such that \"${CUML_PREFIX}/include/cuml\"",
        "is the directory of RAPIDS cuML header files and \"${CUML_PREFIX}/lib\"",
        "is the directory of RAPIDS cuML shared library files.)."
      )
      FALSE
    } else {
      TRUE
    }
  }
}


nvcc_version_from_path <- function(nvcc) {
  suppressWarnings(
    nvcc <- tryCatch(system2(nvcc, "--version", stdout = TRUE, stderr = TRUE), error = function(e) NULL)
  )

  if (is.null(nvcc) || !any(grepl("release", nvcc))) {
    return(NULL)
  }

  version <- gsub(".*release |, V.*", "", nvcc[grepl("release", nvcc)])
  package_version(version)
}

find_nvcc <- function(stop_if_missing = TRUE) {

  # Check if nvcc from path is available
  nvcc_path <- "nvcc"
  cuda_version <- nvcc_version_from_path(nvcc_path)

  # Check if nvcc from CUDA_HOME is available
  cuda_home <- Sys.getenv("CUDA_HOME")
  if (nzchar(cuda_home) && is.null(cuda_version)) {
    nvcc_path <- file.path(cuda_home, "bin", "nvcc")
    cuda_version <- nvcc_version_from_path(nvcc_path)
  }

  # Check nvcc from default install location.
  if (is.null(cuda_version)) {
    nvcc_path <- "/usr/local/cuda/bin/nvcc"
    cuda_version <- nvcc_version_from_path(nvcc_path)
  }

  # No nvcc found! Error!
  if (is.null(cuda_version)) {
    if (stop_if_missing) {
      stop2(
        "Unable to locate a CUDA compiler (nvcc).",
        "Please ensure it is present in PATH (e.g., run",
        "`export PATH=\"${PATH}:/usr/local/cuda/bin\"` or",
        "similar) and try again."
      )
    } else {
      return(NULL)
    }
  }

  # Nvcc found but wrong cuda version.
  minimum_supported <- package_version("11.0")
  if (cuda_version < minimum_supported) {
    stop2(
      paste0("Found nvcc '", nvcc_path, "'"),
      paste0("CUDA version '", cuda_version, "' is not supported."),
      paste0("The minimum required version is '", minimum_supported, "'")
    )
  }

  # return nvcc path.
  return(list(path = nvcc_path, version = cuda_version))
}

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
  rc <- system2("cmake", args = cmake_args)

  if (rc != 0) {
    stop("Failed to run 'cmake'!")
  }
}

if (is.null(find_nvcc(stop_if_missing = FALSE)) || !has_libcuml()) {
  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())
  define(PKG_CPPFLAGS = normalizePath(file.path(getwd(), "src", "stubs")))
} else {
  define(PKG_CPPFLAGS = "")
  run_cmake()
}
