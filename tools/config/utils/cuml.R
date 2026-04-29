check_libcuml_path <- function(path) {
  cuml_headers_dir <- file.path(path, "include", "cuml")
  dir.exists(cuml_headers_dir)
}

get_cuml_prefix <- function() {
  cuml_prefix <- Sys.getenv("CUML_PREFIX", unset = NA_character_)
  if (is.na(cuml_prefix)) {
    # Try the 'CUDA_PATH' env variable if it is present.
    cuml_prefix <- Sys.getenv("CUDA_PATH", unset = NA_character_)
  }
  if (is.na(cuml_prefix)) {
    cuml_prefix <- "/usr"
    if (check_libcuml_path(cuml_prefix)) {
      warning2(
        "'CUML_PREFIX' env variable is missing",
        "will boldly assume it is '/usr' !"
      )
      return(cuml_prefix)
    } else {

      # devtools::load_all() might run the config script from the `src` directory.
      cuml_prefix <- file.path(pkg_root(), "libcuml")
      if (check_libcuml_path(cuml_prefix)) {
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
