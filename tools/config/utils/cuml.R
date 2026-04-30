check_libcuml_path <- function(path) {
  cuml_headers_dir <- file.path(path, "include", "cuml")
  cuml_lib <- file.path(path, "lib", "libcuml++.so")
  dir.exists(cuml_headers_dir) && file.exists(cuml_lib)
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

      cuml_prefix <- bootstrap_libcuml_from_pip()
      if (!is.na(cuml_prefix)) {
        return(cuml_prefix)
      }

      # We will download a pre-built copy of `libcuml`
      return(NA_character_)
    }
  }

  return(cuml_prefix)
}

has_libcuml <- function(nvcc = find_nvcc()) {
  # this is here to make sure we only proceed to automatically downloading if we
  # find a compatible nvcc version.
  cuml_prefix <- get_cuml_prefix()
  if (is.na(cuml_prefix)) {
    if (identical(Sys.getenv("CUML_BOOTSTRAP_FAILED", unset = "0"), "1")) {
      FALSE
    } else if (can_download_libcuml(cuda_version = nvcc$version$major)) {
      # Skip subsequent checks if we are downloading a pre-built copy of `libcuml`
      TRUE
    } else {
      warning2(
        "No `libcuml` installation has been found.",
        paste0(
          "No bundled `libcuml` download is available for CUDA ",
          nvcc$version$major,
          " and cuML ",
          Sys.getenv("CUML_VERSION", unset = "21.08"),
          "."
        ),
        "Falling back to a stub-only build."
      )
      FALSE
    }
  } else {
    cuml_headers_dir <- file.path(cuml_prefix, "include", "cuml")
    cuml_lib <- file.path(cuml_prefix, "lib", "libcuml++.so")

    if (!check_libcuml_path(cuml_prefix)) {
      missing_paths <- c(cuml_headers_dir, cuml_lib)
      missing_paths <- missing_paths[!file.exists(missing_paths)]
      warning2(
        paste0("Invalid CUML_PREFIX: ", cuml_prefix),
        paste0("Missing expected path(s): ", paste(missing_paths, collapse = ", ")),
        "",
        "{cuda.ml} requires a valid RAPIDS installation.",
        "Please follow https://rapids.ai/start.html#get-rapids to install RAPIDS first"
      )
      warning2(
        "{cuda.ml} must be installed from an environment containing a valid",
        "CUML_PREFIX env variable such that \"${CUML_PREFIX}/include/cuml\"",
        "is the directory of RAPIDS cuML header files and \"${CUML_PREFIX}/lib\"",
        "is the directory of RAPIDS cuML shared library files. RAPIDS can be",
        "installed with pip, conda, or from source."
      )
      FALSE
    } else {
      TRUE
    }
  }
}

libcuml_download_url <- function(
  cuml_version = Sys.getenv("CUML_VERSION", unset = "21.08"),
  cuda_version = as.character(find_nvcc()$version$major)
) {
  url <- Sys.getenv("CUML_URL", unset = NA_character_)
  if (!is.na(url) && nzchar(url)) {
    return(url)
  }

  version_urls <- libcuml_versions[[cuml_version]]
  if (is.null(version_urls)) {
    return(NA_character_)
  }

  url <- version_urls[[as.character(cuda_version)]]
  if (is.null(url) || length(url) != 1L || is.na(url) || !nzchar(url)) {
    return(NA_character_)
  }

  url
}

can_download_libcuml <- function(
  cuml_version = Sys.getenv("CUML_VERSION", unset = "21.08"),
  cuda_version = as.character(find_nvcc()$version$major)
) {
  if (identical(Sys.getenv("DOWNLOAD_CUML", unset = "1"), "0")) {
    return(FALSE)
  }

  !is.na(libcuml_download_url(cuml_version, cuda_version))
}

download_libcuml <- function(cuml_version = Sys.getenv("CUML_VERSION", unset = "21.08")) {
  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())

  if (identical(Sys.getenv("DOWNLOAD_CUML", unset = "1"), "0")) {
    stop2("No `libcuml` installation has been found and downloading has been prevented by `DOWNLOAD_CUML=0`.")
  }

  old_timeout <- getOption("timeout")
  options(timeout = 1000)
  on.exit(options(timeout = old_timeout), add = TRUE)

  tmp <- tempfile(fileext = ".zip")
  cuda_version <- as.character(find_nvcc()$version$major)

  url <- libcuml_download_url(cuml_version, cuda_version)
  if (is.na(url)) {
    stop2(
      "No `libcuml` installation has been found.",
      paste0(
        "No bundled `libcuml` download is available for CUDA ",
        cuda_version,
        " and cuML ",
        cuml_version,
        "."
      ),
      "Set `CUML_PREFIX` to an existing RAPIDS installation or set `CUML_URL` to a compatible `libcuml` archive."
    )
  }

  download.file(url, tmp)
  unzip(tmp, exdir = ".")

  zip_file_name <- basename(url)
  dir_name <- gsub("\\.zip$", "", zip_file_name)
  file.rename(file.path(".", dir_name), file.path(".", "libcuml"))
}
