check_libcuml_path <- function(path) {
  cuml_headers_dir <- file.path(path, "include", "cuml")
  cuml_libs <- file.path(path, "lib", c("libcuml.so", "libcuml++.so"))
  dir.exists(cuml_headers_dir) && any(file.exists(cuml_libs))
}

get_cuml_prefix <- function(nvcc = find_nvcc(stop_if_missing = FALSE)) {
  cuml_prefix <- Sys.getenv("CUML_PREFIX", unset = NA_character_)
  if (!is.na(cuml_prefix)) {
    return(cuml_prefix)
  }

  cuda_path <- Sys.getenv("CUDA_PATH", unset = NA_character_)
  if (!is.na(cuda_path) && check_libcuml_path(cuda_path)) {
    return(cuda_path)
  }

  cuml_prefix <- "/usr"
  if (check_libcuml_path(cuml_prefix)) {
    warning2(
      "'CUML_PREFIX' env variable is missing",
      "will boldly assume it is '/usr' !"
    )
    return(cuml_prefix)
  }

  # devtools::load_all() might run the config script from the `src` directory.
  cuml_prefix <- file.path(pkg_root(), "libcuml")
  if (check_libcuml_path(cuml_prefix)) {
    return(cuml_prefix)
  }

  cuml_prefix <- bootstrap_libcuml_from_pip(nvcc)
  if (!is.na(cuml_prefix)) {
    return(cuml_prefix)
  }

  NA_character_
}

has_libcuml <- function(nvcc = find_nvcc()) {
  cuml_prefix <- get_cuml_prefix(nvcc)
  if (is.na(cuml_prefix)) {
    if (!identical(Sys.getenv("CUML_BOOTSTRAP_FAILED", unset = "0"), "1")) {
      warning2(
        "No `libcuml` installation has been found.",
        "Install RAPIDS cuML 24.0 or newer and set `CUML_PREFIX`, or enable",
        "automatic bootstrap with `CUML_BOOTSTRAP=1`.",
        "Falling back to a stub-only build."
      )
    }
    FALSE
  } else {
    cuml_headers_dir <- file.path(cuml_prefix, "include", "cuml")
    cuml_libs <- file.path(
      cuml_prefix,
      "lib",
      c("libcuml.so", "libcuml++.so")
    )

    if (!check_libcuml_path(cuml_prefix)) {
      missing_paths <- cuml_headers_dir[!dir.exists(cuml_headers_dir)]
      if (!any(file.exists(cuml_libs))) {
        missing_paths <- c(
          missing_paths,
          paste(cuml_libs, collapse = " or ")
        )
      }
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
