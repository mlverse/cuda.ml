cuml_pip_version <- function() {
  Sys.getenv("CUML_PIP_VERSION", unset = "26.4.0")
}

cuml_cuda_cccl_version <- function() {
  Sys.getenv("CUML_CUDA_CCCL_VERSION", unset = "0.6.0")
}

cuml_cran_like <- function() {
  nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_")) ||
    identical(Sys.getenv("CRAN", unset = ""), "true")
}

cuml_bootstrap_enabled <- function() {
  !identical(Sys.getenv("CUML_BOOTSTRAP", unset = "1"), "0") &&
    !cuml_cran_like()
}

cuml_bootstrap_cache_dir <- function() {
  cache_dir <- Sys.getenv("CUML_BOOTSTRAP_CACHE", unset = NA_character_)
  if (!is.na(cache_dir) && nzchar(cache_dir)) {
    return(normalizePath(cache_dir, mustWork = FALSE))
  }

  xdg_cache <- Sys.getenv("XDG_CACHE_HOME", unset = NA_character_)
  if (!is.na(xdg_cache) && nzchar(xdg_cache)) {
    return(file.path(xdg_cache, "cuda.ml"))
  }

  home <- Sys.getenv("HOME", unset = NA_character_)
  if (!is.na(home) && nzchar(home)) {
    return(file.path(home, ".cache", "cuda.ml"))
  }

  file.path(tempdir(), "cuda.ml")
}

cuml_cuda_suffix <- function(cuda_version) {
  major <- as.integer(cuda_version$major)
  if (major %in% c(12L, 13L)) {
    paste0("cu", major)
  } else {
    NA_character_
  }
}

cuml_bootstrap_prefix <- function(
  cuda_suffix,
  rapids_version = cuml_pip_version()
) {
  file.path(
    cuml_bootstrap_cache_dir(),
    "rapids",
    paste0("rapids-", rapids_version, "-", cuda_suffix)
  )
}

cuml_bootstrap_target <- function(
  cuda_suffix,
  rapids_version = cuml_pip_version()
) {
  file.path(
    cuml_bootstrap_cache_dir(),
    "wheel-targets",
    paste0("rapids-", rapids_version, "-", cuda_suffix)
  )
}

cuml_nvidia_gpu_available <- function() {
  nvidia_smi <- Sys.which("nvidia-smi")
  if (!nzchar(nvidia_smi)) {
    return(FALSE)
  }

  out <- tryCatch(
    suppressWarnings(
      system2(
        nvidia_smi,
        c("--query-gpu=name,driver_version", "--format=csv,noheader"),
        stdout = TRUE,
        stderr = TRUE
      )
    ),
    error = function(e) character()
  )
  status <- attr(out, "status", exact = TRUE)

  (is.null(status) || identical(status, 0L)) && length(out) > 0L && any(nzchar(out))
}

warn_missing_nvidia_gpu <- function() {
  warning2(
    "No usable NVIDIA GPU/driver was detected with `nvidia-smi`.",
    "Install or fix the NVIDIA driver, then verify that `nvidia-smi` lists",
    "your GPU before reinstalling {cuda.ml}.",
    "Falling back to a stub-only build."
  )
}

warn_missing_nvcc <- function() {
  warning2(
    "A CUDA compiler (`nvcc`) was not found.",
    "Install an NVIDIA CUDA Toolkit that includes `nvcc`, then verify that",
    "`nvcc --version` works. If the toolkit is installed outside `PATH`, set",
    "`CUDA_HOME` to the toolkit prefix before reinstalling {cuda.ml}.",
    "On Ubuntu, after adding NVIDIA's CUDA apt repository for your release:",
    "`sudo apt install cuda-toolkit`",
    "Falling back to a stub-only build."
  )
}

cuml_find_uv <- function() {
  uv <- Sys.which("uv")
  if (nzchar(uv)) {
    return(uv)
  }

  if (requireNamespace("reticulate", quietly = TRUE)) {
    uv <- tryCatch(reticulate:::uv_binary(), error = function(e) "")
    if (nzchar(uv) && file.exists(uv)) {
      return(uv)
    }
  }

  ""
}

cuml_installer_works <- function(command, args) {
  tryCatch(
    {
      out <- system2(command, args, stdout = TRUE, stderr = TRUE)
      status <- attr(out, "status", exact = TRUE)
      is.null(status) || identical(status, 0L)
    },
    error = function(e) FALSE
  )
}

cuml_find_package_installer <- function() {
  uv <- cuml_find_uv()
  if (nzchar(uv) && cuml_installer_works(uv, "--version")) {
    return(list(
      type = "uv",
      label = paste("uv", uv),
      command = uv,
      install_args = c("pip", "install")
    ))
  }

  for (python in c(Sys.which("python"), Sys.which("python3"))) {
    if (nzchar(python) && cuml_installer_works(python, c("-m", "pip", "--version"))) {
      return(list(
        type = "pip",
        label = paste("python -m pip", python),
        command = python,
        install_args = c("-m", "pip", "install")
      ))
    }
  }

  for (pip in c(Sys.which("pip"), Sys.which("pip3"))) {
    if (nzchar(pip) && cuml_installer_works(pip, "--version")) {
      return(list(
        type = "pip",
        label = paste("pip", pip),
        command = pip,
        install_args = "install"
      ))
    }
  }

  NULL
}

cuml_pip_packages <- function(cuda_suffix) {
  c(
    paste0("libcuml-", cuda_suffix, "==", cuml_pip_version()),
    paste0("cuda-cccl==", cuml_cuda_cccl_version())
  )
}

cuml_package_index_args <- function(installer) {
  if (identical(installer$type, "uv")) {
    c(
      "--no-config",
      "--index", "https://pypi.nvidia.com",
      "--default-index", "https://pypi.org/simple",
      "--index-strategy", "unsafe-best-match"
    )
  } else {
    c("--extra-index-url", "https://pypi.nvidia.com")
  }
}

cuml_package_install_args <- function(installer, target, packages) {
  c(
    installer$install_args,
    cuml_package_index_args(installer),
    "--target", target,
    "--only-binary", ":all:",
    "--upgrade",
    packages
  )
}

cuml_package_install_env <- function(installer) {
  if (identical(installer$type, "uv")) {
    c(
      "UV_NO_CONFIG=1",
      "UV_INDEX_STRATEGY=unsafe-best-match"
    )
  } else {
    character()
  }
}

cuml_package_install_command <- function(installer) {
  if (identical(installer$type, "uv")) {
    env <- unname(Sys.which("env"))
    if (nzchar(env)) env else "env"
  } else {
    installer$command
  }
}

cuml_package_install_command_args <- function(installer, args) {
  if (identical(installer$type, "uv")) {
    c("-u", "UV_EXCLUDE_NEWER", "-u", "UV_EXCLUDE_NEWER_PACKAGE", installer$command, args)
  } else {
    args
  }
}

cuml_run_package_install <- function(installer, target, packages) {
  dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  unlink(target, recursive = TRUE, force = TRUE)

  args <- cuml_package_install_args(installer, target, packages)
  env <- cuml_package_install_env(installer)

  status <- system2(
    cuml_package_install_command(installer),
    cuml_package_install_command_args(installer, args),
    env = env
  )
  identical(status, 0L)
}

copy_dir_contents <- function(src, dst) {
  if (!dir.exists(src)) {
    return(FALSE)
  }

  dir.create(dst, recursive = TRUE, showWarnings = FALSE)
  status <- system2("cp", c("-a", file.path(src, "."), dst))
  identical(status, 0L)
}

extract_cuml_pip_prefix <- function(target, prefix) {
  unlink(prefix, recursive = TRUE, force = TRUE)
  dir.create(file.path(prefix, "include"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(prefix, "lib"), recursive = TRUE, showWarnings = FALSE)

  for (pkg in c("libcuml", "libraft", "librmm", "rapids_logger")) {
    copy_dir_contents(file.path(target, pkg, "include"), file.path(prefix, "include"))

    for (libdir in c("lib", "lib64", ".libs")) {
      copy_dir_contents(file.path(target, pkg, libdir), file.path(prefix, "lib"))
    }
  }

  copy_dir_contents(
    file.path(target, "cuda", "cccl", "headers", "include"),
    file.path(prefix, "include")
  )
  copy_dir_contents(
    file.path(target, "cuda", "cccl", "headers", "lib"),
    file.path(prefix, "lib")
  )

  nvidia_dir <- file.path(target, "nvidia")
  if (dir.exists(nvidia_dir)) {
    for (component in list.files(nvidia_dir, full.names = TRUE)) {
      copy_dir_contents(file.path(component, "include"), file.path(prefix, "include"))
      copy_dir_contents(file.path(component, "lib"), file.path(prefix, "lib"))
    }
  }

  for (bundle_dir in list.files(target, pattern = "\\.libs$", full.names = TRUE)) {
    copy_dir_contents(bundle_dir, file.path(prefix, "lib"))
  }

  check_libcuml_path(prefix)
}

bootstrap_libcuml_from_pip <- function(nvcc = find_nvcc(stop_if_missing = FALSE)) {
  if (!cuml_bootstrap_enabled() || is.null(nvcc)) {
    return(NA_character_)
  }

  cuda_suffix <- cuml_cuda_suffix(nvcc$version)
  if (is.na(cuda_suffix)) {
    if (!can_download_libcuml(cuda_version = nvcc$version$major)) {
      warning2(
        paste0("Automatic RAPIDS pip bootstrap does not support CUDA ", nvcc$version, "."),
        "Install RAPIDS cuML yourself and set `CUML_PREFIX`, or install a supported",
        "CUDA toolkit and retry.",
        "Falling back to a stub-only build."
      )
    }
    return(NA_character_)
  }

  Sys.setenv(CUML_BOOTSTRAP_FAILED = "1")

  if (!cuml_nvidia_gpu_available()) {
    warn_missing_nvidia_gpu()
    return(NA_character_)
  }

  prefix <- cuml_bootstrap_prefix(cuda_suffix)
  if (check_libcuml_path(prefix)) {
    Sys.setenv(CUML_BOOTSTRAP_FAILED = "0")
    Sys.setenv(CUML_PREFIX = prefix)
    return(prefix)
  }

  installer <- cuml_find_package_installer()
  if (is.null(installer)) {
    warning2(
      "Unable to find a Python package installer for bootstrapping RAPIDS cuML.",
      "Install `uv` or install Python with pip, then reinstall {cuda.ml}.",
      "On Ubuntu, the Python fallback can be installed with:",
      "`sudo apt install python3 python3-pip python3-venv`",
      "Falling back to a stub-only build."
    )
    return(NA_character_)
  }

  target <- cuml_bootstrap_target(cuda_suffix)
  packages <- cuml_pip_packages(cuda_suffix)

  message(format_msg(
    "Bootstrapping RAPIDS cuML from pip wheels.",
    paste0("Installer: ", installer$label),
    paste0("Packages: ", paste(packages, collapse = ", ")),
    paste0("Prefix: ", prefix)
  ))

  if (!cuml_run_package_install(installer, target, packages)) {
    args <- cuml_package_install_args(installer, target, packages)
    warning2(
      "Failed to install RAPIDS cuML pip wheels.",
      "You can retry manually with:",
      paste(
        shQuote(cuml_package_install_command(installer)),
        paste(shQuote(cuml_package_install_command_args(installer, args)), collapse = " ")
      ),
      "Or install RAPIDS yourself and set `CUML_PREFIX`.",
      "Falling back to a stub-only build."
    )
    return(NA_character_)
  }

  if (!extract_cuml_pip_prefix(target, prefix)) {
    warning2(
      "RAPIDS cuML pip wheels were installed, but the expected C/C++ headers",
      "and shared libraries could not be extracted.",
      "Install RAPIDS yourself and set `CUML_PREFIX`.",
      "Falling back to a stub-only build."
    )
    return(NA_character_)
  }

  unlink(target, recursive = TRUE, force = TRUE)
  Sys.setenv(CUML_BOOTSTRAP_FAILED = "0")
  Sys.setenv(CUML_PREFIX = prefix)
  prefix
}
