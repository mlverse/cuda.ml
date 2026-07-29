cuml_managed_cuda_version <- function() {
  "13.2"
}

cuml_managed_cuda_toolkit_version <- function() {
  "13.2.2"
}

cuml_managed_cuda_component_version <- function() {
  "13.2.86"
}

cuml_managed_cuda_cccl_version <- function() {
  "0.6.0"
}

cuml_managed_rapids_version <- function() {
  "26.06"
}

cuml_managed_rapids_pip_version <- function() {
  "26.6.0"
}

cuml_managed_cuda_architectures <- function() {
  paste(
    c(
      "75-real",
      "80-real",
      "86-real",
      "89-real",
      "90-real",
      "100-real",
      "120-real",
      "120-virtual"
    ),
    collapse = ";"
  )
}

cuml_cran_like <- function() {
  nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_")) ||
    identical(Sys.getenv("CRAN", unset = ""), "true")
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

cuml_managed_bootstrap_prefix <- function() {
  file.path(
    cuml_bootstrap_cache_dir(),
    "managed-build",
    paste0(
      "cuda-", cuml_managed_cuda_toolkit_version(),
      "-rapids-", cuml_managed_rapids_pip_version()
    )
  )
}

cuml_managed_bootstrap_target <- function() {
  file.path(
    cuml_bootstrap_cache_dir(),
    "wheel-targets",
    paste0(
      "managed-cuda-", cuml_managed_cuda_toolkit_version(),
      "-rapids-", cuml_managed_rapids_pip_version()
    )
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

cuml_managed_pip_packages <- function() {
  component_version <- cuml_managed_cuda_component_version()
  rapids_version <- cuml_managed_rapids_pip_version()

  c(
    paste0("libcuml-cu13==", rapids_version),
    paste0("cuda-cccl==", cuml_managed_cuda_cccl_version()),
    paste0("cuda-toolkit==", cuml_managed_cuda_toolkit_version()),
    paste0("libnvforest-cu13==", rapids_version),
    paste0("libraft-cu13==", rapids_version),
    paste0("librmm-cu13==", rapids_version),
    "rapids-logger==0.2.3",
    "nvidia-cublas==13.4.1.3",
    paste0("nvidia-cuda-crt==", component_version),
    paste0("nvidia-cuda-cuobjdump==", component_version),
    paste0("nvidia-cuda-nvcc==", component_version),
    paste0("nvidia-cuda-nvrtc==", component_version),
    paste0("nvidia-cuda-runtime==", component_version),
    "nvidia-cufft==12.2.0.57",
    "nvidia-curand==10.4.2.66",
    "nvidia-cusolver==12.2.0.11",
    "nvidia-cusparse==12.7.10.12",
    "nvidia-nccl-cu13==2.30.7",
    paste0("nvidia-nvjitlink==", component_version),
    paste0("nvidia-nvvm==", component_version),
    "cuda-core==1.1.0",
    "cuda-pathfinder==1.6.0",
    "numpy==2.5.1",
    "typing-extensions==4.16.0"
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
    "--no-deps",
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

create_shared_library_linker_names <- function(lib_dir) {
  libraries <- list.files(
    lib_dir,
    pattern = "^lib.+[.]so[.][0-9].*$",
    full.names = TRUE
  )

  for (library in libraries) {
    linker_name <- sub("[.]so[.].*$", ".so", basename(library))
    linker_path <- file.path(lib_dir, linker_name)

    if (!file.exists(linker_path) && is.na(Sys.readlink(linker_path))) {
      if (!file.symlink(basename(library), linker_path)) {
        stop2("Failed to create the linker name for ", basename(library), ".")
      }
    }
  }

  invisible(TRUE)
}

extract_cuml_pip_prefix <- function(target, prefix) {
  unlink(prefix, recursive = TRUE, force = TRUE)
  dir.create(file.path(prefix, "include"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(prefix, "lib"), recursive = TRUE, showWarnings = FALSE)

  for (pkg in c(
    "libcuml",
    "libnvforest",
    "libraft",
    "librmm",
    "rapids_logger"
  )) {
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
      copy_dir_contents(file.path(component, "bin"), file.path(prefix, "bin"))
      copy_dir_contents(file.path(component, "nvvm"), file.path(prefix, "nvvm"))
    }
  }

  for (bundle_dir in list.files(target, pattern = "\\.libs$", full.names = TRUE)) {
    copy_dir_contents(bundle_dir, file.path(prefix, "lib"))
  }

  create_shared_library_linker_names(file.path(prefix, "lib"))

  check_libcuml_path(prefix)
}

cuml_managed_package_set_hash <- function() {
  digest::digest(
    paste(cuml_managed_pip_packages(), collapse = "\n"),
    algo = "sha256",
    serialize = FALSE
  )
}

cuml_managed_build_metadata <- function() {
  c(
    Schema = "1",
    CUDA = cuml_managed_cuda_version(),
    `CUDA-Toolkit` = cuml_managed_cuda_toolkit_version(),
    `CUDA-Component` = cuml_managed_cuda_component_version(),
    RAPIDS = cuml_managed_rapids_version(),
    `RAPIDS-Package` = cuml_managed_rapids_pip_version(),
    `Package-Set-SHA256` = cuml_managed_package_set_hash()
  )
}

cuml_managed_build_marker <- function(prefix) {
  file.path(prefix, "cuda-ml-managed-build.dcf")
}

write_cuml_managed_build_marker <- function(prefix) {
  metadata <- cuml_managed_build_metadata()
  write.dcf(
    matrix(
      unname(metadata),
      nrow = 1L,
      dimnames = list(NULL, names(metadata))
    ),
    file = cuml_managed_build_marker(prefix)
  )
  invisible(TRUE)
}

read_cuml_managed_build_marker <- function(prefix) {
  path <- cuml_managed_build_marker(prefix)
  if (!file.exists(path)) {
    return(NULL)
  }

  tryCatch(
    {
      metadata <- read.dcf(path)
      if (nrow(metadata) != 1L) {
        return(NULL)
      }
      metadata[1L, , drop = TRUE]
    },
    error = function(e) NULL
  )
}

cuda_header_version_from_prefix <- function(prefix) {
  header <- file.path(prefix, "include", "cuda.h")
  if (!file.exists(header)) {
    return(NA_character_)
  }

  lines <- readLines(header, warn = FALSE)
  line <- grep(
    "^#define[[:space:]]+CUDA_VERSION[[:space:]]+[0-9]+[[:space:]]*$",
    lines,
    value = TRUE
  )
  if (length(line) != 1L) {
    return(NA_character_)
  }

  version <- as.integer(sub(".*[[:space:]]", "", line))
  sprintf("%d.%d", version %/% 1000L, (version %% 1000L) %/% 10L)
}

nvcc_component_version_from_path <- function(nvcc) {
  output <- suppressWarnings(
    tryCatch(
      system2(nvcc, "--version", stdout = TRUE, stderr = TRUE),
      error = function(e) character()
    )
  )
  line <- grep(", V[0-9]+[.][0-9]+[.][0-9]+", output, value = TRUE)
  if (length(line) != 1L) {
    return(NA_character_)
  }
  sub(".*, V([0-9]+[.][0-9]+[.][0-9]+).*", "\\1", line)
}

check_managed_build_prefix <- function(prefix) {
  nvcc <- file.path(prefix, "bin", "nvcc")
  cuobjdump <- file.path(prefix, "bin", "cuobjdump")
  version <- nvcc_version_from_path(nvcc)
  metadata <- read_cuml_managed_build_marker(prefix)
  expected_metadata <- cuml_managed_build_metadata()
  linker_names <- file.path(
    prefix,
    "lib",
    c("libcublas.so", "libcudart.so", "libcusolver.so", "libcusparse.so")
  )

  check_libcuml_path(prefix) &&
    file.exists(nvcc) &&
    file.exists(cuobjdump) &&
    all(file.exists(linker_names)) &&
    !is.null(version) &&
    identical(
      paste(version$major, version$minor, sep = "."),
      cuml_managed_cuda_version()
    ) &&
    identical(
      nvcc_component_version_from_path(nvcc),
      cuml_managed_cuda_component_version()
    ) &&
    identical(
      cuda_header_version_from_prefix(prefix),
      cuml_managed_cuda_version()
    ) &&
    identical(
      cuml_version_from_prefix(prefix),
      cuml_managed_rapids_version()
    ) &&
    !is.null(metadata) &&
    all(names(expected_metadata) %in% names(metadata)) &&
    identical(
      unname(metadata[names(expected_metadata)]),
      unname(expected_metadata)
    )
}

bootstrap_managed_build_from_pip <- function() {
  stopifnot(cuml_r_universe_build())

  if (!cuml_linux_x86_64()) {
    stop2(
      "Managed {cuda.ml} builds are supported only on Linux x86_64.",
      paste0(
        "Detected: ", Sys.info()[["sysname"]], " ",
        Sys.info()[["machine"]], "."
      )
    )
  }

  prefix <- cuml_managed_bootstrap_prefix()
  if (check_managed_build_prefix(prefix)) {
    Sys.setenv(
      CUDA_HOME = prefix,
      CUDA_PATH = prefix,
      CUML_PREFIX = prefix
    )
    return(list(
      prefix = prefix,
      nvcc = list(
        path = file.path(prefix, "bin", "nvcc"),
        version = package_version(cuml_managed_cuda_version())
      )
    ))
  }

  installer <- cuml_find_package_installer()
  if (is.null(installer)) {
    stop2(
      "A managed R-universe build requires uv or Python 3 with pip.",
      "No package installer was found, so the pinned CUDA/RAPIDS build",
      "toolchain could not be provisioned."
    )
  }

  target <- cuml_managed_bootstrap_target()
  packages <- cuml_managed_pip_packages()

  message(format_msg(
    "Provisioning the managed CUDA/RAPIDS build toolchain.",
    paste0("Installer: ", installer$label),
    paste0("Packages: ", paste(packages, collapse = ", ")),
    paste0("Prefix: ", prefix)
  ))

  if (!cuml_run_package_install(installer, target, packages)) {
    stop2(
      "Failed to install the pinned CUDA/RAPIDS build wheels.",
      paste0("CUDA Toolkit: ", cuml_managed_cuda_toolkit_version()),
      paste0("RAPIDS cuML: ", cuml_managed_rapids_pip_version())
    )
  }

  if (!extract_cuml_pip_prefix(target, prefix)) {
    stop2(
      "The managed build wheels did not contain the expected cuML headers",
      "and shared libraries."
    )
  }

  write_cuml_managed_build_marker(prefix)
  unlink(target, recursive = TRUE, force = TRUE)

  if (!check_managed_build_prefix(prefix)) {
    stop2(
      "The managed build wheels did not contain a working CUDA 13.2 nvcc",
      "and RAPIDS cuML 26.06 prefix."
    )
  }

  Sys.setenv(
    CUDA_HOME = prefix,
    CUDA_PATH = prefix,
    CUML_PREFIX = prefix
  )

  list(
    prefix = prefix,
    nvcc = list(
      path = file.path(prefix, "bin", "nvcc"),
      version = package_version(cuml_managed_cuda_version())
    )
  )
}
