cuml_managed_cuda_version <- function() {
  unname(cuml_artifact_metadata()[["CUDA"]])
}

cuml_managed_cuda_toolkit_version <- function() {
  unname(cuml_artifact_metadata()[["CUDA-Toolkit"]])
}

cuml_managed_cuda_component_version <- function() {
  unname(cuml_artifact_metadata()[["CUDA-Component"]])
}

cuml_managed_cuda_cccl_version <- function() {
  unname(cuml_artifact_metadata()[["CUDA-CCCL"]])
}

cuml_managed_rapids_version <- function() {
  unname(cuml_artifact_metadata()[["RAPIDS"]])
}

cuml_managed_rapids_pip_version <- function() {
  unname(cuml_artifact_metadata()[["RAPIDS-Package"]])
}

cuml_managed_nvforest_version <- function() {
  unname(cuml_artifact_metadata()[["nvForest"]])
}

cuml_managed_treelite_version <- function() {
  unname(cuml_artifact_metadata()[["Treelite"]])
}

cuml_managed_cuda_architectures <- function() {
  unname(cuml_artifact_metadata()[["Architectures"]])
}

cuml_managed_platform <- function() {
  unname(cuml_artifact_metadata()[["Platform"]])
}

cuml_managed_minimum_driver <- function() {
  unname(cuml_artifact_metadata()[["Minimum-Driver"]])
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
      "cuda-",
      cuml_managed_cuda_toolkit_version(),
      "-rapids-",
      cuml_managed_rapids_pip_version()
    )
  )
}

cuml_managed_executables <- function(prefix) {
  file.path(
    prefix,
    c(
      "bin/__nvcc_device_query",
      "bin/bin2c",
      "bin/cudafe++",
      "bin/cuobjdump",
      "bin/fatbinary",
      "bin/nvcc",
      "bin/nvlink",
      "bin/ptxas",
      "nvvm/bin/cicc"
    )
  )
}

prepare_cuml_managed_executables <- function(prefix) {
  executables <- cuml_managed_executables(prefix)
  if (any(!file.exists(executables))) {
    stop2("The locked CUDA build artifacts are missing required executables.")
  }
  Sys.chmod(executables, mode = "0755")
  if (any(file.access(executables, mode = 1L) != 0L)) {
    stop2("The locked CUDA build executables could not be made executable.")
  }
  invisible(TRUE)
}

copy_dir_contents <- function(src, dst) {
  if (!dir.exists(src)) {
    return(FALSE)
  }

  dir.create(dst, recursive = TRUE, showWarnings = FALSE)
  status <- system2(
    "cp",
    c("-a", shQuote(file.path(src, ".")), shQuote(dst))
  )
  identical(status, 0L)
}

copy_required_dir <- function(src, dst) {
  if (!dir.exists(src) || !copy_dir_contents(src, dst)) {
    stop2("The locked build artifact layout is missing directory: ", src)
  }
  invisible(TRUE)
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

write_treelite_version_header <- function(prefix) {
  version <- strsplit(cuml_managed_treelite_version(), ".", fixed = TRUE)[[1L]]
  stopifnot(length(version) == 3L, all(grepl("^[0-9]+$", version)))
  writeLines(
    c(
      "#ifndef TREELITE_VERSION_H_",
      "#define TREELITE_VERSION_H_",
      "",
      paste0("#define TREELITE_VER_MAJOR ", version[[1L]]),
      paste0("#define TREELITE_VER_MINOR ", version[[2L]]),
      paste0("#define TREELITE_VER_PATCH ", version[[3L]]),
      paste0(
        "#define TREELITE_VERSION_STR \"",
        paste(version, collapse = "."),
        "\""
      ),
      "",
      "#endif  // TREELITE_VERSION_H_"
    ),
    file.path(prefix, "include", "treelite", "version.h")
  )
}

extract_cuml_artifact_prefix <- function(target, prefix) {
  unlink(prefix, recursive = TRUE, force = TRUE)
  dir.create(
    file.path(prefix, "include"),
    recursive = TRUE,
    showWarnings = FALSE
  )
  dir.create(file.path(prefix, "lib"), recursive = TRUE, showWarnings = FALSE)

  for (pkg in c(
    "libcuml",
    "libnvforest",
    "libraft",
    "librmm",
    "rapids_logger"
  )) {
    copy_required_dir(
      file.path(target, pkg, "include"),
      file.path(prefix, "include")
    )
    copy_required_dir(
      file.path(target, pkg, "lib64"),
      file.path(prefix, "lib")
    )
  }

  copy_required_dir(
    file.path(target, "cuda", "cccl", "headers", "include"),
    file.path(prefix, "include")
  )
  copy_required_dir(
    file.path(target, "nvidia", "cu13", "include"),
    file.path(prefix, "include")
  )
  copy_required_dir(
    file.path(target, "nvidia", "cu13", "lib"),
    file.path(prefix, "lib")
  )
  copy_required_dir(
    file.path(target, "nvidia", "cu13", "bin"),
    file.path(prefix, "bin")
  )
  copy_required_dir(
    file.path(target, "nvidia", "cu13", "nvvm"),
    file.path(prefix, "nvvm")
  )
  copy_required_dir(
    file.path(target, "nvidia", "nccl", "include"),
    file.path(prefix, "include")
  )
  copy_required_dir(
    file.path(target, "nvidia", "nccl", "lib"),
    file.path(prefix, "lib")
  )
  copy_required_dir(
    file.path(target, "libcuml_cu13.libs"),
    file.path(prefix, "lib")
  )
  copy_required_dir(
    file.path(target, "treelite", "lib"),
    file.path(prefix, "lib")
  )
  copy_required_dir(
    file.path(target, "treelite.libs"),
    file.path(prefix, "lib")
  )
  copy_required_dir(
    file.path(
      target,
      paste0("treelite-", cuml_managed_treelite_version()),
      "cpp_src",
      "include",
      "treelite"
    ),
    file.path(prefix, "include", "treelite")
  )
  write_treelite_version_header(prefix)

  prepare_cuml_managed_executables(prefix)
  create_shared_library_linker_names(file.path(prefix, "lib"))

  invisible(TRUE)
}

cuml_managed_artifact_lock_hash <- function() {
  cuml_artifact_hash(cuml_artifact_lock_path())
}

cuml_managed_build_metadata <- function() {
  c(
    Schema = "2",
    CUDA = cuml_managed_cuda_version(),
    `CUDA-Toolkit` = cuml_managed_cuda_toolkit_version(),
    `CUDA-Component` = cuml_managed_cuda_component_version(),
    RAPIDS = cuml_managed_rapids_version(),
    `RAPIDS-Package` = cuml_managed_rapids_pip_version(),
    nvForest = cuml_managed_nvforest_version(),
    Treelite = cuml_managed_treelite_version(),
    `Artifact-Lock-SHA256` = cuml_managed_artifact_lock_hash()
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

header_define_integer <- function(path, name) {
  if (!file.exists(path)) {
    return(NA_integer_)
  }
  pattern <- paste0(
    "^#define[[:space:]]+",
    name,
    "[[:space:]]+([0-9]+)[[:space:]]*$"
  )
  line <- grep(pattern, readLines(path, warn = FALSE), value = TRUE)
  if (length(line) != 1L) {
    return(NA_integer_)
  }
  as.integer(sub(pattern, "\\1", line))
}

nvforest_version_from_prefix <- function(prefix) {
  header <- file.path(prefix, "include", "nvforest", "version_config.hpp")
  values <- vapply(
    c("MAJOR", "MINOR", "PATCH"),
    function(part) {
      header_define_integer(header, paste0("NVForest_VERSION_", part))
    },
    integer(1)
  )
  if (anyNA(values)) {
    return(NA_character_)
  }
  sprintf("%d.%02d.%d", values[[1L]], values[[2L]], values[[3L]])
}

treelite_version_from_prefix <- function(prefix) {
  header <- file.path(prefix, "include", "treelite", "version.h")
  values <- vapply(
    c("MAJOR", "MINOR", "PATCH"),
    function(part) header_define_integer(header, paste0("TREELITE_VER_", part)),
    integer(1)
  )
  if (anyNA(values)) {
    return(NA_character_)
  }
  paste(values, collapse = ".")
}

check_managed_build_prefix <- function(prefix) {
  nvcc <- file.path(prefix, "bin", "nvcc")
  cuobjdump <- file.path(prefix, "bin", "cuobjdump")
  executables <- cuml_managed_executables(prefix)
  version <- nvcc_version_from_path(nvcc)
  metadata <- read_cuml_managed_build_marker(prefix)
  expected_metadata <- cuml_managed_build_metadata()
  linker_names <- file.path(
    prefix,
    "lib",
    c("libcublas.so", "libcudart.so", "libcusolver.so", "libcusparse.so")
  )
  required <- file.path(
    prefix,
    c(
      "include/nvforest/forest_model.hpp",
      "include/nvforest/treelite_importer.hpp",
      "include/treelite/tree.h",
      "include/treelite/version.h",
      "lib/libnvforest++.so",
      "lib/libtreelite.so",
      "lib/libgomp-855c301a.so.1.0.0",
      "lib/libgomp-a34b3233.so.1.0.0"
    )
  )

  check_functional_prefix(prefix) &&
    file.exists(nvcc) &&
    file.exists(cuobjdump) &&
    all(file.access(executables, mode = 1L) == 0L) &&
    all(file.exists(linker_names)) &&
    all(file.exists(required)) &&
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
    identical(
      nvforest_version_from_prefix(prefix),
      cuml_managed_nvforest_version()
    ) &&
    identical(
      treelite_version_from_prefix(prefix),
      cuml_managed_treelite_version()
    ) &&
    !is.null(metadata) &&
    all(names(expected_metadata) %in% names(metadata)) &&
    identical(
      unname(metadata[names(expected_metadata)]),
      unname(expected_metadata)
    )
}

bootstrap_managed_build_from_artifacts <- function() {
  stopifnot(identical(cuml_build_mode(), "managed"))

  if (!cuml_ubuntu_2604_x86_64()) {
    stop2(
      "Managed {cuda.ml} builds require Ubuntu 26.04 x86_64.",
      paste0(
        "Detected: ",
        Sys.info()[["sysname"]],
        " ",
        Sys.info()[["machine"]],
        "."
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

  artifacts <- cuml_artifact_lock()
  artifacts <- artifacts[artifacts$build, , drop = FALSE]
  staging_root <- file.path(cuml_bootstrap_cache_dir(), "staging")
  dir.create(staging_root, recursive = TRUE, showWarnings = FALSE)
  target <- tempfile("managed-build-", tmpdir = staging_root)
  dir.create(target)
  on.exit(unlink(target, recursive = TRUE, force = TRUE), add = TRUE)
  message(format_msg(
    "Provisioning the managed CUDA/RAPIDS build toolchain.",
    paste0("Locked artifacts: ", nrow(artifacts)),
    paste0("Prefix: ", prefix)
  ))
  for (i in seq_len(nrow(artifacts))) {
    cuml_extract_artifact(artifacts[i, , drop = FALSE], target)
  }

  extract_cuml_artifact_prefix(target, prefix)
  write_cuml_managed_build_marker(prefix)

  if (!check_managed_build_prefix(prefix)) {
    stop2(
      "The locked artifacts did not produce the exact managed build prefix.",
      paste0("CUDA Toolkit: ", cuml_managed_cuda_toolkit_version()),
      paste0("RAPIDS cuML: ", cuml_managed_rapids_version()),
      paste0("nvForest: ", cuml_managed_nvforest_version()),
      paste0("Treelite: ", cuml_managed_treelite_version())
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
