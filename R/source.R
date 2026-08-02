cuda_ml_source_lock_metadata <- function() {
  path <- system.file(
    "artifacts",
    paste0(cuda_ml_platform(), ".tsv"),
    package = "cuda.ml"
  )
  if (!nzchar(path)) {
    stop("The cuda.ml source-build lock is missing.", call. = FALSE)
  }

  lines <- readLines(path, warn = FALSE)
  lines <- lines[startsWith(lines, "# ")]
  matches <- regmatches(lines, regexec("^# ([^:]+): (.*)$", lines))
  if (!length(matches) || any(lengths(matches) != 3L)) {
    stop("The cuda.ml source-build lock is invalid.", call. = FALSE)
  }
  metadata <- vapply(matches, `[[`, character(1), 3L)
  names(metadata) <- vapply(matches, `[[`, character(1), 2L)
  required <- c(
    "Schema",
    "CUDA",
    "CUDA-Toolkit",
    "CUDA-Component",
    "RAPIDS",
    "nvForest",
    "Treelite",
    "CMake",
    "Ninja",
    "Platform"
  )
  if (
    !all(required %in% names(metadata)) ||
      !identical(unname(metadata[["Schema"]]), "1") ||
      !identical(unname(metadata[["Platform"]]), cuda_ml_platform())
  ) {
    stop("The cuda.ml source-build lock is invalid.", call. = FALSE)
  }
  metadata
}

cuda_ml_source_header_integer <- function(path, name) {
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

cuda_ml_source_cuml_version <- function(prefix) {
  header <- file.path(prefix, "include", "cuml", "version_config.hpp")
  values <- vapply(
    c("MAJOR", "MINOR"),
    function(part) {
      cuda_ml_source_header_integer(header, paste0("CUML_VERSION_", part))
    },
    integer(1)
  )
  if (anyNA(values)) {
    return(NA_character_)
  }
  sprintf("%d.%02d", values[[1L]], values[[2L]])
}

cuda_ml_source_nvforest_version <- function(prefix) {
  header <- file.path(prefix, "include", "nvforest", "version_config.hpp")
  values <- vapply(
    c("MAJOR", "MINOR", "PATCH"),
    function(part) {
      cuda_ml_source_header_integer(
        header,
        paste0("NVForest_VERSION_", part)
      )
    },
    integer(1)
  )
  if (anyNA(values)) {
    return(NA_character_)
  }
  sprintf("%d.%02d.%d", values[[1L]], values[[2L]], values[[3L]])
}

cuda_ml_source_treelite_version <- function(prefix) {
  header <- file.path(prefix, "include", "treelite", "version.h")
  values <- vapply(
    c("MAJOR", "MINOR", "PATCH"),
    function(part) {
      cuda_ml_source_header_integer(header, paste0("TREELITE_VER_", part))
    },
    integer(1)
  )
  if (anyNA(values)) {
    return(NA_character_)
  }
  paste(values, collapse = ".")
}

cuda_ml_source_tool_version <- function(path, argument, pattern) {
  output <- suppressWarnings(tryCatch(
    system2(path, argument, stdout = TRUE, stderr = TRUE),
    error = function(e) character()
  ))
  line <- grep(pattern, output, value = TRUE)
  if (length(line) != 1L) {
    return(NA_character_)
  }
  sub(pattern, "\\1", line)
}

cuda_ml_source_cuda_libdir <- function(cuda_home) {
  candidates <- unique(file.path(
    cuda_home,
    c("lib64", "targets/x86_64-linux/lib", "lib")
  ))
  required <- c(
    "libcudart.so",
    "libcublas.so",
    "libcusolver.so",
    "libcusparse.so"
  )
  valid <- vapply(
    candidates,
    function(path) all(file.exists(file.path(path, required))),
    logical(1)
  )
  candidates <- candidates[valid]
  if (!length(candidates)) {
    stop(
      "CUDA_HOME must contain one library directory with the CUDA 13.2 ",
      "development libraries.",
      call. = FALSE
    )
  }
  normalizePath(candidates[[1L]], mustWork = TRUE)
}

cuda_ml_source_detect_architectures <- function() {
  nvidia_smi <- unname(Sys.which("nvidia-smi"))
  if (!nzchar(nvidia_smi)) {
    return(character())
  }
  output <- suppressWarnings(tryCatch(
    system2(
      nvidia_smi,
      c("--query-gpu=index,uuid,compute_cap", "--format=csv,noheader"),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(e) character()
  ))
  status <- attr(output, "status", exact = TRUE)
  fields <- strsplit(output, ",", fixed = TRUE)
  if (
    (!is.null(status) && status != 0L) ||
      !length(fields) ||
      any(lengths(fields) != 3L)
  ) {
    return(character())
  }
  indexes <- vapply(fields, function(x) trimws(x[[1L]]), character(1))
  uuids <- vapply(fields, function(x) trimws(x[[2L]]), character(1))
  capabilities <- vapply(fields, function(x) trimws(x[[3L]]), character(1))
  if (
    any(!grepl("^[0-9]+$", indexes)) ||
      any(!nzchar(uuids)) ||
      any(!grepl("^[0-9]+[.][0-9]+$", capabilities))
  ) {
    return(character())
  }

  visible <- Sys.getenv("CUDA_VISIBLE_DEVICES", unset = NA_character_)
  if (!is.na(visible)) {
    visible <- trimws(visible)
    identifiers <- trimws(strsplit(visible, ",", fixed = TRUE)[[1L]])
    if (
      !nzchar(visible) ||
        !length(identifiers) ||
        any(!nzchar(identifiers)) ||
        any(grepl("^-[0-9]+$", identifiers))
    ) {
      return(character())
    }
    selected <- lapply(
      identifiers,
      function(identifier) {
        if (grepl("^[0-9]+$", identifier)) {
          which(indexes == identifier)
        } else {
          which(startsWith(uuids, identifier))
        }
      }
    )
    if (any(lengths(selected) != 1L)) {
      return(character())
    }
    capabilities <- capabilities[unlist(selected, use.names = FALSE)]
  }

  paste0(
    sort(unique(as.integer(sub(".", "", capabilities, fixed = TRUE)))),
    "-real"
  )
}

cuda_ml_source_native_architectures <- function(required) {
  stopifnot(is.logical(required), length(required) == 1L, !is.na(required))
  architectures <- cuda_ml_source_detect_architectures()
  if (!length(architectures)) {
    if (required) {
      stop(
        "Unable to detect architectures = \"native\" with nvidia-smi. ",
        "Ensure a CUDA-visible NVIDIA GPU and nvidia-smi are available, or ",
        "supply an explicit target such as architectures = \"86-real\".",
        call. = FALSE
      )
    }
    return(NULL)
  }
  message(
    "Detected CUDA architectures: ",
    paste(architectures, collapse = ", "),
    ". Use architectures = \"portable\" for a relocatable or ",
    "heterogeneous-GPU build."
  )
  paste(architectures, collapse = ";")
}

cuda_ml_source_architectures <- function(value) {
  stopifnot(is.character(value), length(value) == 1L, !is.na(value))
  if (identical(value, "native")) {
    return(cuda_ml_source_native_architectures(required = TRUE))
  }
  if (identical(value, "portable")) {
    message("Using portable CUDA architectures.")
    value <- unname(.cuda_ml_state$metadata[["Architectures"]])
  }
  architectures <- strsplit(value, ";", fixed = TRUE)[[1L]]
  if (
    !length(architectures) ||
      any(!grepl("^[0-9]+(-(real|virtual))?$", architectures)) ||
      anyDuplicated(architectures) > 0L
  ) {
    stop(
      "CUML_CUDA_ARCHITECTURES must be \"native\", \"portable\", or an ",
      "explicit semicolon-separated CMake CUDA architecture list.",
      call. = FALSE
    )
  }
  paste(architectures, collapse = ";")
}

cuda_ml_source_compiler_error <- function(path, version = NA_character_) {
  stopifnot(
    is.character(path),
    length(path) == 1L,
    !is.na(path),
    is.character(version),
    length(version) == 1L
  )
  message <- paste(
    "A source installation requires GNU C++ 14 or newer.",
    "The pinned nvForest 26.06 binary uses GCC 14 C++ symbol mangling."
  )
  if (!is.na(version)) {
    message <- paste0(
      message,
      " Found ",
      path,
      " at version ",
      version,
      "."
    )
  }

  cxx <- unname(Sys.which("g++-14"))
  cxx_version <- cuda_ml_source_tool_version(
    cxx,
    "-dumpfullversion",
    "^([0-9]+([.][0-9]+)*)$"
  )
  if (
    nzchar(cxx) &&
      !is.na(cxx_version) &&
      base::package_version(cxx_version) >= base::package_version("14.0")
  ) {
    stop(
      message,
      " g++-14 is installed but not selected. In R, run ",
      "`Sys.setenv(CUDA_ML_CXX = ",
      encodeString(cxx, quote = '"'),
      ")`, then retry.",
      call. = FALSE
    )
  }

  if (nzchar(Sys.which("apt"))) {
    stop(
      message,
      " On Debian or Ubuntu, run ",
      "`sudo apt update && sudo apt install g++-14`, then in R run ",
      "`Sys.setenv(CUDA_ML_CXX = \"/usr/bin/g++-14\")`, then retry.",
      call. = FALSE
    )
  }

  stop(message, call. = FALSE)
}

cuda_ml_source_compiler <- function(path) {
  stopifnot(is.character(path), length(path) == 1L, !is.na(path))
  if (!nzchar(path) || !file.exists(path)) {
    cuda_ml_source_compiler_error(path)
  }
  path <- normalizePath(path, mustWork = TRUE)
  cxx_version <- cuda_ml_source_tool_version(
    path,
    "-dumpfullversion",
    "^([0-9]+([.][0-9]+)*)$"
  )
  if (
    is.na(cxx_version) ||
      base::package_version(cxx_version) < base::package_version("14.0")
  ) {
    cuda_ml_source_compiler_error(path, cxx_version)
  }
  list(path = path, version = cxx_version)
}

cuda_ml_source_cmake <- function(path = unname(Sys.which("cmake"))) {
  stopifnot(is.character(path), length(path) == 1L, !is.na(path))
  cmake <- path
  cmake_version <- cuda_ml_source_tool_version(
    cmake,
    "--version",
    "^cmake version ([0-9]+[.][0-9]+[.][0-9]+)$"
  )
  if (
    !nzchar(cmake) ||
      is.na(cmake_version) ||
      base::package_version(cmake_version) < base::package_version("3.21.1")
  ) {
    stop("A source installation requires CMake 3.21.1 or newer.", call. = FALSE)
  }
  list(
    path = normalizePath(cmake, mustWork = TRUE),
    version = cmake_version
  )
}

cuda_ml_source_build_inputs <- function(dependencies, architectures = NULL) {
  stopifnot(
    is.character(dependencies),
    length(dependencies) == 1L,
    !is.na(dependencies),
    dependencies %in% c("managed", "host"),
    is.null(architectures) ||
      (
        is.character(architectures) &&
          length(architectures) == 1L &&
          !is.na(architectures)
      )
  )

  platform <- cuda_ml_platform()
  source <- system.file("backend-src", package = "cuda.ml")
  artifact_lock_path <- system.file(
    "artifacts",
    paste0(platform, ".tsv"),
    package = "cuda.ml"
  )
  if (!nzchar(source) || !dir.exists(source)) {
    stop("The cuda.ml backend source is missing.", call. = FALSE)
  }
  if (!nzchar(artifact_lock_path) || !file.exists(artifact_lock_path)) {
    stop("The cuda.ml source-build lock is missing.", call. = FALSE)
  }

  if (identical(dependencies, "managed")) {
    cxx_path <- Sys.getenv("CUDA_ML_CXX", unset = "")
    if (!nzchar(cxx_path)) {
      cxx_path <- unname(Sys.which("g++-14"))
      if (!nzchar(cxx_path)) {
        cxx_path <- unname(Sys.which("g++"))
      }
    }
    compiler <- cuda_ml_source_compiler(cxx_path)
    tools <- cuda_ml_source_build_tools()
    if (is.null(architectures)) {
      architectures <- cuda_ml_source_native_architectures(required = FALSE)
      if (is.null(architectures)) {
        message(
          "No CUDA-visible NVIDIA GPU was detected; using portable CUDA ",
          "architectures."
        )
        architectures <- tools$cuml_managed_cuda_architectures()
      }
    }
    architectures <- cuda_ml_source_architectures(architectures)

    bootstrap_lock <- cuda_ml_acquire_lock("source-toolchain")
    on.exit(filelock::unlock(bootstrap_lock), add = TRUE)
    managed <- tools$bootstrap_managed_build_from_artifacts(compiler$path)
    cuda_home <- managed$prefix
    cuml_prefix <- managed$prefix
    cmake <- cuda_ml_source_cmake(managed$cmake)
    ninja <- normalizePath(managed$ninja, mustWork = TRUE)
    ninja_version <- cuda_ml_source_tool_version(
      ninja,
      "--version",
      "^([0-9]+[.][0-9]+[.][0-9]+)([.].*)?$"
    )
    if (is.na(ninja_version)) {
      stop("The managed Ninja executable is invalid.", call. = FALSE)
    }
  } else {
    names <- c(
      "CUDA_HOME",
      "CUML_PREFIX",
      "CUML_CUDA_ARCHITECTURES",
      "CUDA_ML_CXX"
    )
    values <- Sys.getenv(names, unset = "")
    if (!is.null(architectures)) {
      values[["CUML_CUDA_ARCHITECTURES"]] <- architectures
    }
    if (any(!nzchar(values))) {
      stop(
        "A host source installation requires explicit build inputs. ",
        "Set CUDA_HOME, CUML_PREFIX, CUML_CUDA_ARCHITECTURES, and ",
        "CUDA_ML_CXX.",
        call. = FALSE
      )
    }
    cuda_home <- normalizePath(values[["CUDA_HOME"]], mustWork = TRUE)
    cuml_prefix <- normalizePath(values[["CUML_PREFIX"]], mustWork = TRUE)
    architectures <- cuda_ml_source_architectures(
      values[["CUML_CUDA_ARCHITECTURES"]]
    )
    compiler <- cuda_ml_source_compiler(values[["CUDA_ML_CXX"]])
    cmake <- cuda_ml_source_cmake()
    ninja <- NULL
    ninja_version <- NULL
  }

  nvcc <- file.path(cuda_home, "bin", "nvcc")
  if (!file.exists(nvcc)) {
    stop("CUDA_HOME does not contain bin/nvcc.", call. = FALSE)
  }

  lock <- cuda_ml_source_lock_metadata()
  nvcc_output <- suppressWarnings(tryCatch(
    system2(nvcc, "--version", stdout = TRUE, stderr = TRUE),
    error = function(e) character()
  ))
  nvcc_release <- grep("release [0-9]+[.][0-9]+", nvcc_output, value = TRUE)
  nvcc_component <- grep(", V[0-9]+[.][0-9]+[.][0-9]+", nvcc_output, value = TRUE)
  if (length(nvcc_release) == 1L) {
    nvcc_release <- sub(".*release ([0-9]+[.][0-9]+),.*", "\\1", nvcc_release)
  }
  if (length(nvcc_component) == 1L) {
    nvcc_component <- sub(
      ".*, V([0-9]+[.][0-9]+[.][0-9]+).*",
      "\\1",
      nvcc_component
    )
  }
  if (
    !identical(nvcc_release, unname(lock[["CUDA"]])) ||
      !identical(nvcc_component, unname(lock[["CUDA-Component"]]))
  ) {
    stop(
      "CUDA_HOME must contain the locked CUDA ",
      lock[["CUDA-Toolkit"]],
      " toolchain (nvcc ",
      lock[["CUDA-Component"]],
      ").",
      call. = FALSE
    )
  }

  required <- file.path(
    cuml_prefix,
    c(
      "include/cuml/version_config.hpp",
      "include/nvforest/forest_model.hpp",
      "include/nvforest/treelite_importer.hpp",
      "include/nvforest/version_config.hpp",
      "include/treelite/c_api.h",
      "include/treelite/tree.h",
      "include/treelite/version.h",
      "lib/libcuml.so",
      "lib/libnvforest++.so",
      "lib/libtreelite_static.a"
    )
  )
  versions <- c(
    cuda_ml_source_cuml_version(cuml_prefix),
    cuda_ml_source_nvforest_version(cuml_prefix),
    cuda_ml_source_treelite_version(cuml_prefix)
  )
  expected <- unname(lock[c("RAPIDS", "nvForest", "Treelite")])
  if (any(!file.exists(required)) || !identical(versions, expected)) {
    stop(
      "CUML_PREFIX must contain cuML and nvForest 26.06 and Treelite 4.7.0, ",
      "including lib/libtreelite_static.a.",
      call. = FALSE
    )
  }

  list(
    source = normalizePath(source, mustWork = TRUE),
    cuda_home = cuda_home,
    cuda_libdir = cuda_ml_source_cuda_libdir(cuda_home),
    cuml_prefix = cuml_prefix,
    architectures = architectures,
    cxx = compiler$path,
    cxx_version = compiler$version,
    cmake = cmake$path,
    cmake_version = cmake$version,
    ninja = ninja,
    ninja_version = ninja_version,
    nvcc = normalizePath(nvcc, mustWork = TRUE),
    dependencies = dependencies,
    lock = lock,
    artifact_lock = cuda_ml_hash_file(artifact_lock_path)
  )
}

cuda_ml_source_identity <- function(inputs) {
  files <- list.files(
    inputs$source,
    full.names = TRUE,
    recursive = TRUE,
    all.files = TRUE,
    no.. = TRUE
  )
  files <- files[!file.info(files)$isdir]
  relative <- substring(files, nchar(inputs$source) + 2L)
  hashes <- vapply(files, cuda_ml_hash_file, character(1))
  source_lock <- unname(digest::digest(
    paste(relative, hashes, collapse = "\n"),
    algo = "sha256",
    serialize = FALSE
  ))
  values <- c(
    as.character(utils::packageVersion("cuda.ml")),
    cuda_ml_r_version(),
    source_lock,
    inputs$cuda_home,
    inputs$cuml_prefix,
    inputs$architectures,
    inputs$cxx,
    inputs$cxx_version,
    inputs$cmake,
    inputs$cmake_version,
    inputs$ninja,
    inputs$ninja_version,
    inputs$dependencies,
    inputs$artifact_lock,
    unname(inputs$lock[c("CUDA-Toolkit", "RAPIDS", "nvForest", "Treelite")])
  )
  list(
    source_hash = unname(digest::digest(
      paste(values, collapse = "\n"),
      algo = "sha256",
      serialize = FALSE
    )),
    source_lock = source_lock,
    inputs = inputs
  )
}

cuda_ml_source_backend_path <- function(source_hash) {
  stopifnot(
    is.character(source_hash),
    length(source_hash) == 1L,
    grepl("^[[:xdigit:]]{64}$", source_hash)
  )
  file.path(
    cuda_ml_cache_dir(),
    "source-backends-v1",
    unname(.cuda_ml_state$metadata[["Platform"]]),
    paste0("r-", cuda_ml_r_version()),
    source_hash
  )
}

cuda_ml_source_backend_complete <- function(path, source_hash, audit = FALSE) {
  backend <- file.path(path, "lib", paste0("cuda.ml", .Platform$dynlib.ext))
  inventory <- file.path(path, "inventory.tsv")
  metadata <- cuda_ml_complete_metadata(path)
  fields <- c(
    "Schema",
    "Source-SHA256",
    "Backend-SHA256",
    "Package-Version",
    "R-Version",
    "Platform",
    "CUDA",
    "RAPIDS",
    "nvForest",
    "Treelite",
    "Architectures",
    "Inventory-SHA256"
  )
  if (
    !file.exists(backend) ||
      !file.exists(inventory) ||
      is.null(metadata) ||
      !all(fields %in% names(metadata)) ||
      !identical(unname(metadata[["Schema"]]), "1") ||
      !identical(unname(metadata[["Source-SHA256"]]), source_hash) ||
      !identical(
        unname(metadata[["Package-Version"]]),
        as.character(utils::packageVersion("cuda.ml"))
      ) ||
      !identical(unname(metadata[["R-Version"]]), cuda_ml_r_version()) ||
      !identical(unname(metadata[["Platform"]]), cuda_ml_platform()) ||
      !identical(
        unname(metadata[["CUDA"]]),
        unname(.cuda_ml_state$metadata[["CUDA"]])
      ) ||
      !identical(
        unname(metadata[["RAPIDS"]]),
        unname(.cuda_ml_state$metadata[["RAPIDS"]])
      ) ||
      !identical(
        unname(metadata[["nvForest"]]),
        unname(.cuda_ml_state$metadata[["nvForest"]])
      ) ||
      !identical(
        unname(metadata[["Treelite"]]),
        unname(.cuda_ml_state$metadata[["Treelite"]])
      ) ||
      !identical(
        unname(metadata[["Inventory-SHA256"]]),
        cuda_ml_hash_file(inventory)
      ) ||
      !cuda_ml_inventory_complete(path, audit = audit)
  ) {
    return(FALSE)
  }
  !audit || identical(
    cuda_ml_hash_file(backend),
    unname(metadata[["Backend-SHA256"]])
  )
}

cuda_ml_backend_selection_path <- function() {
  file.path(
    cuda_ml_cache_dir(),
    "backend-selection-v1",
    unname(.cuda_ml_state$metadata[["Platform"]]),
    paste0("r-", cuda_ml_r_version(), ".dcf")
  )
}

cuda_ml_backend_selection <- function() {
  path <- cuda_ml_backend_selection_path()
  if (!file.exists(path)) {
    return(list(backend = "download", source_hash = NULL))
  }
  metadata <- tryCatch(read.dcf(path), error = function(e) NULL)
  required <- c("Schema", "Backend")
  if (
    is.null(metadata) ||
      nrow(metadata) != 1L ||
      !all(required %in% colnames(metadata))
  ) {
    stop("The cuda.ml backend selection is invalid.", call. = FALSE)
  }
  metadata <- metadata[1L, , drop = TRUE]
  backend <- unname(metadata[["Backend"]])
  valid <- identical(unname(metadata[["Schema"]]), "1") &&
    backend %in% c("download", "source")
  source_hash <- NULL
  if (identical(backend, "source")) {
    valid <- valid && "Source-SHA256" %in% names(metadata)
    if (valid) {
      source_hash <- unname(metadata[["Source-SHA256"]])
      valid <- length(source_hash) == 1L &&
        grepl("^[[:xdigit:]]{64}$", source_hash)
    }
  }
  if (!valid) {
    stop("The cuda.ml backend selection is invalid.", call. = FALSE)
  }
  list(backend = backend, source_hash = source_hash)
}

cuda_ml_write_backend_selection <- function(backend, source_hash = NULL) {
  stopifnot(
    is.character(backend),
    length(backend) == 1L,
    backend %in% c("download", "source"),
    identical(backend, "download") ||
      (
        is.character(source_hash) &&
          length(source_hash) == 1L &&
          grepl("^[[:xdigit:]]{64}$", source_hash)
      )
  )
  path <- cuda_ml_backend_selection_path()
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile("backend-selection-", tmpdir = dirname(path))
  fields <- c(Schema = "1", Backend = backend)
  if (identical(backend, "source")) {
    fields <- c(fields, `Source-SHA256` = source_hash)
  }
  write.dcf(
    matrix(
      unname(fields),
      nrow = 1L,
      dimnames = list(NULL, names(fields))
    ),
    file = staging
  )
  if (!file.rename(staging, path)) {
    unlink(staging, force = TRUE)
    stop("Unable to select the installed cuda.ml backend.", call. = FALSE)
  }
  invisible(path)
}

cuda_ml_configure_source_tree <- function(inputs, destination) {
  dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  files <- list.files(
    inputs$source,
    full.names = TRUE,
    all.files = TRUE,
    no.. = TRUE
  )
  stopifnot(length(files) > 0L, !any(file.info(files)$isdir))
  if (!all(file.copy(files, destination, copy.mode = TRUE))) {
    stop("Unable to stage the cuda.ml backend source.", call. = FALSE)
  }

  template <- file.path(destination, "CMakeLists.txt.in")
  contents <- readLines(template, warn = FALSE)
  contents <- gsub(
    "@R_INCLUDE_DIR@",
    normalizePath(R.home("include"), mustWork = TRUE),
    contents,
    fixed = TRUE
  )
  contents <- gsub(
    "@RCPP_INCLUDE_DIR@",
    normalizePath(system.file("include", package = "Rcpp"), mustWork = TRUE),
    contents,
    fixed = TRUE
  )
  writeLines(contents, file.path(destination, "CMakeLists.txt"))
  invisible(destination)
}

cuda_ml_install_source_backend <- function(identity, final_path) {
  inputs <- identity$inputs
  lock <- cuda_ml_acquire_lock(paste0("source-backend-", identity$source_hash))
  on.exit(filelock::unlock(lock), add = TRUE)
  if (cuda_ml_source_backend_complete(final_path, identity$source_hash)) {
    return(final_path)
  }

  dir.create(dirname(final_path), recursive = TRUE, showWarnings = FALSE)
  workspace <- tempfile(
    paste0(basename(final_path), "-staging-"),
    tmpdir = dirname(final_path)
  )
  dir.create(workspace)
  on.exit(unlink(workspace, recursive = TRUE, force = TRUE), add = TRUE)
  source <- cuda_ml_configure_source_tree(
    inputs,
    file.path(workspace, "source")
  )
  build <- file.path(workspace, "build")
  cache <- file.path(workspace, "cache")
  libdir <- file.path(cache, "lib")
  dir.create(build)
  dir.create(libdir, recursive = TRUE)

  rpath <- paste(
    unique(c(
      "$ORIGIN",
      file.path(inputs$cuml_prefix, "lib"),
      inputs$cuda_libdir
    )),
    collapse = ";"
  )
  generator_args <- if (is.null(inputs$ninja)) {
    character()
  } else {
    c("-G", "Ninja", paste0("-DCMAKE_MAKE_PROGRAM=", inputs$ninja))
  }
  configure_args <- c(
    generator_args,
    "-S",
    source,
    "-B",
    build,
    "-DCMAKE_BUILD_TYPE=Release",
    paste0("-DCMAKE_CUDA_ARCHITECTURES=", inputs$architectures),
    paste0("-DCUML_INCLUDE_DIR=", file.path(inputs$cuml_prefix, "include")),
    paste0("-DCUML_LIB_DIR=", file.path(inputs$cuml_prefix, "lib")),
    paste0("-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=", build),
    paste0("-DCMAKE_CUDA_COMPILER=", inputs$nvcc),
    paste0("-DCMAKE_CUDA_HOST_COMPILER=", inputs$cxx),
    paste0("-DCMAKE_CXX_COMPILER=", inputs$cxx),
    paste0("-DCMAKE_PREFIX_PATH=", inputs$cuml_prefix),
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=TRUE",
    paste0("-DCUDA_ML_INSTALL_RPATH=", rpath)
  )
  status <- system2(inputs$cmake, shQuote(configure_args))
  if (!identical(status, 0L)) {
    stop("Failed to configure the cuda.ml source backend.", call. = FALSE)
  }
  status <- system2(
    inputs$cmake,
    shQuote(c("--build", build, "--target", "cuda.ml", "--parallel", "2"))
  )
  backend <- file.path(build, paste0("cuda.ml", .Platform$dynlib.ext))
  if (
    !identical(status, 0L) ||
      !file.exists(backend) ||
      !cuda_ml_is_elf(backend)
  ) {
    stop("Failed to compile the cuda.ml source backend.", call. = FALSE)
  }
  if (!file.copy(backend, file.path(libdir, basename(backend)), copy.mode = TRUE)) {
    stop("Unable to stage the compiled cuda.ml backend.", call. = FALSE)
  }

  backend <- file.path(libdir, basename(backend))
  dll <- cuda_ml_load_backend(cache)
  dyn.unload(dll[["path"]])
  inventory <- cuda_ml_write_inventory(cache)
  fields <- c(
    Schema = "1",
    `Source-SHA256` = identity$source_hash,
    `Backend-SHA256` = cuda_ml_hash_file(backend),
    `Package-Version` = as.character(utils::packageVersion("cuda.ml")),
    `R-Version` = cuda_ml_r_version(),
    Platform = cuda_ml_platform(),
    CUDA = unname(.cuda_ml_state$metadata[["CUDA"]]),
    RAPIDS = unname(.cuda_ml_state$metadata[["RAPIDS"]]),
    nvForest = unname(.cuda_ml_state$metadata[["nvForest"]]),
    Treelite = unname(.cuda_ml_state$metadata[["Treelite"]]),
    Architectures = inputs$architectures,
    `CUDA-Home` = inputs$cuda_home,
    `CUML-Prefix` = inputs$cuml_prefix,
    CXX = inputs$cxx,
    CMake = inputs$cmake,
    Dependencies = inputs$dependencies,
    `Inventory-SHA256` = cuda_ml_hash_file(inventory)
  )
  if (!is.null(inputs$ninja)) {
    fields <- c(fields, Ninja = inputs$ninja)
  }
  write.dcf(
    matrix(
      unname(fields),
      nrow = 1L,
      dimnames = list(NULL, names(fields))
    ),
    file = file.path(cache, ".complete")
  )
  if (!cuda_ml_source_backend_complete(cache, identity$source_hash, audit = TRUE)) {
    stop("The compiled cuda.ml source backend failed its audit.", call. = FALSE)
  }

  if (dir.exists(final_path)) {
    unlink(final_path, recursive = TRUE, force = TRUE)
  }
  if (!file.rename(cache, final_path)) {
    stop("Unable to publish the compiled cuda.ml source backend.", call. = FALSE)
  }
  final_path
}

cuda_ml_prepare_source_backend <- function(inputs) {
  identity <- cuda_ml_source_identity(inputs)
  path <- cuda_ml_source_backend_path(identity$source_hash)
  if (!cuda_ml_source_backend_complete(path, identity$source_hash)) {
    cuda_ml_install_source_backend(identity, path)
  }
  list(identity = identity, path = path)
}
