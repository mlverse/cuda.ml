.cuda_ml_state <- new.env(parent = emptyenv())

cuda_ml_backend_metadata <- function(pkgname = "cuda.ml") {
  path <- system.file("cuda-ml-backend.dcf", package = pkgname)
  if (!nzchar(path)) {
    stop(
      "The cuda.ml backend manifest is missing from this installation.",
      call. = FALSE
    )
  }

  metadata <- read.dcf(path)
  required <- c(
    "Schema",
    "Backend",
    "Build-Mode",
    "CUDA",
    "RAPIDS",
    "nvForest",
    "Treelite",
    "Platform",
    "Minimum-glibc",
    "Minimum-Driver",
    "Architectures"
  )
  if (nrow(metadata) != 1L || !all(required %in% colnames(metadata))) {
    stop("The cuda.ml backend manifest is invalid.", call. = FALSE)
  }
  metadata <- metadata[1L, , drop = TRUE]
  if (
    !identical(unname(metadata[["Schema"]]), "3") ||
      !identical(unname(metadata[["Backend"]]), "download") ||
      !identical(unname(metadata[["Build-Mode"]]), "release")
  ) {
    stop("The cuda.ml backend manifest is invalid.", call. = FALSE)
  }
  metadata
}

#' Report native-backend metadata
#'
#' @return A named list describing the selected backend and whether its exact
#'   cache is complete. This function performs read-only cache and inventory
#'   checks. It does not create or modify the cache, access the network, inspect
#'   an NVIDIA GPU or driver, or load the native backend.
#' @export
cuda_ml_backend_info <- function() {
  metadata <- .cuda_ml_state$metadata
  value <- function(name) {
    result <- unname(metadata[[name]])
    if (nzchar(result)) result else NA_character_
  }
  selection <- cuda_ml_backend_selection()
  backend <- value("Backend")
  build_mode <- value("Build-Mode")
  architectures <- value("Architectures")
  runtime_installed <- FALSE
  runtime_path <- NA_character_
  if (identical(selection$backend, "source")) {
    backend <- "source"
    build_mode <- "local"
    backend_available <- nzchar(system.file("backend-src", package = "cuda.ml"))
    architectures <- character()
    if (cuda_ml_supported_platform()) {
      candidate_backend <- cuda_ml_source_backend_path(selection$source_hash)
      runtime_installed <- cuda_ml_source_backend_complete(
        candidate_backend,
        selection$source_hash
      )
      if (runtime_installed) {
        runtime_path <- candidate_backend
        source_metadata <- cuda_ml_complete_metadata(candidate_backend)
        architectures <- strsplit(
          unname(source_metadata[["Architectures"]]),
          ";",
          fixed = TRUE
        )[[1L]]
      }
    }
  } else {
    release <- cuda_ml_backend_release(required = FALSE)
    backend_available <- !is.null(release)
    if (backend_available && cuda_ml_supported_platform()) {
      runtime_identity <- cuda_ml_runtime_identity()
      candidate_runtime <- cuda_ml_runtime_path(runtime_identity)
      backend_identity <- cuda_ml_backend_identity(release)
      candidate_backend <- cuda_ml_backend_cache_path(
        runtime_identity,
        backend_identity
      )
      runtime_installed <- cuda_ml_runtime_complete(
        candidate_runtime,
        runtime_identity
      ) &&
        cuda_ml_backend_complete(
          candidate_backend,
          runtime_identity,
          backend_identity
        )
      if (runtime_installed) {
        runtime_path <- candidate_runtime
      }
    }
  }
  if (length(architectures) == 1L) {
    if (is.na(architectures)) {
      architectures <- character()
    } else {
      architectures <- strsplit(architectures, ";", fixed = TRUE)[[1L]]
    }
  }

  list(
    package_version = as.character(utils::packageVersion("cuda.ml")),
    backend = backend,
    build_mode = build_mode,
    backend_available = backend_available,
    r_version = cuda_ml_r_version(),
    cuda_version = value("CUDA"),
    rapids_version = value("RAPIDS"),
    nvforest_version = value("nvForest"),
    treelite_version = value("Treelite"),
    platform = value("Platform"),
    minimum_glibc = value("Minimum-glibc"),
    minimum_driver = {
      driver <- value("Minimum-Driver")
      if (is.na(driver)) NA_integer_ else as.integer(driver)
    },
    architectures = architectures,
    runtime_installed = runtime_installed,
    runtime_path = runtime_path,
    backend_loaded = !is.null(.cuda_ml_state$dll)
  )
}

cuda_ml_native_symbols <- function(pkgname = "cuda.ml") {
  path <- system.file("native-symbols.txt", package = pkgname)
  if (!nzchar(path)) {
    stop(
      "The cuda.ml native-symbol manifest is missing from this installation.",
      call. = FALSE
    )
  }

  symbols <- tryCatch(
    utils::read.delim(
      path,
      stringsAsFactors = FALSE,
      check.names = FALSE,
      colClasses = c("character", "integer")
    ),
    error = function(e) NULL
  )
  valid <- !is.null(symbols) &&
    identical(names(symbols), c("symbol", "arity")) &&
    nrow(symbols) > 0L &&
    all(grepl("^_cuda_ml_[[:alnum:]_]+$", symbols$symbol)) &&
    !anyDuplicated(symbols$symbol) &&
    all(symbols$arity >= 0L)
  if (!valid) {
    stop("The cuda.ml native-symbol manifest is invalid.", call. = FALSE)
  }
  symbols
}

cuda_ml_cache_dir <- function() {
  cache <- Sys.getenv("CUDA_ML_CACHE_DIR", unset = "")
  if (!nzchar(cache)) {
    cache <- tools::R_user_dir("cuda.ml", "cache")
  }
  normalizePath(path.expand(cache), mustWork = FALSE)
}

cuda_ml_r_version <- function() {
  paste(
    R.version$major,
    strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][[1L]],
    sep = "."
  )
}

cuda_ml_glibc_version <- function() {
  output <- suppressWarnings(tryCatch(
    system2("getconf", "GNU_LIBC_VERSION", stdout = TRUE, stderr = FALSE),
    error = function(e) character()
  ))
  match <- regexec("^glibc ([0-9]+[.][0-9]+)$", output)
  values <- regmatches(output, match)
  if (length(values) != 1L || length(values[[1L]]) != 2L) {
    return(NA_character_)
  }
  values[[1L]][[2L]]
}

cuda_ml_supported_platform <- function() {
  sysinfo <- Sys.info()
  if (
    !identical(unname(sysinfo[["sysname"]]), "Linux") ||
      !unname(sysinfo[["machine"]]) %in% c("x86_64", "amd64")
  ) {
    return(FALSE)
  }
  glibc <- cuda_ml_glibc_version()
  !is.na(glibc) &&
    base::package_version(glibc) >=
      base::package_version(.cuda_ml_state$metadata[["Minimum-glibc"]])
}

cuda_ml_platform <- function() {
  if (!cuda_ml_supported_platform()) {
    stop(
      "The cuda.ml native backend requires Linux x86_64 with glibc ",
      .cuda_ml_state$metadata[["Minimum-glibc"]],
      " or newer.",
      call. = FALSE
    )
  }
  unname(.cuda_ml_state$metadata[["Platform"]])
}

cuda_ml_runtime_manifest <- function() {
  platform <- cuda_ml_platform()
  manifest_path <- system.file(
    "runtime",
    paste0(platform, ".tsv"),
    package = "cuda.ml"
  )
  metadata_path <- system.file(
    "runtime",
    "runtime.dcf",
    package = "cuda.ml"
  )
  if (!nzchar(manifest_path) || !nzchar(metadata_path)) {
    stop(
      "The cuda.ml runtime lock is missing from this installation.",
      call. = FALSE
    )
  }

  manifest <- utils::read.delim(
    manifest_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  required <- c(
    "component",
    "package",
    "version",
    "filename",
    "url",
    "sha256",
    "size",
    "extract_regex"
  )
  stopifnot(identical(names(manifest), required), nrow(manifest) > 0L)

  metadata <- read.dcf(metadata_path)
  required_metadata <- c(
    "Schema",
    "CUDA",
    "CUDA-Toolkit",
    "RAPIDS",
    "nvForest",
    "Platform",
    "Minimum-Driver",
    "NVRTC-Needed-Old",
    "NVRTC-Needed-New"
  )
  stopifnot(
    nrow(metadata) == 1L,
    all(required_metadata %in% colnames(metadata))
  )
  metadata <- metadata[1L, , drop = TRUE]
  nvrtc_needed_old <- unname(metadata[["NVRTC-Needed-Old"]])
  nvrtc_needed_new <- unname(metadata[["NVRTC-Needed-New"]])
  stopifnot(
    identical(unname(metadata[["Schema"]]), "2"),
    identical(unname(metadata[["Platform"]]), platform),
    nzchar(nvrtc_needed_old),
    nzchar(nvrtc_needed_new),
    identical(basename(nvrtc_needed_old), nvrtc_needed_old),
    identical(basename(nvrtc_needed_new), nvrtc_needed_new),
    !identical(nvrtc_needed_old, nvrtc_needed_new)
  )
  list(
    path = manifest_path,
    metadata_path = metadata_path,
    data = manifest,
    metadata = metadata
  )
}

cuda_ml_backend_catalog <- function() {
  platform <- unname(.cuda_ml_state$metadata[["Platform"]])
  path <- system.file(
    "backends",
    paste0(platform, ".tsv"),
    package = "cuda.ml"
  )
  if (!nzchar(path)) {
    stop("The cuda.ml backend catalog is missing.", call. = FALSE)
  }
  catalog <- tryCatch(
    utils::read.delim(
      path,
      stringsAsFactors = FALSE,
      check.names = FALSE,
      colClasses = c(
        "character",
        "character",
        "character",
        "numeric",
        "character",
        "character"
      )
    ),
    error = function(e) NULL
  )
  required <- c(
    "r_version",
    "filename",
    "url",
    "size",
    "sha256",
    "backend_sha256"
  )
  valid <- !is.null(catalog) &&
    identical(names(catalog), required) &&
    !anyDuplicated(catalog$r_version) &&
    all(grepl("^[0-9]+[.][0-9]+$", catalog$r_version)) &&
    all(nzchar(catalog$filename)) &&
    all(basename(catalog$filename) == catalog$filename) &&
    all(endsWith(catalog$filename, ".tar.gz")) &&
    all(startsWith(catalog$url, "https://")) &&
    all(endsWith(catalog$url, paste0("/", catalog$filename))) &&
    all(catalog$size > 0) &&
    all(grepl("^[[:xdigit:]]{64}$", catalog$sha256)) &&
    all(grepl("^[[:xdigit:]]{64}$", catalog$backend_sha256))
  if (!valid) {
    stop("The cuda.ml backend catalog is invalid.", call. = FALSE)
  }
  catalog
}

cuda_ml_backend_release <- function(required = TRUE) {
  catalog <- cuda_ml_backend_catalog()
  release <- catalog[catalog$r_version == cuda_ml_r_version(), , drop = FALSE]
  if (!nrow(release)) {
    if (!required) {
      return(NULL)
    }
    stop(
      "No prebuilt cuda.ml backend is published for R ",
      cuda_ml_r_version(),
      " on ",
      .cuda_ml_state$metadata[["Platform"]],
      ".",
      call. = FALSE
    )
  }
  stopifnot(nrow(release) == 1L)
  release
}

cuda_ml_hash_file <- function(path) {
  unname(digest::digest(file = path, algo = "sha256", serialize = FALSE))
}

cuda_ml_runtime_identity <- function() {
  manifest <- cuda_ml_runtime_manifest()
  if (
    !identical(
      unname(.cuda_ml_state$metadata[["CUDA"]]),
      unname(manifest$metadata[["CUDA-Toolkit"]])
    ) ||
      !identical(
        unname(.cuda_ml_state$metadata[["RAPIDS"]]),
        unname(manifest$metadata[["RAPIDS"]])
      ) ||
      !identical(
        unname(.cuda_ml_state$metadata[["nvForest"]]),
        unname(manifest$metadata[["nvForest"]])
      )
  ) {
    stop(
      "The cuda.ml backend lock does not match its managed runtime lock.",
      call. = FALSE
    )
  }

  lock_hashes <- vapply(
    c(manifest$path, manifest$metadata_path),
    cuda_ml_hash_file,
    character(1)
  )
  list(
    manifest = manifest,
    runtime_hash = unname(digest::digest(
      paste(lock_hashes, collapse = ":"),
      algo = "sha256",
      serialize = FALSE
    ))
  )
}

cuda_ml_backend_identity <- function(release = cuda_ml_backend_release()) {
  list(
    release = release,
    backend_hash = unname(release[["sha256"]]),
    backend_dso_hash = unname(release[["backend_sha256"]]),
    r_version = cuda_ml_r_version()
  )
}

cuda_ml_backend_url <- function(backend_identity) {
  mirror <- Sys.getenv("CUDA_ML_BACKEND_MIRROR", unset = "")
  if (!nzchar(mirror)) {
    return(unname(backend_identity$release[["url"]]))
  }
  if (
    !startsWith(mirror, "https://") &&
      !startsWith(mirror, "file://")
  ) {
    stop(
      "CUDA_ML_BACKEND_MIRROR must be an https:// or file:// URL.",
      call. = FALSE
    )
  }
  paste0(
    sub("/+$", "", mirror),
    "/",
    backend_identity$release[["filename"]]
  )
}

cuda_ml_runtime_path <- function(identity) {
  file.path(
    cuda_ml_cache_dir(),
    "runtime-v3",
    cuda_ml_platform(),
    identity$runtime_hash
  )
}

cuda_ml_backend_asset_path <- function(backend_identity) {
  file.path(
    cuda_ml_cache_dir(),
    "backend-assets-v1",
    cuda_ml_platform(),
    paste0("r-", backend_identity$r_version),
    backend_identity$backend_hash
  )
}

cuda_ml_backend_cache_path <- function(runtime_identity, backend_identity) {
  file.path(
    cuda_ml_cache_dir(),
    "backends-v3",
    cuda_ml_platform(),
    paste0("r-", backend_identity$r_version),
    backend_identity$backend_hash,
    runtime_identity$runtime_hash
  )
}

cuda_ml_read_inventory <- function(path) {
  tryCatch(
    utils::read.delim(
      file.path(path, "inventory.tsv"),
      stringsAsFactors = FALSE,
      check.names = FALSE,
      colClasses = c("character", "numeric", "character", "character"),
      na.strings = character()
    ),
    error = function(e) NULL
  )
}

cuda_ml_inventory_complete <- function(path, audit = FALSE) {
  files <- cuda_ml_read_inventory(path)
  valid_library <- startsWith(files$file, "lib/") &
    dirname(files$file) == "lib" &
    basename(files$file) == sub("^lib/", "", files$file)
  valid_binary <- files$file == "bin/patchelf"
  if (
    is.null(files) ||
      !identical(names(files), c("file", "size", "sha256", "link")) ||
      !nrow(files) ||
      any(!(valid_library | valid_binary))
  ) {
    return(FALSE)
  }

  paths <- file.path(path, files$file)
  actual <- file.path(
    "lib",
    list.files(
      file.path(path, "lib"),
      full.names = FALSE,
      all.files = TRUE,
      no.. = TRUE
    )
  )
  if (file.exists(file.path(path, "bin", "patchelf"))) {
    actual <- c(actual, "bin/patchelf")
  }
  if (
    !identical(sort(actual), sort(files$file)) ||
      any(!file.exists(paths)) ||
      !identical(
        as.numeric(file.info(paths)[["size"]]),
        as.numeric(files$size)
      ) ||
      !identical(
        unname(vapply(paths, Sys.readlink, character(1))),
        unname(files$link)
      )
  ) {
    return(FALSE)
  }

  if (!audit) {
    return(TRUE)
  }
  regular <- !nzchar(files$link)
  identical(
    unname(vapply(paths[regular], cuda_ml_hash_file, character(1))),
    unname(files$sha256[regular])
  )
}

cuda_ml_complete_metadata <- function(path) {
  marker <- file.path(path, ".complete")
  inventory <- file.path(path, "inventory.tsv")
  if (!file.exists(marker) || !file.exists(inventory)) {
    return(NULL)
  }
  metadata <- tryCatch(read.dcf(marker), error = function(e) NULL)
  if (is.null(metadata) || nrow(metadata) != 1L) {
    return(NULL)
  }
  metadata[1L, , drop = TRUE]
}

cuda_ml_runtime_complete <- function(path, identity, audit = FALSE) {
  marker <- file.path(path, ".complete")
  inventory <- file.path(path, "inventory.tsv")
  patchelf <- file.path(path, "bin", "patchelf")
  if (
    !file.exists(marker) ||
      !file.exists(inventory) ||
      !file.exists(patchelf)
  ) {
    return(FALSE)
  }

  metadata <- cuda_ml_complete_metadata(path)
  fields <- c("Schema", "Runtime-SHA256", "Inventory-SHA256")
  if (is.null(metadata) || !all(fields %in% names(metadata))) {
    return(FALSE)
  }
  if (
    !identical(unname(metadata[["Schema"]]), "2") ||
      !identical(unname(metadata[["Runtime-SHA256"]]), identity$runtime_hash) ||
      !identical(
        unname(metadata[["Inventory-SHA256"]]),
        cuda_ml_hash_file(inventory)
      )
  ) {
    return(FALSE)
  }

  cuda_ml_inventory_complete(path, audit = audit)
}

cuda_ml_backend_complete <- function(
  path,
  runtime_identity,
  backend_identity,
  audit = FALSE
) {
  backend <- file.path(path, "lib", paste0("cuda.ml", .Platform$dynlib.ext))
  if (!file.exists(backend)) {
    return(FALSE)
  }

  metadata <- cuda_ml_complete_metadata(path)
  fields <- c(
    "Schema",
    "Runtime-SHA256",
    "Asset-SHA256",
    "Backend-SHA256",
    "Inventory-SHA256"
  )
  if (
    is.null(metadata) ||
      !all(fields %in% names(metadata)) ||
      !identical(unname(metadata[["Schema"]]), "3") ||
      !identical(
        unname(metadata[["Runtime-SHA256"]]),
        runtime_identity$runtime_hash
      ) ||
      !identical(
        unname(metadata[["Asset-SHA256"]]),
        backend_identity$backend_hash
      ) ||
      !identical(
        unname(metadata[["Backend-SHA256"]]),
        backend_identity$backend_dso_hash
      ) ||
      !identical(
        unname(metadata[["Inventory-SHA256"]]),
        cuda_ml_hash_file(file.path(path, "inventory.tsv"))
      )
  ) {
    return(FALSE)
  }
  cuda_ml_inventory_complete(path, audit = audit)
}

cuda_ml_backend_asset_complete <- function(
  path,
  backend_identity,
  audit = FALSE
) {
  backend <- file.path(path, "lib", paste0("cuda.ml", .Platform$dynlib.ext))
  metadata_path <- file.path(path, "lib", "backend.dcf")
  if (!file.exists(backend) || !file.exists(metadata_path)) {
    return(FALSE)
  }

  metadata <- cuda_ml_complete_metadata(path)
  fields <- c(
    "Schema",
    "Asset-SHA256",
    "Backend-SHA256",
    "Inventory-SHA256"
  )
  if (
    is.null(metadata) ||
      !all(fields %in% names(metadata)) ||
      !identical(unname(metadata[["Schema"]]), "3") ||
      !identical(
        unname(metadata[["Asset-SHA256"]]),
        backend_identity$backend_hash
      ) ||
      !identical(
        unname(metadata[["Backend-SHA256"]]),
        backend_identity$backend_dso_hash
      ) ||
      !identical(
        unname(metadata[["Inventory-SHA256"]]),
        cuda_ml_hash_file(file.path(path, "inventory.tsv"))
      )
  ) {
    return(FALSE)
  }
  cuda_ml_inventory_complete(path, audit = audit)
}

cuda_ml_download <- function(component, url, destination, size, sha256) {
  message(
    "Downloading ",
    component,
    " (",
    format(round(as.numeric(size) / 1024^2, 1), trim = TRUE),
    " MiB)"
  )
  old_timeout <- getOption("timeout")
  on.exit(options(timeout = old_timeout), add = TRUE)
  options(timeout = max(600, old_timeout))

  attempts <- 3L
  for (attempt in seq_len(attempts)) {
    unlink(destination, force = TRUE)
    if (attempt > 1L) {
      message(
        "Retrying ",
        component,
        " (attempt ",
        attempt,
        " of ",
        attempts,
        ")"
      )
    }
    status <- tryCatch(
      utils::download.file(url, destination, mode = "wb", quiet = FALSE),
      error = function(e) e
    )
    downloaded <- !inherits(status, "error") && identical(status, 0L)
    if (downloaded) {
      actual_size <- file.info(destination)[["size"]]
      size_matches <- identical(
        as.numeric(actual_size),
        as.numeric(size)
      )
      if (size_matches) {
        actual_hash <- cuda_ml_hash_file(destination)
        if (identical(actual_hash, sha256)) {
          return(invisible(destination))
        }
      }
    }
  }
  unlink(destination, force = TRUE)
  stop(
    "Failed to download and verify component '",
    component,
    "' after ",
    attempts,
    " attempts.",
    call. = FALSE
  )
}

cuda_ml_backend_archive_metadata <- function(path, backend_identity) {
  metadata <- tryCatch(read.dcf(path), error = function(e) NULL)
  required <- c(
    "Schema",
    "Package",
    "Package-Version",
    "R-Version",
    "Platform",
    "CUDA",
    "RAPIDS",
    "nvForest",
    "Treelite",
    "Backend-SHA256",
    "Source-Commit"
  )
  valid <- !is.null(metadata) &&
    nrow(metadata) == 1L &&
    identical(colnames(metadata), required)
  if (!valid) {
    stop("The downloaded cuda.ml backend metadata is invalid.", call. = FALSE)
  }
  metadata <- metadata[1L, , drop = TRUE]
  expected <- c(
    Schema = "1",
    Package = "cuda.ml",
    `Package-Version` = as.character(utils::packageVersion("cuda.ml")),
    `R-Version` = backend_identity$r_version,
    Platform = unname(.cuda_ml_state$metadata[["Platform"]]),
    CUDA = unname(.cuda_ml_state$metadata[["CUDA"]]),
    RAPIDS = unname(.cuda_ml_state$metadata[["RAPIDS"]]),
    nvForest = unname(.cuda_ml_state$metadata[["nvForest"]]),
    Treelite = unname(.cuda_ml_state$metadata[["Treelite"]]),
    `Backend-SHA256` = backend_identity$backend_dso_hash
  )
  if (
    !identical(unname(metadata[names(expected)]), unname(expected)) ||
      !grepl("^[[:xdigit:]]{40}$", metadata[["Source-Commit"]])
  ) {
    stop(
      "The downloaded cuda.ml backend does not match this R package.",
      call. = FALSE
    )
  }
  metadata
}

cuda_ml_install_backend_asset <- function(backend_identity, final_path) {
  lock <- cuda_ml_acquire_lock(paste0(
    "backend-asset-",
    backend_identity$backend_hash
  ))
  on.exit(filelock::unlock(lock), add = TRUE)
  if (cuda_ml_backend_asset_complete(final_path, backend_identity)) {
    return(final_path)
  }

  dir.create(dirname(final_path), recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(
    paste0(basename(final_path), "-staging-"),
    tmpdir = dirname(final_path)
  )
  dir.create(staging)
  on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)

  release <- backend_identity$release
  archive <- file.path(staging, release[["filename"]])
  cuda_ml_download(
    paste0("cuda.ml backend for R ", backend_identity$r_version),
    cuda_ml_backend_url(backend_identity),
    archive,
    release[["size"]],
    release[["sha256"]]
  )

  expected <- c("backend.dcf", paste0("cuda.ml", .Platform$dynlib.ext))
  files <- tryCatch(utils::untar(archive, list = TRUE), error = function(e) NULL)
  if (is.null(files) || !identical(sort(files), sort(expected))) {
    stop("The downloaded cuda.ml backend archive is invalid.", call. = FALSE)
  }
  extract <- file.path(staging, "extract")
  dir.create(extract)
  utils::untar(archive, files = expected, exdir = extract)
  metadata_path <- file.path(extract, "backend.dcf")
  backend <- file.path(extract, paste0("cuda.ml", .Platform$dynlib.ext))
  cuda_ml_backend_archive_metadata(metadata_path, backend_identity)
  if (
    !cuda_ml_is_elf(backend) ||
      !identical(cuda_ml_hash_file(backend), backend_identity$backend_dso_hash)
  ) {
    stop("The downloaded cuda.ml backend library is invalid.", call. = FALSE)
  }

  libdir <- file.path(staging, "lib")
  dir.create(libdir)
  copied <- file.copy(
    c(metadata_path, backend),
    file.path(libdir, basename(c(metadata_path, backend))),
    copy.mode = TRUE,
    copy.date = TRUE
  )
  if (!all(copied)) {
    stop("Unable to stage the downloaded cuda.ml backend.", call. = FALSE)
  }
  unlink(archive, force = TRUE)
  unlink(extract, recursive = TRUE, force = TRUE)
  inventory <- cuda_ml_write_inventory(staging)
  cuda_ml_write_backend_asset_complete(
    staging,
    backend_identity,
    cuda_ml_hash_file(inventory)
  )

  if (dir.exists(final_path)) {
    unlink(final_path, recursive = TRUE, force = TRUE)
  }
  if (!file.rename(staging, final_path)) {
    stop("Unable to publish the downloaded cuda.ml backend.", call. = FALSE)
  }
  final_path
}

cuda_ml_extract_component <- function(archive, row, directory) {
  listing <- utils::unzip(archive, list = TRUE)
  files <- listing$Name[
    grepl(row[["extract_regex"]], listing$Name, perl = TRUE)
  ]
  if (!length(files)) {
    stop(
      "Runtime component '",
      row[["component"]],
      "' did not contain any locked files.",
      call. = FALSE
    )
  }

  component_dir <- file.path(directory, row[["component"]])
  dir.create(component_dir, recursive = TRUE)
  utils::unzip(archive, files = files, exdir = component_dir)
  file.path(component_dir, files)
}

cuda_ml_is_elf <- function(path) {
  if (file.info(path)[["isdir"]]) {
    return(FALSE)
  }
  con <- file(path, open = "rb")
  on.exit(close(con))
  magic <- readBin(con, what = "raw", n = 4L)
  identical(magic, as.raw(c(0x7f, 0x45, 0x4c, 0x46)))
}

cuda_ml_copy_runtime_file <- function(source, libdir) {
  destination <- file.path(libdir, basename(source))
  if (file.exists(destination)) {
    if (!identical(cuda_ml_hash_file(source), cuda_ml_hash_file(destination))) {
      stop(
        "Runtime components contain conflicting files named '",
        basename(source),
        "'.",
        call. = FALSE
      )
    }
    return(invisible(destination))
  }
  if (!file.copy(source, destination, copy.mode = TRUE, copy.date = TRUE)) {
    stop(
      "Failed to stage runtime file '",
      basename(source),
      "'.",
      call. = FALSE
    )
  }
  invisible(destination)
}

cuda_ml_patchelf <- function(patchelf, file) {
  status <- system2(
    patchelf,
    c(
      "--force-rpath",
      "--set-rpath",
      shQuote("$ORIGIN"),
      shQuote(file)
    ),
    stdout = TRUE,
    stderr = TRUE
  )
  code <- attr(status, "status", exact = TRUE)
  if (!is.null(code) && !identical(code, 0L)) {
    stop("Failed to patch runtime file '", basename(file), "'.", call. = FALSE)
  }
  invisible(file)
}

cuda_ml_needed <- function(patchelf, file) {
  needed <- system2(
    patchelf,
    c("--print-needed", shQuote(file)),
    stdout = TRUE,
    stderr = TRUE
  )
  code <- attr(needed, "status", exact = TRUE)
  if (!is.null(code) && !identical(code, 0L)) {
    stop(
      "Unable to inspect runtime dependencies in '",
      basename(file),
      "'.",
      call. = FALSE
    )
  }
  unname(needed)
}

cuda_ml_replace_needed <- function(patchelf, file, old, new) {
  stopifnot(
    file.exists(file),
    identical(old, basename(old)),
    identical(new, basename(new)),
    !identical(old, new)
  )

  needed <- cuda_ml_needed(patchelf, file)
  if (sum(needed == old) != 1L || new %in% needed) {
    stop(
      "The locked NVRTC dependency in '",
      basename(file),
      "' does not match the runtime manifest.",
      call. = FALSE
    )
  }

  status <- system2(
    patchelf,
    c(
      "--replace-needed",
      shQuote(old),
      shQuote(new),
      shQuote(file)
    ),
    stdout = TRUE,
    stderr = TRUE
  )
  code <- attr(status, "status", exact = TRUE)
  if (!is.null(code) && !identical(code, 0L)) {
    stop(
      "Failed to replace the locked NVRTC dependency in '",
      basename(file),
      "'.",
      call. = FALSE
    )
  }

  needed <- cuda_ml_needed(patchelf, file)
  if (old %in% needed || sum(needed == new) != 1L) {
    stop(
      "Failed to validate the replacement NVRTC dependency in '",
      basename(file),
      "'.",
      call. = FALSE
    )
  }
  invisible(file)
}

cuda_ml_create_soname_links <- function(patchelf, files, libdir) {
  for (file in files) {
    soname <- system2(
      patchelf,
      c("--print-soname", shQuote(file)),
      stdout = TRUE,
      stderr = TRUE
    )
    code <- attr(soname, "status", exact = TRUE)
    if (!is.null(code) && !identical(code, 0L)) {
      stop("Unable to inspect runtime SONAMEs.", call. = FALSE)
    }
    if (!length(soname) || !nzchar(soname[[1L]])) {
      next
    }

    soname <- soname[[1L]]
    if (!identical(soname, basename(soname))) {
      stop("A managed runtime library has an invalid SONAME.", call. = FALSE)
    }
    link <- file.path(libdir, soname)
    if (!identical(basename(file), soname)) {
      existing_link <- Sys.readlink(link)
      if (file.exists(link) || nzchar(existing_link)) {
        same_file <- file.exists(link) &&
          identical(
            normalizePath(link, mustWork = TRUE),
            normalizePath(file, mustWork = TRUE)
          )
        if (!same_file) {
          stop(
            "Runtime components contain a conflicting SONAME '",
            soname,
            "'.",
            call. = FALSE
          )
        }
      } else if (!file.symlink(basename(file), link)) {
        stop(
          "Unable to preserve runtime SONAME '",
          soname,
          "'.",
          call. = FALSE
        )
      }
    }
  }
  invisible(TRUE)
}

cuda_ml_system_library <- function(library) {
  grepl(
    paste0(
      "^(lib(R|c|m|dl|pthread|rt|gcc_s|stdc\\+\\+|z|bz2|lzma|crypt|util|",
      "resolv|cuda|nvidia-ml)\\.so(\\..*)?|ld-linux-x86-64\\.so\\.2)$"
    ),
    library
  )
}

cuda_ml_validate_dependencies <- function(patchelf, libdir) {
  files <- list.files(libdir, full.names = TRUE)
  if (any(grepl("^libcuda[.]so([.].*)?$", basename(files)))) {
    stop(
      "The managed runtime must not bundle the NVIDIA driver.",
      call. = FALSE
    )
  }

  elf_files <- files[vapply(files, cuda_ml_is_elf, logical(1))]
  available <- basename(files)

  for (file in elf_files) {
    rpath <- system2(
      patchelf,
      c("--print-rpath", shQuote(file)),
      stdout = TRUE,
      stderr = TRUE
    )
    code <- attr(rpath, "status", exact = TRUE)
    if (
      (!is.null(code) && !identical(code, 0L)) ||
        !identical(rpath, "$ORIGIN")
    ) {
      stop(
        "The managed runtime contains an invalid RPATH in ",
        basename(file),
        ".",
        call. = FALSE
      )
    }

    needed <- cuda_ml_needed(patchelf, file)
    missing <- needed[
      !needed %in% available &
        !vapply(needed, cuda_ml_system_library, logical(1))
    ]
    if (length(missing)) {
      stop(
        "The managed runtime is missing shared libraries required by ",
        basename(file),
        ": ",
        paste(missing, collapse = ", "),
        call. = FALSE
      )
    }
  }
  invisible(TRUE)
}

cuda_ml_write_inventory <- function(path) {
  files <- list.files(
    file.path(path, "lib"),
    full.names = TRUE,
    all.files = TRUE,
    no.. = TRUE
  )
  patchelf <- file.path(path, "bin", "patchelf")
  if (file.exists(patchelf)) {
    files <- c(files, patchelf)
  }
  files <- sort(files)
  stopifnot(length(files) > 0L)
  links <- vapply(files, Sys.readlink, character(1))
  hashes <- rep.int("", length(files))
  regular <- !nzchar(links)
  hashes[regular] <- vapply(
    files[regular],
    cuda_ml_hash_file,
    character(1)
  )
  inventory <- data.frame(
    file = substring(files, nchar(path) + 2L),
    size = as.numeric(file.info(files)[["size"]]),
    sha256 = hashes,
    link = links,
    stringsAsFactors = FALSE
  )
  inventory_path <- file.path(path, "inventory.tsv")
  utils::write.table(
    inventory,
    file = inventory_path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )
  inventory_path
}

cuda_ml_write_runtime_complete <- function(path, identity, inventory_hash) {
  write.dcf(
    matrix(
      c(
        "2",
        identity$runtime_hash,
        inventory_hash
      ),
      nrow = 1L,
      dimnames = list(
        NULL,
        c(
          "Schema",
          "Runtime-SHA256",
          "Inventory-SHA256"
        )
      )
    ),
    file = file.path(path, ".complete")
  )
}

cuda_ml_write_backend_complete <- function(
  path,
  runtime_identity,
  backend_identity,
  inventory_hash
) {
  write.dcf(
    matrix(
      c(
        "3",
        runtime_identity$runtime_hash,
        backend_identity$backend_hash,
        backend_identity$backend_dso_hash,
        inventory_hash
      ),
      nrow = 1L,
      dimnames = list(
        NULL,
        c(
          "Schema",
          "Runtime-SHA256",
          "Asset-SHA256",
          "Backend-SHA256",
          "Inventory-SHA256"
        )
      )
    ),
    file = file.path(path, ".complete")
  )
}

cuda_ml_write_backend_asset_complete <- function(
  path,
  backend_identity,
  inventory_hash
) {
  write.dcf(
    matrix(
      c(
        "3",
        backend_identity$backend_hash,
        backend_identity$backend_dso_hash,
        inventory_hash
      ),
      nrow = 1L,
      dimnames = list(
        NULL,
        c(
          "Schema",
          "Asset-SHA256",
          "Backend-SHA256",
          "Inventory-SHA256"
        )
      )
    ),
    file = file.path(path, ".complete")
  )
}

cuda_ml_acquire_lock <- function(name) {
  cache <- cuda_ml_cache_dir()
  lock_dir <- file.path(cache, "locks")
  dir.create(lock_dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(lock_dir)) {
    stop(
      "Unable to create the cuda.ml cache directory '",
      cache,
      "'.",
      call. = FALSE
    )
  }

  lock <- filelock::lock(
    file.path(lock_dir, paste0(name, ".lock")),
    timeout = 60 * 60 * 1000
  )
  if (is.null(lock)) {
    stop(
      "Timed out waiting for another cuda.ml cache operation to finish.",
      call. = FALSE
    )
  }
  lock
}

cuda_ml_install_runtime <- function(identity, final_path) {
  lock <- cuda_ml_acquire_lock(paste0("runtime-", identity$runtime_hash))
  on.exit(filelock::unlock(lock), add = TRUE)
  if (cuda_ml_runtime_complete(final_path, identity)) {
    return(final_path)
  }

  dir.create(dirname(final_path), recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(
    paste0(basename(final_path), "-staging-"),
    tmpdir = dirname(final_path)
  )
  dir.create(staging)
  on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)

  download_dir <- file.path(staging, "downloads")
  extract_dir <- file.path(staging, "extract")
  libdir <- file.path(staging, "lib")
  bindir <- file.path(staging, "bin")
  dir.create(download_dir)
  dir.create(extract_dir)
  dir.create(libdir)
  dir.create(bindir)

  patchelf <- NULL
  manifest <- identity$manifest$data
  for (i in seq_len(nrow(manifest))) {
    row <- manifest[i, , drop = FALSE]
    archive <- file.path(download_dir, row[["filename"]])
    cuda_ml_download(
      row[["component"]],
      row[["url"]],
      archive,
      row[["size"]],
      row[["sha256"]]
    )
    extracted <- cuda_ml_extract_component(archive, row, extract_dir)
    if (identical(row[["component"]], "patchelf")) {
      stopifnot(length(extracted) == 1L)
      patchelf <- file.path(bindir, "patchelf")
      if (!file.copy(extracted, patchelf, copy.mode = TRUE)) {
        stop("Unable to stage the locked patchelf executable.", call. = FALSE)
      }
      Sys.chmod(patchelf, mode = "0755")
    } else {
      for (file in extracted) {
        if (cuda_ml_is_elf(file)) {
          cuda_ml_copy_runtime_file(file, libdir)
        }
      }
    }
  }
  if (is.null(patchelf) || !file.exists(patchelf)) {
    stop("The managed runtime lock does not contain patchelf.", call. = FALSE)
  }

  libcuml <- file.path(libdir, "libcuml.so")
  cuda_ml_replace_needed(
    patchelf,
    libcuml,
    unname(identity$manifest$metadata[["NVRTC-Needed-Old"]]),
    unname(identity$manifest$metadata[["NVRTC-Needed-New"]])
  )

  elf_files <- list.files(libdir, full.names = TRUE)
  elf_files <- elf_files[vapply(elf_files, cuda_ml_is_elf, logical(1))]
  for (file in elf_files) {
    cuda_ml_patchelf(patchelf, file)
  }
  cuda_ml_create_soname_links(patchelf, elf_files, libdir)
  cuda_ml_validate_dependencies(patchelf, libdir)

  unlink(download_dir, recursive = TRUE, force = TRUE)
  unlink(extract_dir, recursive = TRUE, force = TRUE)
  inventory <- cuda_ml_write_inventory(staging)
  cuda_ml_write_runtime_complete(
    staging,
    identity,
    cuda_ml_hash_file(inventory)
  )

  if (dir.exists(final_path)) {
    unlink(final_path, recursive = TRUE, force = TRUE)
  }
  if (!file.rename(staging, final_path)) {
    stop("Unable to publish the prepared cuda.ml runtime cache.", call. = FALSE)
  }
  final_path
}

cuda_ml_link_runtime <- function(runtime_path, backend_path) {
  sources <- sort(list.files(
    file.path(runtime_path, "lib"),
    full.names = TRUE,
    all.files = TRUE,
    no.. = TRUE
  ))
  stopifnot(length(sources) > 0L)
  libdir <- file.path(backend_path, "lib")
  dir.create(libdir, recursive = TRUE, showWarnings = FALSE)

  for (source in sources) {
    link <- file.path(libdir, basename(source))
    target <- normalizePath(source, mustWork = TRUE)
    if (!file.symlink(target, link)) {
      stop(
        "Unable to link the shared runtime library '",
        basename(source),
        "'.",
        call. = FALSE
      )
    }
  }
  invisible(libdir)
}

cuda_ml_install_backend <- function(
  runtime_identity,
  backend_identity,
  backend_asset_path,
  runtime_path,
  final_path
) {
  lock_name <- paste0(
    "backend-",
    backend_identity$backend_hash,
    "-",
    runtime_identity$runtime_hash
  )
  lock <- cuda_ml_acquire_lock(lock_name)
  on.exit(filelock::unlock(lock), add = TRUE)
  if (
    cuda_ml_backend_complete(
      final_path,
      runtime_identity,
      backend_identity
    )
  ) {
    return(final_path)
  }

  dir.create(dirname(final_path), recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile(
    paste0(basename(final_path), "-staging-"),
    tmpdir = dirname(final_path)
  )
  dir.create(staging)
  on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)

  libdir <- cuda_ml_link_runtime(runtime_path, staging)
  backend <- file.path(libdir, paste0("cuda.ml", .Platform$dynlib.ext))
  source_backend <- file.path(
    backend_asset_path,
    "lib",
    paste0("cuda.ml", .Platform$dynlib.ext)
  )
  if (
    !file.copy(
      source_backend,
      backend,
      copy.mode = TRUE,
      copy.date = TRUE
    )
  ) {
    stop("Unable to stage the downloaded cuda.ml backend.", call. = FALSE)
  }

  patchelf <- file.path(runtime_path, "bin", "patchelf")
  cuda_ml_validate_dependencies(patchelf, libdir)
  inventory <- cuda_ml_write_inventory(staging)
  cuda_ml_write_backend_complete(
    staging,
    runtime_identity,
    backend_identity,
    cuda_ml_hash_file(inventory)
  )

  if (dir.exists(final_path)) {
    unlink(final_path, recursive = TRUE, force = TRUE)
  }
  if (!file.rename(staging, final_path)) {
    stop("Unable to publish the prepared cuda.ml backend cache.", call. = FALSE)
  }
  final_path
}

cuda_ml_prepare_runtime <- function() {
  cuda_ml_platform()
  backend_identity <- cuda_ml_backend_identity()
  backend_asset_path <- cuda_ml_backend_asset_path(backend_identity)
  if (!cuda_ml_backend_asset_complete(backend_asset_path, backend_identity)) {
    cuda_ml_install_backend_asset(backend_identity, backend_asset_path)
  }

  runtime_identity <- cuda_ml_runtime_identity()
  runtime_path <- cuda_ml_runtime_path(runtime_identity)
  if (!cuda_ml_runtime_complete(runtime_path, runtime_identity)) {
    cuda_ml_install_runtime(runtime_identity, runtime_path)
  }

  backend_path <- cuda_ml_backend_cache_path(
    runtime_identity,
    backend_identity
  )
  if (
    !cuda_ml_backend_complete(
      backend_path,
      runtime_identity,
      backend_identity
    )
  ) {
    cuda_ml_install_backend(
      runtime_identity,
      backend_identity,
      backend_asset_path,
      runtime_path,
      backend_path
    )
  }

  list(runtime = runtime_path, backend = backend_path)
}

cuda_ml_require_backend <- function() {
  if (!is.null(.cuda_ml_state$dll)) {
    return(.cuda_ml_state$dll)
  }

  cuda_ml_platform()

  selection <- cuda_ml_backend_selection()
  if (identical(selection$backend, "source")) {
    backend_path <- cuda_ml_source_backend_path(selection$source_hash)
    if (!cuda_ml_source_backend_complete(backend_path, selection$source_hash)) {
      stop(
        "The source-built cuda.ml backend is not installed. ",
        "Call cuda_ml_install(source = TRUE) before using native ",
        "cuda.ml operations.",
        call. = FALSE
      )
    }
    dll <- cuda_ml_load_backend(backend_path)
    .cuda_ml_state$dll <- dll
    return(dll)
  }

  runtime_identity <- cuda_ml_runtime_identity()
  runtime_path <- cuda_ml_runtime_path(runtime_identity)
  backend_identity <- cuda_ml_backend_identity()
  backend_path <- cuda_ml_backend_cache_path(
    runtime_identity,
    backend_identity
  )
  if (
    !cuda_ml_runtime_complete(runtime_path, runtime_identity) ||
      !cuda_ml_backend_complete(
        backend_path,
        runtime_identity,
        backend_identity
      )
  ) {
    stop(
      "The managed cuda.ml runtime is not installed. ",
      "Call cuda_ml_install() once before using native cuda.ml operations.",
      call. = FALSE
    )
  }

  dll <- cuda_ml_load_backend(backend_path)
  .cuda_ml_state$dll <- dll
  dll
}

cuda_ml_load_backend <- function(runtime_dir) {
  backend <- file.path(
    runtime_dir,
    "lib",
    paste0("cuda.ml", .Platform$dynlib.ext)
  )
  dll <- dyn.load(backend, local = FALSE, now = TRUE)
  if (!cuda_ml_backend_registration_valid(dll)) {
    dyn.unload(backend)
    stop(
      "The cached cuda.ml backend failed its registration check.",
      call. = FALSE
    )
  }
  dll
}

cuda_ml_backend_registration_valid <- function(dll) {
  tryCatch(
    {
      registered <- getDLLRegisteredRoutines(dll)[[".Call"]]
      registered_manifest <- data.frame(
        symbol = names(registered),
        arity = as.integer(vapply(
          registered,
          `[[`,
          numeric(1),
          "numParameters"
        )),
        stringsAsFactors = FALSE
      )
      registered_manifest <- registered_manifest[
        order(registered_manifest$symbol),
        ,
        drop = FALSE
      ]
      expected_manifest <- .cuda_ml_state$native_symbols[
        order(.cuda_ml_state$native_symbols$symbol),
        ,
        drop = FALSE
      ]
      rownames(registered_manifest) <- NULL
      rownames(expected_manifest) <- NULL
      version_symbol <- getNativeSymbolInfo(
        "_cuda_ml_backend_versions",
        PACKAGE = dll,
        withRegistrationInfo = TRUE
      )
      versions <- do.call(.Call, list(version_symbol))
      toolkit <- as.integer(strsplit(
        unname(.cuda_ml_state$metadata[["CUDA"]]),
        ".",
        fixed = TRUE
      )[[1L]])
      expected_cudart <- toolkit[[1L]] * 1000L + toolkit[[2L]] * 10L
      identical(
        registered_manifest,
        expected_manifest
      ) &&
        identical(
          names(versions),
          c("cuml", "nvforest", "treelite", "cuda_runtime")
        ) &&
        identical(
          package_version(versions[["cuml"]]),
          package_version(unname(.cuda_ml_state$metadata[["RAPIDS"]]))
        ) &&
        identical(
          package_version(versions[["nvforest"]]),
          package_version(unname(.cuda_ml_state$metadata[["nvForest"]]))
        ) &&
        identical(
          package_version(versions[["treelite"]]),
          package_version(unname(.cuda_ml_state$metadata[["Treelite"]]))
        ) &&
        identical(as.integer(versions[["cuda_runtime"]]), expected_cudart)
    },
    error = function(e) FALSE
  )
}

#' Install a cuda.ml native backend
#'
#' By default, downloads, verifies, extracts, and caches the precompiled backend
#' and its runtime libraries. Alternatively, bootstraps a locked CUDA and RAPIDS
#' build toolchain and compiles the native backend on the host. Calling it again
#' with the same inputs is a no-op.
#'
#' @param source A logical value. If \code{FALSE}, install the prebuilt backend
#'   and managed runtime. If \code{TRUE}, compile the backend from the native
#'   sources included in the R package.
#' @param dependencies For a source installation, either \code{"managed"} to
#'   download and cache the exact locked build dependencies, or \code{"host"}
#'   to use explicit host installations.
#' @param architectures For a source installation, \code{NULL}, \code{"native"},
#'   \code{"portable"}, or an explicit semicolon-separated CMake CUDA
#'   architecture list. Managed source builds detect CUDA-visible GPUs by
#'   default and otherwise use the package's portable architecture list.
#'   \code{"native"} requires detection, and \code{"portable"} forces the
#'   package list. Host source builds use \code{CUML_CUDA_ARCHITECTURES} when
#'   this argument is \code{NULL}.
#'
#' @return Invisibly returns \code{TRUE}.
#'
#' @details
#' The default cache is \code{tools::R_user_dir("cuda.ml", "cache")}. Set
#' \code{CUDA_ML_CACHE_DIR} to use a different cache root. Set
#' \code{CUDA_ML_BACKEND_MIRROR} to an \code{https://} or \code{file://}
#' directory containing the exact locked backend archive.
#'
#' A managed source installation downloads no precompiled cuda.ml backend. It
#' downloads and verifies the locked CUDA 13.2.2 and RAPIDS 26.06 development
#' artifacts, CMake, and Ninja; builds Treelite 4.7.0 statically; and caches
#' that toolchain. Only Linux x86_64 with glibc 2.28 or newer and GNU C++ 14 or
#' newer are required on the host. When \code{CUDA_ML_CXX} is unset, the
#' installer prefers \code{g++-14}, then \code{g++}, on \code{PATH}. Set
#' \code{CUDA_ML_CXX} to override this discovery.
#'
#' By default, a managed source build uses \code{nvidia-smi} to detect distinct
#' CUDA-visible GPU compute capabilities and compiles their real targets. It
#' honors \code{CUDA_VISIBLE_DEVICES}. If detection is unavailable, it uses the
#' package's portable list, so GPU-free build hosts remain supported. Set
#' \code{architectures = "native"} to require detection or
#' \code{architectures = "portable"} to force the package list. Native targets
#' usually reduce build time and backend size, but the resulting backend
#' supports only those GPU architectures.
#'
#' A host source installation makes no downloads. It requires CUDA Toolkit
#' 13.2.2 in \code{CUDA_HOME}; a \code{CUML_PREFIX} containing cuML and
#' nvForest 26.06, Treelite 4.7.0 headers, and
#' \code{lib/libtreelite_static.a}; an explicit CMake CUDA architecture list in
#' \code{CUML_CUDA_ARCHITECTURES}; and GNU C++ 14 or newer in
#' \code{CUDA_ML_CXX}. CMake 3.21.1 or newer must be on \code{PATH}.
#'
#' @examples
#' \dontrun{
#' cuda_ml_install()
#'
#' cuda_ml_install(source = TRUE)
#'
#' cuda_ml_install(source = TRUE, architectures = "native")
#'
#' cuda_ml_install(source = TRUE, architectures = "portable")
#'
#' Sys.setenv(
#'   CUDA_HOME = "/usr/local/cuda-13.2",
#'   CUML_PREFIX = "/opt/rapids-26.06",
#'   CUML_CUDA_ARCHITECTURES = "86-real",
#'   CUDA_ML_CXX = "/usr/bin/g++-14"
#' )
#' cuda_ml_install(source = TRUE, dependencies = "host")
#' }
#' @export
cuda_ml_install <- function(
  source = FALSE,
  dependencies = "managed",
  architectures = NULL
) {
  stopifnot(
    is.logical(source),
    length(source) == 1L,
    !is.na(source),
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
  if (
    !source &&
      (!identical(dependencies, "managed") || !is.null(architectures))
  ) {
    stop(
      "dependencies and architectures apply only to source installations.",
      call. = FALSE
    )
  }

  requested <- if (source) "source" else "download"
  if (source) {
    inputs <- cuda_ml_source_build_inputs(dependencies, architectures)
  } else {
    cuda_ml_platform()
    cuda_ml_backend_release()
  }
  selected <- cuda_ml_backend_selection()
  selected_matches <- identical(selected$backend, requested)
  if (
    source &&
      selected_matches &&
      !is.null(.cuda_ml_state$dll)
  ) {
    identity <- cuda_ml_source_identity(inputs)
    selected_matches <- identical(
      selected$source_hash,
      identity$source_hash
    )
  }
  if (
    !is.null(.cuda_ml_state$dll) &&
      !selected_matches
  ) {
    stop(
      "Restart R before changing the selected cuda.ml backend or its ",
      "source-build inputs.",
      call. = FALSE
    )
  }

  lock <- cuda_ml_acquire_lock("cache-install")
  on.exit(filelock::unlock(lock), add = TRUE)
  if (source) {
    prepared <- cuda_ml_prepare_source_backend(inputs)
    cuda_ml_write_backend_selection(
      "source",
      prepared$identity$source_hash
    )
  } else {
    cuda_ml_prepare_runtime()
    cuda_ml_write_backend_selection("download")
  }
  invisible(TRUE)
}

#' Audit the installed native backend
#'
#' Recomputes the hashes recorded when the selected backend was installed and
#' validates its native registration. For a downloaded backend, also validates
#' the complete managed-runtime dependency closure. Ordinary runtime reuse
#' performs only fast marker, inventory, size, and link checks.
#'
#' @return Invisibly returns \code{TRUE}.
#' @export
cuda_ml_runtime_audit <- function() {
  cuda_ml_platform()

  lock <- cuda_ml_acquire_lock("cache-install")
  on.exit(filelock::unlock(lock), add = TRUE)
  selection <- cuda_ml_backend_selection()
  if (identical(selection$backend, "source")) {
    backend_path <- cuda_ml_source_backend_path(selection$source_hash)
    if (
      !cuda_ml_source_backend_complete(
        backend_path,
        selection$source_hash,
        audit = TRUE
      )
    ) {
      stop(
        "The source-built cuda.ml backend failed its content audit. ",
        "Run cuda_ml_cache_clean(), then ",
        "cuda_ml_install(source = TRUE).",
        call. = FALSE
      )
    }
    if (is.null(.cuda_ml_state$dll)) {
      dll <- cuda_ml_load_backend(backend_path)
      on.exit(dyn.unload(dll[["path"]]), add = TRUE)
    } else if (!cuda_ml_backend_registration_valid(.cuda_ml_state$dll)) {
      stop(
        "The loaded cuda.ml backend failed its registration check.",
        call. = FALSE
      )
    }
    return(invisible(TRUE))
  }

  runtime_identity <- cuda_ml_runtime_identity()
  runtime_path <- cuda_ml_runtime_path(runtime_identity)
  backend_identity <- cuda_ml_backend_identity()
  backend_asset_path <- cuda_ml_backend_asset_path(backend_identity)
  backend_path <- cuda_ml_backend_cache_path(
    runtime_identity,
    backend_identity
  )
  if (
    !cuda_ml_backend_asset_complete(
      backend_asset_path,
      backend_identity,
      audit = TRUE
    ) ||
      !cuda_ml_runtime_complete(runtime_path, runtime_identity, audit = TRUE) ||
      !cuda_ml_backend_complete(
        backend_path,
        runtime_identity,
        backend_identity,
        audit = TRUE
      )
  ) {
    stop(
      "The installed cuda.ml runtime failed its content audit. ",
      "Run cuda_ml_cache_clean(), then cuda_ml_install().",
      call. = FALSE
    )
  }

  cuda_ml_validate_dependencies(
    file.path(runtime_path, "bin", "patchelf"),
    file.path(backend_path, "lib")
  )
  if (is.null(.cuda_ml_state$dll)) {
    dll <- cuda_ml_load_backend(backend_path)
    on.exit(dyn.unload(dll[["path"]]), add = TRUE)
  } else if (!cuda_ml_backend_registration_valid(.cuda_ml_state$dll)) {
    stop(
      "The loaded cuda.ml backend failed its registration check.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Remove cuda.ml native-backend caches
#'
#' Removes downloaded and source-built runtime and backend cache generations,
#' including the selected-backend record. Restart R before calling this function
#' if the native backend has been loaded in this process.
#'
#' @return Invisibly returns \code{TRUE}.
#' @export
cuda_ml_cache_clean <- function() {
  if (!is.null(.cuda_ml_state$dll)) {
    stop(
      "Restart R before cleaning a loaded cuda.ml backend cache.",
      call. = FALSE
    )
  }

  lock <- cuda_ml_acquire_lock("cache-install")
  on.exit(filelock::unlock(lock), add = TRUE)
  cache <- cuda_ml_cache_dir()
  generations <- c(
    "runtime-v2",
    "backends-v2",
    "runtime-v3",
    "backend-assets-v1",
    "backends-v3",
    "source-backends-v1",
    "backend-selection-v1",
    "source-toolchains-v1"
  )
  targets <- file.path(cache, generations)
  stopifnot(
    all(dirname(targets) == cache),
    identical(basename(targets), generations)
  )
  for (target in targets[dir.exists(targets)]) {
    unlink(target, recursive = TRUE, force = TRUE)
  }
  if (any(dir.exists(targets))) {
    stop("Unable to remove the cuda.ml managed cache.", call. = FALSE)
  }
  invisible(TRUE)
}
