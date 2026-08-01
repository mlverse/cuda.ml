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
    "Minimum-Driver",
    "Architectures"
  )
  if (nrow(metadata) != 1L || !all(required %in% colnames(metadata))) {
    stop("The cuda.ml backend manifest is invalid.", call. = FALSE)
  }
  metadata <- metadata[1L, , drop = TRUE]
  if (
    !identical(unname(metadata[["Schema"]]), "2") ||
      !unname(metadata[["Backend"]]) %in% c("full", "stub")
  ) {
    stop("The cuda.ml backend manifest is invalid.", call. = FALSE)
  }
  metadata
}

#' Report backend and managed-runtime metadata
#'
#' @return A named list describing the packaged backend and whether its exact
#'   managed-runtime cache is complete. This function performs read-only cache
#'   and inventory checks. It does not create or modify the cache, access the
#'   network, inspect an NVIDIA GPU or driver, or load the native backend.
#' @export
cuda_ml_backend_info <- function() {
  metadata <- .cuda_ml_state$metadata
  value <- function(name) {
    result <- unname(metadata[[name]])
    if (nzchar(result)) result else NA_character_
  }
  architectures <- value("Architectures")
  if (is.na(architectures)) {
    architectures <- character()
  } else {
    architectures <- strsplit(architectures, ";", fixed = TRUE)[[1L]]
  }

  runtime_installed <- FALSE
  runtime_path <- NA_character_
  if (
    identical(value("Backend"), "full") &&
      cuda_ml_supported_platform()
  ) {
    runtime_identity <- cuda_ml_runtime_identity()
    candidate_runtime <- cuda_ml_runtime_path(runtime_identity)
    backend_identity <- cuda_ml_backend_identity()
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

  list(
    package_version = as.character(utils::packageVersion("cuda.ml")),
    backend = value("Backend"),
    build_mode = value("Build-Mode"),
    cuda_version = value("CUDA"),
    rapids_version = value("RAPIDS"),
    nvforest_version = value("nvForest"),
    treelite_version = value("Treelite"),
    platform = value("Platform"),
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

cuda_ml_has_backend <- function() {
  identical(.cuda_ml_state$metadata[["Backend"]], "full")
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

cuda_ml_os_release_value <- function(name) {
  path <- "/etc/os-release"
  if (!file.exists(path)) {
    return("")
  }
  lines <- readLines(path, warn = FALSE)
  values <- lines[startsWith(lines, paste0(name, "="))]
  if (length(values) != 1L) {
    return("")
  }
  value <- substring(values, nchar(name) + 2L)
  sub('^"(.*)"$', "\\1", value)
}

cuda_ml_supported_platform <- function() {
  sysinfo <- Sys.info()
  identical(unname(sysinfo[["sysname"]]), "Linux") &&
    unname(sysinfo[["machine"]]) %in% c("x86_64", "amd64") &&
    identical(cuda_ml_os_release_value("ID"), "ubuntu") &&
    identical(cuda_ml_os_release_value("VERSION_ID"), "26.04")
}

cuda_ml_platform <- function() {
  if (!cuda_ml_supported_platform()) {
    stop(
      "The managed cuda.ml runtime supports Ubuntu 26.04 x86_64 only ",
      "(including WSL2 running that distribution).",
      call. = FALSE
    )
  }
  "ubuntu-26.04-x86_64"
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

cuda_ml_backend_path <- function() {
  libdir <- system.file("libs", package = "cuda.ml")
  candidates <- list.files(
    libdir,
    pattern = paste0(
      "^cuda\\.ml",
      gsub("\\.", "\\\\.", .Platform$dynlib.ext),
      "$"
    ),
    recursive = TRUE,
    full.names = TRUE
  )
  if (length(candidates) != 1L) {
    stop(
      "Unable to locate the packaged cuda.ml backend.",
      call. = FALSE
    )
  }
  normalizePath(candidates, mustWork = TRUE)
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
      "The packaged cuda.ml backend does not match its managed runtime lock.",
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

cuda_ml_backend_identity <- function() {
  backend <- cuda_ml_backend_path()
  r_minor <- strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][[1L]]
  list(
    backend = backend,
    backend_hash = cuda_ml_hash_file(backend),
    r_version = paste0(R.version$major, ".", r_minor)
  )
}

cuda_ml_runtime_path <- function(identity) {
  file.path(
    cuda_ml_cache_dir(),
    "runtime-v2",
    cuda_ml_platform(),
    identity$runtime_hash
  )
}

cuda_ml_backend_cache_path <- function(runtime_identity, backend_identity) {
  file.path(
    cuda_ml_cache_dir(),
    "backends-v2",
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
    "Backend-SHA256",
    "Inventory-SHA256"
  )
  if (
    is.null(metadata) ||
      !all(fields %in% names(metadata)) ||
      !identical(unname(metadata[["Schema"]]), "2") ||
      !identical(
        unname(metadata[["Runtime-SHA256"]]),
        runtime_identity$runtime_hash
      ) ||
      !identical(
        unname(metadata[["Backend-SHA256"]]),
        backend_identity$backend_hash
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
    "Failed to download and verify runtime component '",
    component,
    "' after ",
    attempts,
    " attempts.",
    call. = FALSE
  )
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
        "2",
        runtime_identity$runtime_hash,
        backend_identity$backend_hash,
        inventory_hash
      ),
      nrow = 1L,
      dimnames = list(
        NULL,
        c(
          "Schema",
          "Runtime-SHA256",
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
  if (
    !file.copy(
      backend_identity$backend,
      backend,
      copy.mode = TRUE,
      copy.date = TRUE
    )
  ) {
    stop("Unable to stage the packaged cuda.ml backend.", call. = FALSE)
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

cuda_ml_stub_error <- function() {
  r_minor <- strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][[1L]]
  repository <- paste0(
    "https://mlverse.r-universe.dev/bin/linux/resolute-x86_64/",
    R.version$major,
    ".",
    r_minor,
    "/"
  )
  stop(
    "This cuda.ml installation contains only the CRAN-compatible stub. ",
    "Install the functional Ubuntu 26.04 (Resolute) x86_64 binary from ",
    "the mlverse R-universe repository at ",
    repository,
    " before calling cuda_ml_install() or a modeling function.",
    call. = FALSE
  )
}

cuda_ml_prepare_runtime <- function() {
  if (!cuda_ml_has_backend()) {
    cuda_ml_stub_error()
  }
  cuda_ml_platform()

  runtime_identity <- cuda_ml_runtime_identity()
  runtime_path <- cuda_ml_runtime_path(runtime_identity)
  if (!cuda_ml_runtime_complete(runtime_path, runtime_identity)) {
    cuda_ml_install_runtime(runtime_identity, runtime_path)
  }

  backend_identity <- cuda_ml_backend_identity()
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

  if (!cuda_ml_has_backend()) {
    cuda_ml_stub_error()
  }
  cuda_ml_platform()

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

#' Prepare the managed CUDA and RAPIDS runtime
#'
#' Downloads, verifies, extracts, and caches the runtime libraries required by
#' the precompiled \pkg{cuda.ml} backend. This operation does not load the
#' backend or require a GPU, driver, CUDA toolkit, compiler, Python, or conda.
#' Calling it again with the same package build is a no-op.
#'
#' @return Invisibly returns \code{TRUE}.
#'
#' @details
#' The default cache is \code{tools::R_user_dir("cuda.ml", "cache")}. Set
#' \code{CUDA_ML_CACHE_DIR} to use a different cache root.
#'
#' @examples
#' \dontrun{
#' cuda_ml_install()
#' }
#' @export
cuda_ml_install <- function() {
  if (!cuda_ml_has_backend()) {
    cuda_ml_stub_error()
  }
  cuda_ml_platform()

  lock <- cuda_ml_acquire_lock("cache-install")
  on.exit(filelock::unlock(lock), add = TRUE)
  cuda_ml_prepare_runtime()
  invisible(TRUE)
}

#' Audit the installed managed runtime
#'
#' Recomputes the hashes recorded when the managed runtime was installed and
#' validates the complete native dependency closure. Ordinary runtime reuse
#' performs only fast marker, inventory, size, and link checks.
#'
#' @return Invisibly returns \code{TRUE}.
#' @export
cuda_ml_runtime_audit <- function() {
  if (!cuda_ml_has_backend()) {
    cuda_ml_stub_error()
  }
  cuda_ml_platform()

  lock <- cuda_ml_acquire_lock("cache-install")
  on.exit(filelock::unlock(lock), add = TRUE)
  runtime_identity <- cuda_ml_runtime_identity()
  runtime_path <- cuda_ml_runtime_path(runtime_identity)
  backend_identity <- cuda_ml_backend_identity()
  backend_path <- cuda_ml_backend_cache_path(
    runtime_identity,
    backend_identity
  )
  if (
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

#' Remove managed cuda.ml runtime caches
#'
#' Removes cuda.ml runtime and backend cache generations. Restart R before
#' calling this function if the native backend has been loaded in this process.
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
  generations <- c("runtime-v2", "backends-v2")
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
