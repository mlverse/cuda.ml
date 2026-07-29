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
  required <- c("Schema", "Backend", "CUDA", "RAPIDS", "Architectures")
  if (nrow(metadata) != 1L || !all(required %in% colnames(metadata))) {
    stop("The cuda.ml backend manifest is invalid.", call. = FALSE)
  }
  metadata <- metadata[1L, , drop = TRUE]
  if (
    !identical(unname(metadata[["Schema"]]), "1") ||
      !unname(metadata[["Backend"]]) %in% c("full", "stub")
  ) {
    stop("The cuda.ml backend manifest is invalid.", call. = FALSE)
  }
  metadata
}

cuda_ml_native_symbol <- function(fn) {
  if (!is.function(fn)) {
    return(NULL)
  }

  expr <- body(fn)
  if (
    is.call(expr) &&
    identical(expr[[1L]], as.name("{")) &&
    length(expr) == 2L
  ) {
    expr <- expr[[2L]]
  }
  if (
    !is.call(expr) ||
    !identical(expr[[1L]], as.name(".Call")) ||
    length(expr) < 2L ||
    !is.symbol(expr[[2L]])
  ) {
    return(NULL)
  }

  symbol <- as.character(expr[[2L]])
  if (!startsWith(symbol, "_cuda_ml_")) {
    return(NULL)
  }
  symbol
}

cuda_ml_native_wrappers <- function(ns) {
  names <- ls(ns, all.names = TRUE)
  names <- names[!startsWith(names, "_cuda_ml_")]
  objects <- mget(names, ns, inherits = FALSE)
  symbols <- lapply(objects, cuda_ml_native_symbol)
  keep <- !vapply(symbols, is.null, logical(1))
  unlist(symbols[keep], use.names = TRUE)
}

cuda_ml_referenced_native_symbols <- function(ns) {
  names <- ls(ns, all.names = TRUE)
  names <- names[!startsWith(names, "_cuda_ml_")]
  objects <- mget(names, ns, inherits = FALSE)
  functions <- objects[vapply(objects, is.function, logical(1))]
  source <- unlist(
    lapply(functions, function(fn) deparse(body(fn))),
    use.names = FALSE
  )
  matches <- regmatches(
    source,
    gregexpr("`_cuda_ml_[[:alnum:]_]+`", source)
  )
  unique(gsub("`", "", unlist(matches, use.names = FALSE), fixed = TRUE))
}

cuda_ml_bind_native_symbols <- function(ns, wrappers) {
  for (symbol in unname(wrappers)) {
    eval_env <- list2env(
      list(symbol = symbol),
      parent = environment()
    )
    delayedAssign(
      symbol,
      cuda_ml_resolve_native_symbol(symbol),
      eval.env = eval_env,
      assign.env = ns
    )
  }
  invisible(NULL)
}

cuda_ml_resolve_native_symbol <- function(symbol) {
  # Package installation forces namespace bindings before the installed
  # backend and runtime are available.
  if (identical(Sys.getenv("R_INSTALL_PKG"), "cuda.ml")) {
    return(cuda_ml_check_native_symbol(symbol))
  }
  # Static checks of a stub build need registration metadata, not a backend.
  if (
    identical(Sys.getenv("_R_CHECK_PACKAGE_NAME_"), "cuda.ml") &&
      !has_cuML()
  ) {
    return(cuda_ml_check_native_symbol(symbol))
  }
  getNativeSymbolInfo(
    symbol,
    PACKAGE = .cuda_ml_state$dll,
    withRegistrationInfo = TRUE
  )
}

cuda_ml_check_native_symbol <- function(symbol) {
  structure(
    list(
      name = symbol,
      address = NULL,
      dll = structure(list(name = "cuda.ml"), class = "DLLInfo"),
      numParameters = -1L
    ),
    class = c("CallRoutine", "NativeSymbolInfo")
  )
}

cuda_ml_cache_dir <- function() {
  cache <- Sys.getenv("CUDA_ML_CACHE_DIR", unset = "")
  if (!nzchar(cache)) {
    cache <- tools::R_user_dir("cuda.ml", "cache")
  }
  normalizePath(path.expand(cache), mustWork = FALSE)
}

cuda_ml_platform <- function() {
  sysinfo <- Sys.info()
  if (
    !identical(unname(sysinfo[["sysname"]]), "Linux") ||
    !unname(sysinfo[["machine"]]) %in% c("x86_64", "amd64")
  ) {
    stop(
      "The managed cuda.ml runtime supports Linux x86_64 only ",
      "(including Linux under WSL2).",
      call. = FALSE
    )
  }
  "linux-x86_64"
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
    "component", "package", "version", "filename", "url", "sha256",
    "size", "extract_regex"
  )
  stopifnot(identical(names(manifest), required), nrow(manifest) > 0L)

  metadata <- read.dcf(metadata_path)
  required_metadata <- c(
    "Schema", "CUDA", "RAPIDS", "Platform", "Minimum-Driver",
    "Architectures", "NVRTC-Needed-Old", "NVRTC-Needed-New"
  )
  stopifnot(
    nrow(metadata) == 1L,
    all(required_metadata %in% colnames(metadata))
  )
  metadata <- metadata[1L, , drop = TRUE]
  nvrtc_needed_old <- unname(metadata[["NVRTC-Needed-Old"]])
  nvrtc_needed_new <- unname(metadata[["NVRTC-Needed-New"]])
  stopifnot(
    identical(unname(metadata[["Schema"]]), "1"),
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
    pattern = paste0("^cuda\\.ml", gsub("\\.", "\\\\.", .Platform$dynlib.ext), "$"),
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
  backend <- cuda_ml_backend_path()
  if (
    !identical(
      unname(.cuda_ml_state$metadata[["CUDA"]]),
      unname(manifest$metadata[["CUDA"]])
    ) ||
      !identical(
        unname(.cuda_ml_state$metadata[["RAPIDS"]]),
        unname(manifest$metadata[["RAPIDS"]])
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
  r_minor <- strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][[1L]]
  list(
    manifest = manifest,
    backend = backend,
    runtime_hash = unname(digest::digest(
      paste(lock_hashes, collapse = ":"),
      algo = "sha256",
      serialize = FALSE
    )),
    backend_hash = cuda_ml_hash_file(backend),
    r_version = paste0(R.version$major, ".", r_minor)
  )
}

cuda_ml_runtime_path <- function(identity) {
  file.path(
    cuda_ml_cache_dir(),
    "runtime-v1",
    cuda_ml_platform(),
    paste0("r-", identity$r_version),
    identity$runtime_hash,
    identity$backend_hash
  )
}

cuda_ml_runtime_complete <- function(path, identity) {
  marker <- file.path(path, ".complete")
  inventory <- file.path(path, "inventory.tsv")
  backend <- file.path(path, "lib", paste0("cuda.ml", .Platform$dynlib.ext))
  if (
    !file.exists(marker) ||
      !file.exists(inventory) ||
      !file.exists(backend)
  ) {
    return(FALSE)
  }

  metadata <- tryCatch(
    read.dcf(marker),
    error = function(e) NULL
  )
  if (is.null(metadata) || nrow(metadata) != 1L) {
    return(FALSE)
  }
  fields <- c(
    "Schema", "Runtime-SHA256", "Backend-SHA256", "Inventory-SHA256"
  )
  if (!all(fields %in% colnames(metadata))) {
    return(FALSE)
  }
  if (
    !identical(unname(metadata[1L, "Schema"]), "1") ||
      !identical(
        unname(metadata[1L, "Runtime-SHA256"]),
        identity$runtime_hash
      ) ||
      !identical(
        unname(metadata[1L, "Backend-SHA256"]),
        identity$backend_hash
      ) ||
      !identical(
        unname(metadata[1L, "Inventory-SHA256"]),
        cuda_ml_hash_file(inventory)
      )
  ) {
    return(FALSE)
  }

  files <- tryCatch(
    utils::read.delim(
      inventory,
      stringsAsFactors = FALSE,
      check.names = FALSE,
      colClasses = c("character", "numeric", "character", "character"),
      na.strings = character()
    ),
    error = function(e) NULL
  )
  if (
    is.null(files) ||
      !identical(names(files), c("file", "size", "sha256", "link")) ||
      !nrow(files)
  ) {
    return(FALSE)
  }
  if (
    any(dirname(files$file) != "lib") ||
      any(basename(files$file) != sub("^lib/", "", files$file))
  ) {
    return(FALSE)
  }

  paths <- file.path(path, files$file)
  actual <- sort(list.files(
    file.path(path, "lib"),
    full.names = FALSE,
    all.files = TRUE,
    no.. = TRUE
  ))
  identical(actual, sort(basename(paths))) &&
    all(file.exists(paths)) &&
    identical(
      as.numeric(file.info(paths)[["size"]]),
      as.numeric(files$size)
    ) &&
    identical(
      unname(vapply(paths, Sys.readlink, character(1))),
      unname(files$link)
    ) &&
    identical(
      unname(vapply(paths, cuda_ml_hash_file, character(1))),
      unname(files$sha256)
    )
}

cuda_ml_download <- function(component, url, destination, size, sha256) {
  message(
    "Downloading ", component, " (",
    format(round(as.numeric(size) / 1024^2, 1), trim = TRUE),
    " MiB)"
  )
  old_timeout <- getOption("timeout")
  on.exit(options(timeout = old_timeout), add = TRUE)
  options(timeout = max(600, old_timeout))

  status <- tryCatch(
    utils::download.file(url, destination, mode = "wb", quiet = FALSE),
    error = function(e) e
  )
  if (inherits(status, "error") || !identical(status, 0L)) {
    stop("Failed to download runtime component '", component, "'.", call. = FALSE)
  }
  actual_size <- file.info(destination)[["size"]]
  if (!identical(as.numeric(actual_size), as.numeric(size))) {
    stop("Downloaded size mismatch for runtime component '", component, "'.", call. = FALSE)
  }
  actual_hash <- cuda_ml_hash_file(destination)
  if (!identical(actual_hash, sha256)) {
    stop("SHA-256 mismatch for runtime component '", component, "'.", call. = FALSE)
  }
  invisible(destination)
}

cuda_ml_extract_component <- function(archive, row, directory) {
  listing <- utils::unzip(archive, list = TRUE)
  files <- listing$Name[
    grepl(row[["extract_regex"]], listing$Name, perl = TRUE)
  ]
  if (!length(files)) {
    stop(
      "Runtime component '", row[["component"]],
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
        basename(source), "'.",
        call. = FALSE
      )
    }
    return(invisible(destination))
  }
  if (!file.copy(source, destination, copy.mode = TRUE, copy.date = TRUE)) {
    stop("Failed to stage runtime file '", basename(source), "'.", call. = FALSE)
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
      basename(file), "'.",
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
      "The locked NVRTC dependency in '", basename(file),
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
      basename(file), "'.",
      call. = FALSE
    )
  }

  needed <- cuda_ml_needed(patchelf, file)
  if (old %in% needed || sum(needed == new) != 1L) {
    stop(
      "Failed to validate the replacement NVRTC dependency in '",
      basename(file), "'.",
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
            soname, "'.",
            call. = FALSE
          )
        }
      } else if (!file.symlink(basename(file), link)) {
        stop(
          "Unable to preserve runtime SONAME '",
          soname, "'.",
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
    stop("The managed runtime must not bundle the NVIDIA driver.", call. = FALSE)
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
        basename(file), ".",
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
        basename(file), ": ", paste(missing, collapse = ", "),
        call. = FALSE
      )
    }
  }
  invisible(TRUE)
}

cuda_ml_write_inventory <- function(path) {
  files <- sort(list.files(
    file.path(path, "lib"),
    full.names = TRUE,
    all.files = TRUE,
    no.. = TRUE
  ))
  stopifnot(length(files) > 1L)
  inventory <- data.frame(
    file = file.path("lib", basename(files)),
    size = as.numeric(file.info(files)[["size"]]),
    sha256 = vapply(files, cuda_ml_hash_file, character(1)),
    link = vapply(files, Sys.readlink, character(1)),
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

cuda_ml_write_complete <- function(path, identity, inventory_hash) {
  write.dcf(
    matrix(
      c(
        "1",
        identity$runtime_hash,
        identity$backend_hash,
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

cuda_ml_install_runtime <- function(identity, final_path) {
  cache <- cuda_ml_cache_dir()
  lock_dir <- file.path(cache, "locks")
  dir.create(lock_dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(lock_dir)) {
    stop("Unable to create the cuda.ml cache directory '", cache, "'.", call. = FALSE)
  }

  lock_path <- file.path(
    lock_dir,
    paste0(identity$runtime_hash, "-", identity$backend_hash, ".lock")
  )
  lock <- filelock::lock(lock_path, timeout = Inf)
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
  dir.create(download_dir)
  dir.create(extract_dir)
  dir.create(libdir)

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
      Sys.chmod(extracted, mode = "0755")
      patchelf <- extracted
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

  backend <- file.path(libdir, paste0("cuda.ml", .Platform$dynlib.ext))
  if (!file.copy(identity$backend, backend, copy.mode = TRUE, copy.date = TRUE)) {
    stop("Unable to copy the cuda.ml backend into its runtime cache.", call. = FALSE)
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
  cuda_ml_write_complete(
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

cuda_ml_prepare_runtime <- function() {
  cuda_ml_platform()

  if (!has_cuML()) {
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
      "the mlverse R-universe repository at ", repository,
      " before calling cuda_ml_install().",
      call. = FALSE
    )
  }

  identity <- cuda_ml_runtime_identity()
  path <- cuda_ml_runtime_path(identity)
  if (cuda_ml_runtime_complete(path, identity)) {
    return(path)
  }
  cuda_ml_install_runtime(identity, path)
}

cuda_ml_load_backend <- function(runtime_dir) {
  backend <- file.path(
    runtime_dir,
    "lib",
    paste0("cuda.ml", .Platform$dynlib.ext)
  )
  dll <- dyn.load(backend, local = FALSE, now = TRUE)
  valid <- tryCatch(
    {
      registered <- names(getDLLRegisteredRoutines(dll)[[".Call"]])
      capability <- getNativeSymbolInfo(
        "_cuda_ml_has_cuML",
        PACKAGE = dll,
        withRegistrationInfo = TRUE
      )
      major <- getNativeSymbolInfo(
        "_cuda_ml_cuML_major_version",
        PACKAGE = dll,
        withRegistrationInfo = TRUE
      )
      minor <- getNativeSymbolInfo(
        "_cuda_ml_cuML_minor_version",
        PACKAGE = dll,
        withRegistrationInfo = TRUE
      )
      version <- sprintf(
        "%s.%02d",
        as.character(do.call(.Call, list(major))),
        as.integer(do.call(.Call, list(minor)))
      )
      identical(
        sort(registered),
        sort(.cuda_ml_state$native_symbols)
      ) &&
        isTRUE(do.call(.Call, list(capability))) &&
        identical(
          version,
          unname(.cuda_ml_state$metadata[["RAPIDS"]])
        )
    },
    error = function(e) FALSE
  )
  if (!valid) {
    dyn.unload(backend)
    stop("The cached cuda.ml backend failed its registration check.", call. = FALSE)
  }
  dll
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
  .cuda_ml_state$runtime_dir
  invisible(TRUE)
}
