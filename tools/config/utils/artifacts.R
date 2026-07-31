cuml_artifact_lock_path <- function(platform = "ubuntu-26.04-x86_64") {
  file.path(pkg_root(), "inst", "artifacts", paste0(platform, ".tsv"))
}

cuml_artifact_metadata <- function(platform = "ubuntu-26.04-x86_64") {
  path <- cuml_artifact_lock_path(platform)
  stopifnot(file.exists(path))
  lines <- readLines(path, warn = FALSE)
  lines <- lines[startsWith(lines, "# ")]
  pattern <- "^# ([^:]+): (.*)$"
  matches <- regexec(pattern, lines)
  values <- regmatches(lines, matches)
  stopifnot(length(values) > 0L, all(lengths(values) == 3L))
  metadata <- setNames(
    vapply(values, `[[`, character(1), 3L),
    vapply(values, `[[`, character(1), 2L)
  )
  required <- c(
    "Schema",
    "CUDA",
    "CUDA-Toolkit",
    "CUDA-Component",
    "CUDA-CCCL",
    "RAPIDS",
    "RAPIDS-Package",
    "nvForest",
    "Treelite",
    "Platform",
    "Minimum-Driver",
    "Architectures",
    "NVRTC-Needed-Old",
    "NVRTC-Needed-New"
  )
  stopifnot(
    identical(names(metadata), required),
    identical(unname(metadata[["Schema"]]), "1"),
    identical(unname(metadata[["Platform"]]), platform)
  )
  metadata
}

cuml_artifact_lock <- function(platform = "ubuntu-26.04-x86_64") {
  path <- cuml_artifact_lock_path(platform)
  stopifnot(file.exists(path))
  artifacts <- utils::read.delim(
    path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    na.strings = character(),
    comment.char = "#"
  )
  stopifnot(all(artifacts$build %in% c("true", "false")))
  artifacts$build <- artifacts$build == "true"
  required <- c(
    "component",
    "package",
    "version",
    "archive",
    "filename",
    "url",
    "sha256",
    "size",
    "build",
    "runtime_extract_regex"
  )
  stopifnot(
    identical(names(artifacts), required),
    nrow(artifacts) > 0L,
    !anyDuplicated(artifacts$component),
    all(grepl("^[a-z0-9-]+$", artifacts$component)),
    all(artifacts$archive %in% c("zip", "tar.gz")),
    all(basename(artifacts$filename) == artifacts$filename),
    all(startsWith(artifacts$url, "https://files.pythonhosted.org/")),
    all(grepl("^[[:xdigit:]]{64}$", artifacts$sha256)),
    all(artifacts$size > 0)
  )
  metadata <- cuml_artifact_metadata(platform)
  version <- function(component) {
    value <- artifacts$version[artifacts$component == component]
    stopifnot(length(value) == 1L)
    value
  }
  stopifnot(
    identical(version("libcuml"), unname(metadata[["RAPIDS-Package"]])),
    identical(
      package_version(version("libnvforest")),
      package_version(unname(metadata[["nvForest"]]))
    ),
    identical(version("treelite"), unname(metadata[["Treelite"]])),
    identical(version("treelite-headers"), unname(metadata[["Treelite"]])),
    identical(version("cccl"), unname(metadata[["CUDA-CCCL"]])),
    identical(version("nvcc"), unname(metadata[["CUDA-Component"]])),
    identical(version("nvrtc"), unname(metadata[["CUDA-Component"]])),
    identical(version("cudart"), unname(metadata[["CUDA-Component"]]))
  )
  artifacts
}

cuml_generate_runtime_lock <- function() {
  artifacts <- cuml_artifact_lock()
  runtime <- artifacts[nzchar(artifacts$runtime_extract_regex), , drop = FALSE]
  runtime <- runtime[c(
    "component",
    "package",
    "version",
    "filename",
    "url",
    "sha256",
    "size",
    "runtime_extract_regex"
  )]
  names(runtime)[[8L]] <- "extract_regex"

  path <- file.path(
    pkg_root(),
    "inst",
    "runtime",
    "ubuntu-26.04-x86_64.tsv"
  )
  utils::write.table(
    runtime,
    file = path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )
  writeLines(
    c(
      "Schema: 2",
      paste0("CUDA: ", cuml_managed_cuda_version()),
      paste0("CUDA-Toolkit: ", cuml_managed_cuda_toolkit_version()),
      paste0("RAPIDS: ", cuml_managed_rapids_version()),
      paste0("nvForest: ", cuml_managed_nvforest_version()),
      paste0("Treelite: ", cuml_managed_treelite_version()),
      paste0("Platform: ", cuml_managed_platform()),
      paste0("Minimum-Driver: ", cuml_managed_minimum_driver()),
      paste0(
        "NVRTC-Needed-Old: ",
        unname(cuml_artifact_metadata()[["NVRTC-Needed-Old"]])
      ),
      paste0(
        "NVRTC-Needed-New: ",
        unname(cuml_artifact_metadata()[["NVRTC-Needed-New"]])
      )
    ),
    file.path(pkg_root(), "inst", "runtime", "runtime.dcf")
  )
  invisible(path)
}

cuml_artifact_cache_dir <- function() {
  file.path(cuml_bootstrap_cache_dir(), "artifacts")
}

cuml_artifact_hash <- function(path) {
  unname(digest::digest(file = path, algo = "sha256", serialize = FALSE))
}

cuml_artifact_valid <- function(path, row) {
  file.exists(path) &&
    identical(
      as.numeric(file.info(path)[["size"]]),
      as.numeric(row[["size"]])
    ) &&
    identical(cuml_artifact_hash(path), row[["sha256"]])
}

cuml_download_artifact <- function(row) {
  directory <- file.path(cuml_artifact_cache_dir(), row[["sha256"]])
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  destination <- file.path(directory, row[["filename"]])
  if (cuml_artifact_valid(destination, row)) {
    return(destination)
  }
  if (file.exists(destination)) {
    unlink(destination)
  }

  temporary <- tempfile("download-", tmpdir = directory)
  on.exit(unlink(temporary), add = TRUE)
  old_timeout <- getOption("timeout")
  on.exit(options(timeout = old_timeout), add = TRUE)
  options(timeout = max(600, old_timeout))
  attempts <- 3L
  verified <- FALSE
  for (attempt in seq_len(attempts)) {
    unlink(temporary, force = TRUE)
    if (attempt > 1L) {
      message(
        "Retrying locked build artifact ",
        row[["filename"]],
        " (attempt ",
        attempt,
        " of ",
        attempts,
        ")"
      )
    }
    status <- tryCatch(
      utils::download.file(
        row[["url"]],
        temporary,
        mode = "wb",
        quiet = FALSE
      ),
      error = function(e) e
    )
    downloaded <- !inherits(status, "error") && identical(status, 0L)
    verified <- downloaded && cuml_artifact_valid(temporary, row)
    if (verified) {
      break
    }
  }
  if (!verified) {
    stop2(
      "Failed to download and verify the locked build artifact ",
      row[["filename"]],
      " after ",
      attempts,
      " attempts."
    )
  }
  if (!file.rename(temporary, destination)) {
    stop2(
      "Unable to publish the verified build artifact ",
      row[["filename"]],
      "."
    )
  }
  destination
}

cuml_extract_artifact <- function(row, destination) {
  archive <- cuml_download_artifact(row)
  if (identical(row[["archive"]], "zip")) {
    utils::unzip(archive, exdir = destination)
  } else {
    utils::untar(archive, exdir = destination)
  }
  invisible(TRUE)
}
