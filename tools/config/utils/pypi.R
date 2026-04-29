# Resolve the full dependency tree for a PyPI package and return download URLs
# for all packages (including transitive deps) that contain C++ headers.
#
# This is used to download libcuml and all its header-only dependencies
# (libraft, librmm, rapids-logger, nvidia-cccl, etc.) from PyPI without
# needing pip installed.

pypi_package_info <- function(package, version = NULL) {
  url <- if (is.null(version)) {
    sprintf("https://pypi.org/pypi/%s/json", package)
  } else {
    sprintf("https://pypi.org/pypi/%s/%s/json", package, version)
  }
  tmp <- tempfile(fileext = ".json")
  download.file(url, tmp, quiet = TRUE)
  jsonlite::fromJSON(tmp)
}

pypi_wheel_url <- function(package, version = NULL, platform = "x86_64") {
  info <- pypi_package_info(package, version)
  urls <- info$urls
  # Find a matching wheel for the platform
  idx <- grep(platform, urls$filename)
  if (length(idx) == 0) {
    # Try platform-independent wheels
    idx <- grep("none-any", urls$filename)
  }
  if (length(idx) == 0) {
    stop(sprintf("No wheel found for %s (platform: %s)", package, platform))
  }
  list(
    url = urls$url[idx[1]],
    filename = urls$filename[idx[1]],
    version = info$info$version,
    requires_dist = info$info$requires_dist
  )
}

# Parse a PEP 508 dependency string into package name and version
# e.g. "libraft-cu12==25.12.*" -> list(name = "libraft-cu12", version = "25.12.0")
# e.g. "numpy>=1.0; extra == 'test'" -> NULL (skip extras)
parse_dep <- function(dep_str) {
  # Skip deps with extras/markers like "; extra == ..."
  if (grepl("; extra\\s*==", dep_str)) return(NULL)
  # Also skip deps with platform markers that could exclude linux
  if (grepl(";", dep_str) && !grepl("linux", dep_str)) return(NULL)
  # Extract package name (everything before version specifier or semicolon)
  name <- gsub("[\\s;(<>=!\\[,].*", "", dep_str, perl = TRUE)
  # Extract pinned version if present (e.g. ==25.12.*)
  version <- NULL
  if (grepl("==", dep_str)) {
    ver_str <- sub(".*==\\s*", "", sub(";.*", "", dep_str))
    ver_str <- gsub("\\*", "0", ver_str)  # 25.12.* -> 25.12.0
    version <- ver_str
  }
  list(name = name, version = version)
}

# Resolve all transitive dependencies that look like C++ library packages
# (lib*, rapids-*, nvidia-cccl-*, nvidia-nvjitlink-*)
# The package_spec can be "libcuml-cu12" or "libcuml-cu12==25.12.*"
resolve_native_deps <- function(package_spec, seen = character()) {
  dep <- parse_dep(package_spec)
  if (is.null(dep)) return(list())
  package <- dep$name
  version <- dep$version

  if (package %in% seen) return(list())
  seen <- c(seen, package)

  info <- tryCatch(
    pypi_wheel_url(package, version),
    error = function(e) NULL
  )
  if (is.null(info)) return(list())

  result <- list()
  result[[package]] <- info$url

  # Only chase transitive deps for native/C++ packages
  if (!is.null(info$requires_dist)) {
    for (dep_str in info$requires_dist) {
      dep <- parse_dep(dep_str)
      if (is.null(dep)) next
      # Only follow native library deps (lib*, rapids-*, nvidia-cccl*, nvidia-nvjitlink*)
      if (grepl("^(lib|rapids-|nvidia-cccl|nvidia-nvjitlink)", dep$name)) {
        sub_deps <- resolve_native_deps(dep_str, seen = seen)
        seen <- c(seen, names(sub_deps))
        result <- c(result, sub_deps)
      }
    }
  }

  result
}
