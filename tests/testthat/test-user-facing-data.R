test_that("package sources do not use deprecated example data", {
  package_root <- normalizePath(testthat::test_path("..", ".."))
  source_directories <- c(
    "R",
    "tests/testthat",
    "vignettes"
  )
  source_paths <- file.path(package_root, source_directories)
  files <- list.files(
    source_paths,
    pattern = "\\.(R|Rmd)$",
    recursive = TRUE,
    full.names = TRUE
  )
  contents <- unlist(lapply(files, readLines, warn = FALSE), use.names = FALSE)

  deprecated_name <- paste0("ir", "is")
  expect_false(any(grepl(
    paste0("\\b", deprecated_name, "\\b"),
    contents,
    ignore.case = TRUE
  )))
})
