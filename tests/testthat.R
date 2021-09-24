library(testthat)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  filter <- Sys.getenv("TESTTHAT_FILTER", unset = "")
  if (identical(filter, "")) filter <- NULL

  reporter <- MultiReporter$new(reporters = list(
    CheckReporter$new(),
    LocationReporter$new(),
    SummaryReporter$new(show_praise = FALSE)
  ))

  test_check("cuda.ml", filter = filter, reporter = reporter)
}

