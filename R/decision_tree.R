decision_tree_match_split_criterion <- function(
  criterion = NULL,
  classification = TRUE
) {
  choices <- if (classification) {
    c("gini", "entropy")
  } else {
    c("mse", "poisson", "gamma", "inverse_gaussian")
  }
  criterion <- match.arg(criterion %||% choices[[1L]], choices)

  switch(
    criterion,
    gini = 0L,
    entropy = 1L,
    mse = 2L,
    poisson = 4L,
    gamma = 5L,
    inverse_gaussian = 6L
  )
}
