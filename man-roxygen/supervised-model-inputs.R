#' @param x Depending on the context:
#'
#'   * A __data frame__ of predictors.
#'   * A __matrix__ of predictors.
#'   * A __recipe__ specifying a set of preprocessing steps
#'   * created from [recipes::recipe()].
#'   * A __formula__ specifying the predictors and the outcome.
#' @param formula A formula specifying the outcome terms on the left-hand side,
#'  and the predictor terms on the right-hand side.
#' @param data When a __recipe__ or __formula__ is used, \code{data} is
#'   specified as a  __data frame__ containing the predictors and (if
#'   applicable) the outcome.
