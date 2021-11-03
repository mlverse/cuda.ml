#' @param output_class_probabilities Whether to output class probabilities.
#'   NOTE: setting \code{output_class_probabilities} to \code{TRUE} is only
#'   valid when the model being applied is a classification model and supports
#'   class probabilities output. CuML classification models supporting class
#'   probabilities include \code{knn}, \code{fil}, and \code{rand_forest}.
#'   A warning message will be emitted if \code{output_class_probabilities}
#'   is set to \code{TRUE} or \code{FALSE} but the model being applied does
#'   not support class probabilities output.
