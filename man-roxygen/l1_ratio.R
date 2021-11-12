#' @param l1_ratio The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
#'   For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1
#'   penalty.
#'   For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
#'   The penalty term is computed using the following formula:
#'     penalty = \code{alpha} * \code{l1_ratio} * ||w||_1 +
#'               0.5 * \code{alpha} * (1 - \code{l1_ratio}) * ||w||^2_2
#'   where ||w||_1 is the L1 norm of the coefficients, and ||w||_2 is the L2
#'   norm of the coefficients.
