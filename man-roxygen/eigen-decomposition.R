#' @param eig_algo Eigen decomposition algorithm to be applied to the covariance
#'   matrix. Valid choices are "dq" (divid-and-conquer method for symmetric
#'   matrices) and "jacobi" (the Jacobi method for symmetric matrices).
#'   Default: "dq".
#' @param tol Tolerance for singular values computed by the Jacobi method.
#'   Default: 1e-7.
#' @param n_iters Maximum number of iterations for the Jacobi method.
#'   Default: 15.
