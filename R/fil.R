#' Determine whether Forest Inference Library (FIL) functionalities are enabled
#' in the current installation of `cuml4r`.
#'
#' CuML Forest Inference Library (FIL) functionalities (see
#' https://github.com/rapidsai/cuml/tree/main/python/cuml/fil#readme) will
#' require Treelite C API. If you need FIL to run tree-based model ensemble on
#' GPU, and \code{cuml4r_fil_enabled()} returns FALSE, then please consider
#' installing Treelite and then re-installing 'cuml4r'.
#'
#' @examples
#' if (cuml4r_fil_enabled()) {
#'   # run GPU-accelerated Forest Inference Library (FIL) functionalities
#' } else {
#'   message(
#'     "FIL functionalities are disabled in the current installation of "
#'     "`cuml4r`."
#'   )
#' }
#'
#' @export
cuml4r_fil_enabled <- .fil_enabled
