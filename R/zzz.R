.onLoad <- function(libname, pkgname) {
  if (!cuml4r_has_cuml()) {
    warning(
      "
      The current installation of '", pkgname, "' will not function as expected
      because it was not linked with a valid version of the 'cuML' shared
      library.

      To fix this issue, please follow https://rapids.ai/start.html#get-rapids
      to install 'cuML' from Conda and ensure the 'CUDA_PATH' env variable is
      set to a valid RAPIDS conda env directory (e.g.,
      '/home/user/anaconda3/envs/rapids-21.06' or similar) during the install-
      ation of 'cuml4r', or alternatively, follow
      https://github.com/yitao-li/cuml-installation-notes#build-from-source-without-conda-and-without-multi-gpu-support
      or
      https://github.com/yitao-li/cuml-installation-notes#build-from-source-without-conda-and-with-multi-gpu-support
      or similar to build and install 'cuML' from source, and then re-install '",
      pkgname, "'\n\n"
    )
  }
}
