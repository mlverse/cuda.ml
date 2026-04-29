# A list containing libcuml download links for "cuml_versions" and CUDA major versions.
#
# For cuML 21.x: pre-built zip archives from mlverse/libcuml-builds GitHub releases.
# For cuML 26.x+: pip wheels from PyPI (libcuml-cu12). The wheel is a zip containing
#   headers in libcuml/include/cuml/ and shared libs in libcuml/lib64/.
libcuml_versions <- list(
  "21.08" = list(
    "11" = "https://github.com/mlverse/libcuml-builds/releases/download/v21.08-cuda11.2.1/libcuml-21.08-cuda11.2.1.zip"
  ),
  "21.10" = list(
    "11" = "https://github.com/mlverse/libcuml-builds/releases/download/v21.10-cuda11.2.1/libcuml-21.10-cuda11.2.1.zip"
  ),
  "21.12" = list(
    "11" = "https://github.com/mlverse/libcuml-builds/releases/download/v21.12-cuda11.2.1/libcuml-21.12-cuda11.2.1.zip"
  ),
  "26.04" = list(
    "12" = "https://files.pythonhosted.org/packages/84/dd/00031bd84a6cd42f028273ef0acab780d6bb5981a024c11fd1bcd66fdec0/libcuml_cu12-26.4.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  )
)
