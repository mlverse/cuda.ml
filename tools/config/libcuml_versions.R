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
    "12" = list(
      cuml = "https://files.pythonhosted.org/packages/84/dd/00031bd84a6cd42f028273ef0acab780d6bb5981a024c11fd1bcd66fdec0/libcuml_cu12-26.4.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
      raft = "https://files.pythonhosted.org/packages/92/72/a05d2122f1279ce8bc4bb652bc13089b2ee701a64bd1a483537a54639f8c/libraft_cu12-26.4.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
      rmm = "https://files.pythonhosted.org/packages/6d/82/783151d344aece612484041c92a94fa4261653f28a45375a6fc8a6100995/librmm_cu12-26.4.0-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
    )
  )
)
