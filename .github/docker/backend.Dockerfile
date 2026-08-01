# syntax=docker/dockerfile:1.7

ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64@sha256:fdb9a9c223b215604dc7b6f7e8fff4b39bfea5fbaa7777a2e5544a60dfa437f8

FROM ${MANYLINUX_IMAGE} AS base

ARG R_VERSION
RUN test -n "${R_VERSION}" \
    && curl -fsSL \
      "https://cdn.posit.co/r/centos-8/pkgs/R-${R_VERSION}-1-1.x86_64.rpm" \
      -o /tmp/R.rpm \
    && dnf install -y /tmp/R.rpm \
    && rm -f /tmp/R.rpm \
    && dnf clean all

ENV PATH=/opt/R/${R_VERSION}/bin:${PATH}
ENV R_LIBS_USER=/opt/R/library
RUN mkdir -p /opt/R/library

FROM base AS backend-build

ARG SOURCE_COMMIT
ARG BACKEND_BASE_URL
RUN test -n "${SOURCE_COMMIT}" && test -n "${BACKEND_BASE_URL}"

COPY DESCRIPTION /build/DESCRIPTION
COPY R/RcppExports.R /build/R/RcppExports.R
COPY inst/artifacts/ /build/inst/artifacts/
COPY inst/runtime/ /build/inst/runtime/
COPY inst/cuda-ml-backend.dcf inst/native-symbols.txt /build/inst/
COPY tools/config.R /build/tools/config.R
COPY tools/config/ /build/tools/config/
COPY inst/backend-src/ /build/inst/backend-src/
WORKDIR /build

ENV CUDA_ML_BUILD_MODE=managed
ENV CUML_BOOTSTRAP_CACHE=/opt/cuda.ml
ENV CMAKE_BUILD_PARALLEL_LEVEL=2

RUN Rscript -e \
    "install.packages(c('Rcpp', 'digest'), repos = 'https://cloud.r-project.org')"
RUN Rscript tools/config.R configure
RUN cmake \
      --build inst/backend-src/.cmake-build \
      --target cuda.ml \
      --parallel 2

FROM backend-build AS backend

COPY tools/audit-backend.R tools/package-backend.R tools/nvrtc-probe.c /build/tools/

RUN CUDA_ML_PREFIX="$(Rscript -e \
      "pkg_root <- function() '/build'; source('tools/config/utils/artifacts.R'); source('tools/config/utils/bootstrap.R'); cat(cuml_managed_bootstrap_prefix())")" \
    && LD_LIBRARY_PATH="${CUDA_ML_PREFIX}/lib" \
      Rscript tools/audit-backend.R \
        inst/backend-src/.cmake-build/cuda.ml.so \
        "${CUDA_ML_PREFIX}/bin/cuobjdump" \
    && cc -std=c11 -Wall -Wextra -Werror \
      -I"${CUDA_ML_PREFIX}/include" \
      -L"${CUDA_ML_PREFIX}/lib" \
      tools/nvrtc-probe.c \
      -lnvrtc \
      -o /usr/local/bin/nvrtc-probe

RUN mkdir -p /out \
    && Rscript tools/package-backend.R \
      inst/backend-src/.cmake-build/cuda.ml.so \
      /out \
      "${SOURCE_COMMIT}" \
      "${BACKEND_BASE_URL}"

FROM scratch AS export
COPY --from=backend /out /

FROM backend AS test-build

COPY . /build

RUN cp /out/*.row.tsv \
      inst/backends/linux-x86_64-glibc2.28.tsv \
    && Rscript -e \
      "install.packages('pak', repos = 'https://r-lib.github.io/p/pak/stable/'); options(repos = c(CRAN = 'https://packagemanager.posit.co/cran/__linux__/centos8/latest')); pak::local_install_deps('/build', dependencies = TRUE)" \
    && R CMD build . \
    && R CMD INSTALL --install-tests cuda.ml_*.tar.gz

RUN --network=none \
    CUDA_ML_CACHE_DIR=/tmp/cuda-ml-source-cache \
      CUDA_HOME=/opt/cuda.ml/managed-build/cuda-13.2.2-rapids-26.6.0 \
      CUML_PREFIX=/opt/cuda.ml/managed-build/cuda-13.2.2-rapids-26.6.0 \
      CUML_CUDA_ARCHITECTURES=75-real \
      CUDA_ML_CXX="$(command -v g++)" \
      Rscript -e \
        "library(cuda.ml); cuda_ml_install(source = TRUE); cuda_ml_install(source = TRUE)" \
    && test ! -e /tmp/cuda-ml-source-cache/runtime-v3 \
    && test ! -e /tmp/cuda-ml-source-cache/backend-assets-v1 \
    && test ! -e /tmp/cuda-ml-source-cache/backends-v3

RUN CUDA_ML_CACHE_DIR=/tmp/cuda-ml-source-cache \
      Rscript -e \
        "library(cuda.ml); info <- cuda_ml_backend_info(); stopifnot(identical(info\$backend, 'source'), identical(info\$build_mode, 'local'), info\$backend_available, info\$runtime_installed, !info\$backend_loaded, identical(info\$architectures, '75-real')); cuda_ml_runtime_audit()"

FROM base AS runtime

RUN uv venv --python /usr/bin/python3 /opt/cuda.ml/python \
    && uv pip install \
      --python /opt/cuda.ml/python/bin/python \
      scikit-learn

COPY --from=test-build /opt/R/library /opt/R/library
COPY --from=backend /out/*.tar.gz /opt/cuda.ml/backend/
COPY --from=backend /usr/local/bin/nvrtc-probe /usr/local/bin/nvrtc-probe

ENV CUDA_ML_BACKEND_MIRROR=file:///opt/cuda.ml/backend
ENV RETICULATE_PYTHON=/opt/cuda.ml/python/bin/python

RUN Rscript -e \
    "library(cuda.ml); info <- cuda_ml_backend_info(); stopifnot(identical(info\$backend, 'download'), info\$backend_available, !info\$runtime_installed, !info\$backend_loaded, reticulate::py_module_available('sklearn'))" \
    && test ! -e /usr/local/cuda \
    && test -z "$(command -v nvcc)"
