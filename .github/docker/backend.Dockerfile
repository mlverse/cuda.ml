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
COPY inst/build-tools/ /build/inst/build-tools/
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
RUN /opt/cuda.ml/managed-build/cuda-13.2.2-rapids-26.6.0/cmake/bin/cmake \
      --build inst/backend-src/.cmake-build \
      --target cuda.ml \
      --parallel 2
COPY inst/nvforest-cpu-src/ /build/inst/nvforest-cpu-src/
RUN CUDA_ML_PREFIX=/opt/cuda.ml/managed-build/cuda-13.2.2-rapids-26.6.0 \
    && R_INCLUDE_DIR="$(Rscript -e 'cat(R.home("include"))')" \
    && RCPP_INCLUDE_DIR="$(Rscript -e 'cat(system.file("include", package = "Rcpp"))')" \
    && "${CUDA_ML_PREFIX}/cmake/bin/cmake" \
      -S inst/nvforest-cpu-src \
      -B inst/nvforest-cpu-src/.cmake-build \
      -GNinja \
      -DCMAKE_MAKE_PROGRAM="${CUDA_ML_PREFIX}/bin/ninja" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="$(command -v g++)" \
      -DR_INCLUDE_DIR="${R_INCLUDE_DIR}" \
      -DRCPP_INCLUDE_DIR="${RCPP_INCLUDE_DIR}" \
      -DRAPIDS_INCLUDE_DIR="${CUDA_ML_PREFIX}/include" \
      -DRAPIDS_LIB_DIR="${CUDA_ML_PREFIX}/lib" \
    && "${CUDA_ML_PREFIX}/cmake/bin/cmake" \
      --build inst/nvforest-cpu-src/.cmake-build \
      --target cuda.ml.nvforest \
      --parallel 2

FROM backend-build AS backend

COPY LICENSE.md /build/LICENSE.md
COPY inst/COPYRIGHTS /build/inst/COPYRIGHTS
COPY inst/third-party/ /build/inst/third-party/
COPY inst/nvforest-native-symbols.txt /build/inst/
COPY tools/audit-backend.R tools/audit-nvforest-cpu-backend.R tools/package-backend.R tools/package-nvforest-cpu-backend.R tools/nvrtc-probe.c /build/tools/

RUN CUDA_ML_PREFIX="$(Rscript -e \
      "cuml_artifact_root <- function() '/build/inst/artifacts'; source('inst/build-tools/artifacts.R'); source('inst/build-tools/bootstrap.R'); cat(cuml_managed_bootstrap_prefix())")" \
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

RUN mkdir -p /out/full /out/cpu \
    && Rscript tools/package-backend.R \
      inst/backend-src/.cmake-build/cuda.ml.so \
      /out/full \
      "${SOURCE_COMMIT}" \
      "${BACKEND_BASE_URL}" \
    && Rscript tools/package-nvforest-cpu-backend.R \
      inst/nvforest-cpu-src/.cmake-build/cuda.ml.nvforest.so \
      /out/cpu \
      "${SOURCE_COMMIT}" \
      "${BACKEND_BASE_URL}" \
    && mkdir /tmp/cuda-ml-nvforest-cpu-audit \
    && tar -xzf /out/cpu/*.tar.gz \
      -C /tmp/cuda-ml-nvforest-cpu-audit \
    && Rscript tools/audit-nvforest-cpu-backend.R \
      /tmp/cuda-ml-nvforest-cpu-audit/cuda.ml.nvforest.so \
    && rm -rf /tmp/cuda-ml-nvforest-cpu-audit

FROM scratch AS export
COPY --from=backend /out /

FROM backend AS test-build

COPY . /build

# The CRAN jobs build the vignette; this image exercises native functionality
# and intentionally does not install Pandoc.
RUN cp /out/full/*.row.tsv \
      inst/backends/linux-x86_64-glibc2.28.tsv \
    && cp /out/cpu/*.row.tsv \
      inst/nvforest-backends/linux-x86_64-glibc2.28-nvforest-cpu.tsv \
    && Rscript -e \
      "install.packages('pak', repos = 'https://r-lib.github.io/p/pak/stable/'); options(repos = c(CRAN = 'https://packagemanager.posit.co/cran/__linux__/centos8/latest')); pak::local_install_deps('/build', dependencies = TRUE)" \
    && R CMD build --no-build-vignettes . \
    && R CMD INSTALL --install-tests cuda.ml_*.tar.gz

RUN --network=none \
    CUDA_ML_CACHE_DIR=/tmp/cuda-ml-source-cache \
      CUML_BOOTSTRAP_CACHE=/opt/cuda.ml \
      CUDA_ML_CXX="$(command -v g++)" \
      Rscript -e \
        "library(cuda.ml); cuda_ml_install(source = TRUE, architectures = '75-real'); cuda_ml_install(source = TRUE, architectures = '75-real')" \
    && test ! -e /tmp/cuda-ml-source-cache/source-toolchains-v1 \
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
COPY --from=backend /out/full/*.tar.gz /opt/cuda.ml/backend/
COPY --from=backend /out/cpu/*.tar.gz /opt/cuda.ml/backend/
COPY --from=backend /usr/local/bin/nvrtc-probe /usr/local/bin/nvrtc-probe

ENV CUDA_ML_BACKEND_MIRROR=file:///opt/cuda.ml/backend
ENV RETICULATE_PYTHON=/opt/cuda.ml/python/bin/python

RUN Rscript -e \
    "library(cuda.ml); info <- cuda_ml_backend_info(); stopifnot(identical(info\$backend, 'download'), info\$backend_available, !info\$runtime_installed, !info\$backend_loaded, reticulate::py_module_available('sklearn'))" \
    && test ! -e /usr/local/cuda \
    && test -z "$(command -v nvcc)"
