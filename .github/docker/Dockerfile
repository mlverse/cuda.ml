FROM ubuntu:26.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update -y && apt-get install -y \
    sudo software-properties-common dialog apt-utils \
    tzdata locales curl git \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev \
    libharfbuzz-dev libfribidi-dev libtiff-dev libjpeg-dev \
    make gcc g++ binutils cmake pandoc \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install R via rig
RUN curl -L https://rig.r-pkg.org/deb/rig.gpg -o /etc/apt/trusted.gpg.d/rig.gpg \
    && echo "deb http://rig.r-pkg.org/deb rig main" > /etc/apt/sources.list.d/rig.list \
    && apt-get update \
    && apt-get install -y r-rig \
    && rig add release \
    && rig default release \
    && rm -rf /var/lib/apt/lists/*

# Use a fixed library path (not HOME-dependent) so packages are found
# regardless of what HOME is set to at runtime (GitHub Actions sets HOME=/github/home)
ENV R_LIBS_USER=/opt/R/library
RUN mkdir -p /opt/R/library

# Parallel compilation
RUN echo "MAKEFLAGS=-j$(nproc)" >> "$(R RHOME)/etc/Renviron.site"

# Copy source
COPY . /build

ENV CUDA_ML_BUILD_MODE=managed
ENV CUML_BOOTSTRAP_CACHE=/opt/cuda.ml

# Install R dependencies
RUN PKG_SYSREQS=false Rscript -e "\
    install.packages('pak', repos = 'https://r-lib.github.io/p/pak/devel/'); \
    pak::local_install_deps('/build', dependencies = TRUE); \
    stopifnot(requireNamespace('xgboost', quietly = TRUE))" \
    && rm -rf /tmp/* /root/.cache

# Install cuda.ml with tests
RUN R CMD INSTALL --install-tests /build
RUN Rscript -e "library(cuda.ml); info <- cuda_ml_backend_info(); stopifnot(identical(info\$backend, 'full'), identical(info\$build_mode, 'managed'), !info\$runtime_installed)"
RUN managed_prefix="$(Rscript -e "pkg_root <- function() '/build'; source('/build/tools/config/utils/artifacts.R'); source('/build/tools/config/utils/bootstrap.R'); cat(cuml_managed_bootstrap_prefix())")" \
    && LD_LIBRARY_PATH="${managed_prefix}/lib" \
        Rscript /build/tools/audit-backend.R "${managed_prefix}/bin/cuobjdump" \
    && cc -std=c11 -Wall -Wextra -Werror \
        -I"${managed_prefix}/include" \
        -L"${managed_prefix}/lib" \
        /build/tools/nvrtc-probe.c -lnvrtc \
        -o /usr/local/bin/nvrtc-probe

# Clean up
RUN rm -rf /tmp/* /build

FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y \
    ca-certificates curl tzdata locales \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev \
    libharfbuzz-dev libfribidi-dev libtiff-dev libjpeg-dev \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN curl -L https://rig.r-pkg.org/deb/rig.gpg -o /etc/apt/trusted.gpg.d/rig.gpg \
    && echo "deb http://rig.r-pkg.org/deb rig main" > /etc/apt/sources.list.d/rig.list \
    && apt-get update \
    && apt-get install -y r-rig \
    && rig add release \
    && rig default release \
    && rm -rf /var/lib/apt/lists/*

ENV R_LIBS_USER=/opt/R/library

RUN mkdir -p /opt/R/library
COPY --from=build /opt/R/library /opt/R/library
COPY --from=build /usr/local/bin/nvrtc-probe /usr/local/bin/nvrtc-probe

# The final image deliberately has no CUDA toolkit or prepared runtime cache.
RUN test ! -e /usr/local/cuda \
    && test -z "$(command -v nvcc)"
