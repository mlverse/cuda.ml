FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update -y && apt-get install -y \
    sudo software-properties-common dialog apt-utils \
    tzdata locales curl wget git \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev \
    libharfbuzz-dev libfribidi-dev libtiff-dev libjpeg-dev \
    make gcc g++ binutils cmake pandoc python3 python3-pip python3-venv \
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

ENV NOT_CRAN=true
ENV UNIVERSE_NAME=mlverse
ENV CUML_BOOTSTRAP_CACHE=/opt/cuda.ml
ENV CUML_MANAGED_PREFIX=/opt/cuda.ml/managed-build/cuda-13.2.2-rapids-26.6.0
ENV CUML_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual

# Install R dependencies
RUN PKG_SYSREQS=false Rscript -e "\
    install.packages('pak', repos = 'https://r-lib.github.io/p/pak/devel/'); \
    pak::local_install_deps('/build', dependencies = TRUE)" \
    && rm -rf /tmp/* /root/.cache

# Install cuda.ml with tests
RUN R CMD INSTALL --install-tests /build
RUN Rscript -e "library(cuda.ml); stopifnot(has_cuML())"
RUN Rscript /build/tools/audit-backend.R \
    /opt/cuda.ml/managed-build/cuda-13.2.2-rapids-26.6.0/bin/cuobjdump
RUN cc -std=c11 -Wall -Wextra -Werror \
    -I"${CUML_MANAGED_PREFIX}/include" \
    -L"${CUML_MANAGED_PREFIX}/lib" \
    /build/tools/nvrtc-probe.c -lnvrtc \
    -o /usr/local/bin/nvrtc-probe

# Clean up
RUN rm -rf /tmp/* /build

FROM ubuntu:24.04

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
ENV NOT_CRAN=true

RUN mkdir -p /opt/R/library
COPY --from=build /opt/R/library /opt/R/library
COPY --from=build /usr/local/bin/nvrtc-probe /usr/local/bin/nvrtc-probe

RUN Rscript -e "library(cuda.ml); stopifnot(has_cuML())" \
    && test ! -e /usr/local/cuda \
    && test -z "$(command -v nvcc)"
