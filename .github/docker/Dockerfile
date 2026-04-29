FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update -y && apt-get install -y \
    sudo software-properties-common dialog apt-utils \
    tzdata locales curl wget git \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev \
    libharfbuzz-dev libfribidi-dev libtiff5-dev libjpeg-dev \
    make gcc g++ pandoc python3 python3-pip

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

ARG CUML_VERSION=26.04
ENV CUML_VERSION=${CUML_VERSION}

# Cross-compile for T4 GPU (compute capability 7.5) since build runner has no GPU
ARG CMAKE_CUDA_ARCHITECTURES=75
ENV CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}

ENV NOT_CRAN=true

# Install R dependencies
RUN Rscript -e "\
    install.packages('pak', repos = 'https://r-lib.github.io/p/pak/devel/'); \
    pak::local_install_deps('/build', dependencies = TRUE)" \
    && rm -rf /tmp/* /root/.cache

# Install cuda.ml with tests
RUN R CMD INSTALL --install-tests /build

# Clean up
RUN rm -rf /tmp/* /build
