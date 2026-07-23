FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update -y && apt-get install -y \
    sudo software-properties-common dialog apt-utils \
    tzdata locales curl wget git \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libfreetype6-dev libpng-dev \
    libharfbuzz-dev libfribidi-dev libtiff-dev libjpeg-dev \
    make gcc g++ cmake pandoc python3 python3-pip python3-venv \
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

# Install R dependencies. The package itself is installed on the GPU runner.
RUN PKG_SYSREQS=false Rscript -e "\
    install.packages('pak', repos = 'https://r-lib.github.io/p/pak/devel/'); \
    pak::local_install_deps('/build', dependencies = TRUE)" \
    && rm -rf /tmp/* /root/.cache
