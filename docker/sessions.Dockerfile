FROM ghcr.io/iknl/infrastructure/algorithm-base:idea4rc-5.0

# This is a placeholder that should be overloaded by invoking
# docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="v6-sessions"


# rpy2's default (API) build requires an `R` executable on PATH (or R_HOME).
# The base image contains R, but it may not expose `R` during build.
# ABI mode avoids a hard build-time dependency on `R`.
ENV RPY2_CFFI_MODE=ABI

# Base image already contains R; don't hardcode R_HOME, as layouts differ.
# If R is not on PATH in the base image, set RPY2_R_BINARY accordingly.

# install federated algorithm
COPY v6-idea4rc-common /deps/v6-idea4rc-common
RUN pip install /deps/v6-idea4rc-common
COPY v6-sessions /app
RUN pip install /app


# Set environment variable to make name of the package available within the
# docker image.
ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `wrap_algorithm()` when the image is run. This function
# will ensure that the algorithm method is called properly.
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
