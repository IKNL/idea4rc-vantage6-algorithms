FROM ghcr.io/iknl/infrastructure/algorithm-ohdsi-base:idea4rc-5.0

# This is a placeholder that should be overloaded by invoking
# docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="v6-sessions"


# Ensure the image provides a compatible R on PATH (/usr/bin/R).
RUN apt-get update \
    && apt-get install -y --no-install-recommends r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# install federated algorithm
COPY v6-idea4rc-common /deps/v6-idea4rc-common
RUN pip install --upgrade pip && pip install /deps/v6-idea4rc-common
COPY v6-sessions /app
RUN pip install /app


# Set environment variable to make name of the package available within the
# docker image.
ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `wrap_algorithm()` when the image is run. This function
# will ensure that the algorithm method is called properly.
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
