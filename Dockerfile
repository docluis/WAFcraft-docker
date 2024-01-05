# Use a Python 3.8 base image
FROM python:3.8-bullseye

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    vim \
    git \
    g++ \
    apt-utils \
    autoconf \
    automake \
    libcurl4-openssl-dev \
    libgeoip-dev \
    liblmdb-dev \
    libpcre2-dev \
    libtool \
    libxml2-dev \
    libyajl-dev \
    pkgconf \
    zlib1g-dev

# Copy only the necessary files for modsecurity installation
COPY ./modsecurity-v3.0.10 /app/modsecurity-v3.0.10

# Install libmodsecurity manually
RUN cd /app/modsecurity-v3.0.10 && \
    ./build.sh && \
    ./configure && \
    make && \
    make install

# Set the environment variables
ENV LD_LIBRARY_PATH /usr/local/modsecurity/lib:${LD_LIBRARY_PATH}

# Install the necessary Python packages
RUN pip install pybind11 typer

# Copy only the necessary files for pymodsecurity installation
COPY pymodsecurity /app/pymodsecurity

# Install pymodsecurity manually
RUN cd /app/pymodsecurity && \
    python setup.py install

# Copy modsecurity-cli
COPY modsecurity-cli /app/modsecurity-cli

# Copy coreruleset into modsecurity-cli
COPY coreruleset /app/modsecurity-cli/coreruleset

# Copy ml-modsec
COPY ml-modsec /app/ml-modsec

# Cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the default command for the container
CMD ["bash"]
