FROM python:3.9-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy only what's needed for building hnswlib
COPY src/ /build/src/

# Install build dependencies
RUN pip install --no-cache-dir pybind11 wheel setuptools numpy matplotlib tqdm psutil hnswlib

# Build hnswlib
#WORKDIR /build/src/python_bindings
#RUN python3 setup.py install

# Set working directory
WORKDIR /app

# Verify installation
RUN python -c 'import hnswlib; print("hnswlib installed successfully")'

# Copy only the script
COPY complexity_analysis.py /app/

# Create output directory
RUN mkdir -p /app/results /output

# Set environment variables with HNSW parameters
# Pushing parameters as high as possible
ENV M_VALUE=96 \
    EF_CONSTRUCTION=500 \
    EF_SEARCH=500 \
    DIMENSION=128 \
    VECTOR_COUNT=1000000 \
    PYTHONUNBUFFERED=1 \
    DEBUG=1

# Add a wrapper script for capturing errors
RUN echo '#!/bin/bash \n\
set -e \n\
echo "Starting analysis with debug output enabled" \n\
python -u complexity_analysis.py \n\
echo "Analysis completed successfully" \n\
' > /app/run.sh && chmod +x /app/run.sh

# Default command with lower memory batch processing and debug output
CMD ["/app/run.sh"] 