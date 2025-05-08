FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# COPY ann_benchmarks/algorithms/my_algorithm_base_opq/src/requirements.txt . # Removed, file doesn't exist
# RUN pip install --no-cache-dir -r requirements.txt # Removed, no requirements file

# Install Faiss (CPU version). For GPU, use faiss-gpu and ensure CUDA toolkit is available
RUN pip install --no-cache-dir faiss-cpu

# Copy algorithm source code
COPY ann_benchmarks/algorithms/my_algorithm_base_opq/src /workspace/src

# Build and install the C++ library/Python bindings
RUN cd /workspace/src/python_bindings && pip install .

# Copy the module code (ann_benchmarks uses this Python file directly)
COPY ann_benchmarks/algorithms/my_algorithm_base_opq/module.py /workspace/ann_benchmarks/algorithms/my_algorithm_base_opq/

# Set environment variable for Python path
# Ensure Python can find both the installed package and the module file's directory
ENV PYTHONPATH=/workspace:/usr/local/lib/python3.9/site-packages:/workspace/ann_benchmarks/algorithms/my_algorithm_base_opq

# Default command (optional, can be overridden)
CMD ["python"] 