FROM ann-benchmarks

# Install required dependencies for building
RUN pip install pybind11 numpy setuptools wheel

# First, copy files from algorithm directory
COPY ann_benchmarks/algorithms/my_algorithm_base_dimred_128 /home/app/workdir/

# Debug the contents to confirm the structure
RUN echo "Contents of /home/app/workdir:" && ls -la /home/app/workdir/
RUN echo "Contents of src directory:" && ls -la /home/app/workdir/src/

# Build and install the algorithm from source
WORKDIR /home/app/workdir/src/python_bindings
RUN python3 setup.py install

# Verify installation
RUN python -c 'import hnswlib'

# Set working directory back to /home/app where run_algorithm.py is located
WORKDIR /home/app 