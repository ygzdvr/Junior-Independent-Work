# Junior-Independent-Work

This repository contains my junior independent work on approximate nearest neighbor (ANN) search algorithms, focusing on optimizing HNSW (Hierarchical Navigable Small World) graph algorithms for different computational environments.

Before going through the repository, I highly suggest to read the thesis first to better understand the goal and approach of the project.

## Repository Structure

### ann-benchmarks-16vCPU

This directory contains my custom HNSW engine implementation in C++ where I tested the theoretical bounds of:
- Graph insertion operations
- Search algorithm efficiency
- Index construction methodologies
- Memory usage optimization
- Query throughput

The implementation benchmarks various theoretical aspects of ANN algorithms to establish baseline performance metrics. This served as the foundation for the optimization work in the 64vCPU implementation.

### ann-benchmarks-64vCPU

This directory contains optimized versions of the HNSW algorithm using:
1. **Pruning techniques**: Testing different levels of graph pruning (from 1.25% to 10.00%)
   - Reducing the number of edges while maintaining search quality
   - Analyzing the impact on search performance and index size

2. **Dimensionality reduction**: Testing various reduction ratios (0.1 to 0.9)
   - Implementing efficient dimension reduction techniques for high-dimensional vectors
   - Evaluating the trade-offs between accuracy and performance

The 64vCPU implementation focuses on practical optimizations of the baseline algorithm when running on higher-core systems, showing how theoretical improvements translate to real-world performance gains.

## Datasets

The benchmarks use standard ANN benchmark datasets:
- SIFT-128-euclidean
- GIST-960-euclidean
- GloVe-100-angular
- GloVe-25-angular
- Fashion-MNIST-784-euclidean
- NYTimes-256-angular

## Methodology

Each algorithm variant is benchmarked on the following metrics:
- Query accuracy (recall)
- Query time
- Index build time
- Memory consumption
- Preprocessing overhead

## Running the Benchmarks

Each directory contains its own scripts for running the benchmarks. See the corresponding benchmark scripts in each directory:
- `benchmark.sh`: Standard benchmarking
- `benchmark_pruning.sh`: Benchmarks focused on edge pruning
- `benchmark_reduction.sh`: Benchmarks focused on dimension reduction
- `benchmark_diversity.sh`: Benchmarks focused on graph diversity
