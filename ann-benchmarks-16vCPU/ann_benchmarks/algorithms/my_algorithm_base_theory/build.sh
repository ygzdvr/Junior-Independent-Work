#!/bin/bash
# Build and run the Docker container with HNSW algorithm analysis

# Error handling
set -e

echo "Cleaning up Docker resources..."
# Clean up any existing resources
docker system prune -f --volumes
docker rmi $(docker images -q) -f 2>/dev/null || true

echo "Building Docker image..."
# Build the Docker image with clean build
docker build --no-cache -t hnswlib-analyzer .

echo "Creating output directory..."
# Prepare output directory
mkdir -p ./analysis_results
rm -rf ./analysis_results/* 2>/dev/null || true

echo "Running analysis in Docker container..."
# Run with memory and CPU limits to prevent resource exhaustion
docker run --rm \
  -v $(pwd)/analysis_results:/output \
  --memory=120g \
  --cpus=15 \
  --name hnswlib-analysis-container \
  hnswlib-analyzer | tee ./analysis_results/container_output.log

echo "Analysis complete. Results are in ./analysis_results/"