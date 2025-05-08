#!/bin/bash

# Script to run benchmarks on all datasets sequentially
# Uses optimized parallelism settings based on dataset size

# Algorithm to benchmark - change this to run different algorithms
# ALGORITHM="hnsw(vespa)"
# ALGORITHM="annoy"
# ALGORITHM="hnswlib"
# ALGORITHM="hnsw(nmslib)"
ALGORITHM="my-algorithm-vanilla-opt-a2-1"
# ALGORITHM="hnsw(faiss)"
# Datasets from the image
#DATASETS=(
#  "glove-25-angular"
#  "sift-128-euclidean"
#  "nytimes-256-angular"
#  "fashion-mnist-784-euclidean"
#  "glove-100-angular"
#  "gist-960-euclidean"
#)

SMALL_DATASETS=(
  "glove-25-angular"
)

MEDIUM_DATASETS=(
#  "sift-128-euclidean"
#  "nytimes-256-angular"
)

LARGE_DATASETS=(
#  "fashion-mnist-784-euclidean"
#  "glove-100-angular"
)

VERY_LARGE_DATASETS=(
#  "gist-960-euclidean" 
)

# Create directories for logs and results
mkdir -p logs
mkdir -p logs/${ALGORITHM}

# Function to clear memory
clear_system_caches() {
  echo "Clearing system caches..."
  # Drop caches - requires sudo
  sudo sync
  sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
  
  # Wait a moment for system to stabilize
  sleep 5
  
  # Kill any stray Docker containers
  sudo docker ps -q | xargs -r sudo docker stop
  
  echo "System caches cleared"
}

# Function to run benchmark for a dataset with appropriate parallelism
run_benchmark() {
  local dataset=$1
  local parallelism=$2
  local max_algorithms=$3
  
  echo "======================================"
  echo "Starting benchmark for $dataset with $ALGORITHM (Parallelism: $parallelism, Max Algorithms: $max_algorithms)"
  echo "======================================"
  
  # Activate virtual environment 
  cd ~/ann-benchmarks
  source venv/bin/activate
  
  # Get the full path to the Python executable in the virtual environment
  PYTHON_PATH=$(which python)
  echo "Using Python from: $PYTHON_PATH"
  
  # Run the benchmark with appropriate parameters
  sudo -E $PYTHON_PATH run.py \
    --algorithm "$ALGORITHM" --force \
    --dataset $dataset \
    --parallelism $parallelism \
    --run-disabled \
    --timeout 7200
  
  # Generate plot for this dataset
  #sudo -E $PYTHON_PATH plot.py --dataset $dataset --x-scale log --y-scale log
  
  echo "Benchmark for $dataset with $ALGORITHM completed"
  echo "======================================"
}

# Main execution
echo "Starting sequential benchmarks for $ALGORITHM with optimized settings"
date

# Create a log file for this algorithm run
MAIN_LOG="logs/${ALGORITHM}/benchmark_run_$(date +%Y%m%d_%H%M%S).log"
echo "Starting benchmarks for $ALGORITHM at $(date)" | tee -a $MAIN_LOG

# Process small datasets with high parallelism 
for dataset in "${SMALL_DATASETS[@]}"; do
  clear_system_caches
  echo "Starting $dataset at $(date)" | tee -a $MAIN_LOG
  run_benchmark $dataset 15 5 2>&1 | tee -a logs/${ALGORITHM}/${dataset}.log
  echo "Completed $dataset at $(date)" | tee -a $MAIN_LOG
  sleep 10
done

# Process medium datasets with medium parallelism
for dataset in "${MEDIUM_DATASETS[@]}"; do
  clear_system_caches
  echo "Starting $dataset at $(date)" | tee -a $MAIN_LOG
  run_benchmark $dataset 15 5 2>&1 | tee -a logs/${ALGORITHM}/${dataset}.log
  echo "Completed $dataset at $(date)" | tee -a $MAIN_LOG
  sleep 10
done

# Process large datasets with lower parallelism
for dataset in "${LARGE_DATASETS[@]}"; do
  clear_system_caches
  echo "Starting $dataset at $(date)" | tee -a $MAIN_LOG
  run_benchmark $dataset 10 5 2>&1 | tee -a logs/${ALGORITHM}/${dataset}.log
  echo "Completed $dataset at $(date)" | tee -a $MAIN_LOG
  sleep 10
done

# Process very large datasets with minimal parallelism
for dataset in "${VERY_LARGE_DATASETS[@]}"; do
  clear_system_caches
  echo "Starting $dataset at $(date)" | tee -a $MAIN_LOG
  run_benchmark $dataset 10 5 2>&1 | tee -a logs/${ALGORITHM}/${dataset}.log
  echo "Completed $dataset at $(date)" | tee -a $MAIN_LOG
  sleep 10
done

echo "All benchmarks for $ALGORITHM completed" | tee -a $MAIN_LOG
date | tee -a $MAIN_LOG