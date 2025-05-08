#!/bin/bash

# Script to run benchmarks on all datasets with multiple algorithms in parallel
# Optimized for a 64 vCPU machine with CPU affinity

# Define algorithms to benchmark with their docker tags - ALGORITHM,DOCKER_TAG format


# docker tag ann-benchmarks-my_algorithm_prune_1_25 ann-benchmarks-my-algorithm-prune-1-25
# docker tag ann-benchmarks-my_algorithm_prune_7_50 ann-benchmarks-my-algorithm-prune-7-50
# docker tag ann-benchmarks-my_algorithm_prune_10_00 ann-benchmarks-my-algorithm-prune-10-00
#docker tag ann-benchmarks-my_algorithm_base_prune_1_15 ann-benchmarks-my-algorithm-base-prune-1-15

ALGORITHMS=(
  "my-algorithm-prune-2-50,ann-benchmarks-my-algorithm-prune-2-50"
  "my-algorithm-prune-5-00,ann-benchmarks-my-algorithm-prune-5-00"
)

SMALL_DATASETS=(
#  "glove-25-angular"
)

MEDIUM_DATASETS=(
  "sift-128-euclidean"
#  "nytimes-256-angular"
)

LARGE_DATASETS=(
  "fashion-mnist-784-euclidean"
#  "glove-100-angular"
)

VERY_LARGE_DATASETS=(
#  "gist-960-euclidean" 
)

# Total available CPUs (adjust based on your machine)
TOTAL_CPUS=64
# Reserve CPUs for system processes
RESERVED_CPUS=4
# CPUs per algorithm - ideally should match the number of Docker containers each algorithm will spawn
CPUS_PER_ALGORITHM=20

# Create directories for logs and results
mkdir -p logs

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

# Function to run benchmark for a dataset with an algorithm using specific Docker tag
run_benchmark() {
  local dataset=$1
  local algorithm=$2
  local docker_tag=$3
  local cpu_range=$4
  
  echo "======================================"
  echo "Starting benchmark for $dataset with $algorithm (Docker: $docker_tag, CPU Range: $cpu_range)"
  echo "======================================"
  
  # Activate virtual environment 
  cd ~/ann-benchmarks
  source venv/bin/activate
  
  # Get the full path to the Python executable in the virtual environment
  PYTHON_PATH=$(which python)
  echo "Using Python from: $PYTHON_PATH"
  
  # Create algorithm log directory if it doesn't exist
  mkdir -p logs/${algorithm}
  
  # Use taskset to bind the process to specific CPU range
  taskset -c $cpu_range sudo -E $PYTHON_PATH run.py \
    --algorithm "$algorithm" \
    --docker-tag "$docker_tag" \
    --force \
    --dataset $dataset \
    --parallelism 15 \
    --run-disabled \
    --timeout 7200
  
  echo "Benchmark for $dataset with $algorithm completed"
  echo "======================================"
}

# Function to run benchmarks for a dataset with all algorithms in parallel
run_parallel_benchmarks() {
  local datasets=("$@")
  
  # Process datasets sequentially
  for dataset in "${datasets[@]}"; do
    clear_system_caches
    
    echo "Processing dataset: $dataset"
    
    # Track running processes
    pids=()
    
    # CPU counter - start after reserved CPUs
    cpu_start=$RESERVED_CPUS
    
    # Start all algorithms in parallel, each with dedicated CPU range
    for algo_entry in "${ALGORITHMS[@]}"; do
      # Split the entry by comma to get algorithm name and docker tag
      IFS=',' read -r algorithm docker_tag <<< "${algo_entry}"
      
      # Calculate CPU range
      cpu_end=$((cpu_start + CPUS_PER_ALGORITHM - 1))
      if [ $cpu_end -ge $TOTAL_CPUS ]; then
        # Wrap around if we exceed total CPUs
        cpu_end=$((cpu_end - TOTAL_CPUS + RESERVED_CPUS))
        # Create CPU range with wrap-around
        cpu_range="${cpu_start}-$((TOTAL_CPUS-1)),$RESERVED_CPUS-$cpu_end"
      else
        # Standard CPU range
        cpu_range="${cpu_start}-${cpu_end}"
      fi
      
      mkdir -p logs/${algorithm}
      log_file="logs/${algorithm}/benchmark_${dataset}_$(date +%Y%m%d_%H%M%S).log"
      echo "Starting $dataset with $algorithm using $docker_tag at $(date) on CPUs $cpu_range" | tee -a $log_file
      
      # Run this algorithm with its dedicated CPU range
      run_benchmark "$dataset" "$algorithm" "$docker_tag" "$cpu_range" > $log_file 2>&1 &
      
      # Store process ID
      pids+=($!)
      
      # Move to next CPU range
      cpu_start=$((cpu_end + 1))
      
      # If we've reached the CPU limit, cycle back to after reserved CPUs
      if [ $cpu_start -ge $TOTAL_CPUS ]; then
        cpu_start=$RESERVED_CPUS
      fi
    done
    
    # Wait for all background processes to complete
    for pid in "${pids[@]}"; do
      wait $pid
    done
    
    echo "All algorithms completed for dataset $dataset"
    sleep 10
  done
}

# Main execution
echo "Starting parallel benchmarks for all algorithms with CPU affinity"
date

# Create a main log file
MAIN_LOG="logs/parallel_benchmark_run_$(date +%Y%m%d_%H%M%S).log"
echo "Starting benchmarks at $(date)" | tee -a $MAIN_LOG

# Process dataset groups one at a time, but with algorithms in parallel
echo "Processing SMALL datasets" | tee -a $MAIN_LOG
run_parallel_benchmarks "${SMALL_DATASETS[@]}"

echo "Processing MEDIUM datasets" | tee -a $MAIN_LOG
run_parallel_benchmarks "${MEDIUM_DATASETS[@]}"

echo "Processing LARGE datasets" | tee -a $MAIN_LOG
run_parallel_benchmarks "${LARGE_DATASETS[@]}"

echo "Processing VERY_LARGE datasets" | tee -a $MAIN_LOG
run_parallel_benchmarks "${VERY_LARGE_DATASETS[@]}"

echo "All benchmarks completed" | tee -a $MAIN_LOG
date | tee -a $MAIN_LOG