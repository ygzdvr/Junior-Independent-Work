#!/bin/bash

# Script to run benchmarks for my_algorithm on small datasets
# Based on benchmark_all_datasets.sh but customized for my_algorithm

# Set the algorithm name (with underscore, not hyphen)
ALGORITHM="my_algorithm"

# Small dataset to run on
DATASETS=(
  "glove-25-angular"
)

# Create directories for logs and results
mkdir -p logs
mkdir -p logs/${ALGORITHM}

# Function to build the Docker image for the algorithm
build_algorithm_image() {
  echo "Building Docker image for $ALGORITHM..."
  
  # Activate virtual environment 
  cd ~/ann-benchmarks
  source venv/bin/activate
  
  # Get the full path to the Python executable in the virtual environment
  PYTHON_PATH=$(which python)
  
  # Build the Docker image
  sudo -E $PYTHON_PATH install.py --algorithm $ALGORITHM
  
  echo "Docker image build for $ALGORITHM completed"
}

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
    --algorithm "$ALGORITHM" \
    --dataset $dataset \
    --parallelism $parallelism \
    --run-disabled \
    --timeout 7200
  
  # Generate plot for this dataset
  sudo -E $PYTHON_PATH plot.py --dataset $dataset --x-scale log --y-scale log
  
  echo "Benchmark for $dataset with $ALGORITHM completed"
  echo "======================================"
}

# Function to generate all possible plots for a dataset
generate_all_plots() {
  local dataset=$1
  
  echo "======================================"
  echo "Generating all plots for $dataset with $ALGORITHM"
  echo "======================================"
  
  # Activate virtual environment 
  cd ~/ann-benchmarks
  source venv/bin/activate
  
  # Get the full path to the Python executable in the virtual environment
  PYTHON_PATH=$(which python)
  
  # Run the custom plot generation script
  ./generate_all_plots.sh $dataset
  
  echo "Plot generation for $dataset with $ALGORITHM completed"
  echo "======================================"
}

# Main execution
echo "Starting benchmarks for $ALGORITHM on small datasets"
date

# Create a log file for this algorithm run
MAIN_LOG="logs/${ALGORITHM}/benchmark_run_$(date +%Y%m%d_%H%M%S).log"
echo "Starting benchmarks for $ALGORITHM at $(date)" | tee -a $MAIN_LOG

# First build the Docker image
echo "Building Docker image at $(date)" | tee -a $MAIN_LOG
build_algorithm_image 2>&1 | tee -a logs/${ALGORITHM}/docker_build.log
echo "Docker image built at $(date)" | tee -a $MAIN_LOG

# Process datasets with high parallelism (small datasets perform well with high parallelism)
for dataset in "${DATASETS[@]}"; do
  clear_system_caches
  echo "Starting $dataset benchmark at $(date)" | tee -a $MAIN_LOG
  run_benchmark $dataset 15 5 2>&1 | tee -a logs/${ALGORITHM}/${dataset}.log
  echo "Completed $dataset benchmark at $(date)" | tee -a $MAIN_LOG
  
  echo "Starting plot generation for $dataset at $(date)" | tee -a $MAIN_LOG
  generate_all_plots $dataset 2>&1 | tee -a logs/${ALGORITHM}/${dataset}_plots.log
  echo "Completed plot generation for $dataset at $(date)" | tee -a $MAIN_LOG
  
  sleep 10
done

echo "All benchmarks and plots for $ALGORITHM completed" | tee -a $MAIN_LOG
date | tee -a $MAIN_LOG

# Print instructions for viewing results
echo ""
echo "To view the detailed plots, open: plots/${DATASETS[0]}/index.html" 