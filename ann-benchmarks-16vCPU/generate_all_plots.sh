#!/bin/bash

# Script to generate all possible metric combinations from all available datasets

# Set defaults
COUNT=${1:-10}
PYTHON_PATH=${PYTHON_PATH:-"/home/ec2-user/ann-benchmarks/venv/bin/python"}

# Get the list of datasets from the results directory
# Look for directories in the results folder
DATASETS=()
echo "Discovering datasets in results folder..."
for dataset_dir in results/*/ ; do
  if [ -d "$dataset_dir" ]; then
    dataset_name=$(basename "$dataset_dir")
    DATASETS+=("$dataset_name")
    echo "Found dataset: $dataset_name"
  fi
done

# If no datasets found, exit
if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "No datasets found in results directory. Exiting."
  exit 1
fi

echo "Found ${#DATASETS[@]} datasets to process"

# Available metrics (from ann_benchmarks/plotting/metrics.py)
METRICS=(
  "k-nn"
  "rel"
  "qps"
  "p50"
  "p95"
  "p99"
  "p999"
  "queriessize"
)

# Calculate total operations
TOTAL_COMBINATIONS=$((${#METRICS[@]} * (${#METRICS[@]} - 1)))
TOTAL_OPERATIONS=$((TOTAL_COMBINATIONS * ${#DATASETS[@]}))
CURRENT=0

echo "Total operations to perform: $TOTAL_OPERATIONS"

# Process each dataset
for DATASET in "${DATASETS[@]}"; do
  echo ""
  echo "==================================="
  echo "Processing dataset: $DATASET"
  echo "==================================="
  
  OUTPUT_DIR="plots/$DATASET"
  
  # Create output directory
  mkdir -p "$OUTPUT_DIR"
  
  # Create a summary file
  SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
  echo "Metric plots for dataset: $DATASET" > "$SUMMARY_FILE"
  echo "Generated plots: " >> "$SUMMARY_FILE"
  
  # Iterate through all combinations of metrics
  for X_METRIC in "${METRICS[@]}"; do
    for Y_METRIC in "${METRICS[@]}"; do
      # Skip if metrics are the same
      if [ "$X_METRIC" == "$Y_METRIC" ]; then
        continue
      fi
      
      CURRENT=$((CURRENT + 1))
      OUTPUT_FILE="$OUTPUT_DIR/${X_METRIC}_vs_${Y_METRIC}.png"
      
      echo "[$CURRENT/$TOTAL_OPERATIONS] Dataset: $DATASET - Generating: $X_METRIC vs $Y_METRIC"
      
      # Determine appropriate scales based on metric type
      X_SCALE="linear"
      Y_SCALE="linear"
      
      # Performance metrics usually look better on log scale
      PERF_METRICS="qps build candidates distcomps queriessize indexsize"
      if [[ "$PERF_METRICS" == *"$X_METRIC"* ]]; then
        X_SCALE="log"
      fi
      
      if [[ "$PERF_METRICS" == *"$Y_METRIC"* ]]; then
        Y_SCALE="log"
      fi
      
      # Recall-based metrics usually look better on a2 or a3 scale
      RECALL_METRICS="k-nn epsilon largeepsilon"
      if [[ "$RECALL_METRICS" == *"$X_METRIC"* ]]; then
        if [[ "$Y_METRIC" == "qps" || "$Y_METRIC" == "build" ]]; then
          X_SCALE="a3"
        fi
      fi
      
      # Run plot.py with appropriate parameters
      echo "Running: sudo -E $PYTHON_PATH plot.py --dataset $DATASET --count $COUNT -x $X_METRIC -y $Y_METRIC -X $X_SCALE -Y $Y_SCALE -o $OUTPUT_FILE"
      sudo -E $PYTHON_PATH plot.py --dataset "$DATASET" --count "$COUNT" -x "$X_METRIC" -y "$Y_METRIC" -X "$X_SCALE" -Y "$Y_SCALE" -o "$OUTPUT_FILE"
      
      # Check if the plot was successfully generated
      if [ -f "$OUTPUT_FILE" ]; then
        echo "- $X_METRIC vs $Y_METRIC: $OUTPUT_FILE" >> "$SUMMARY_FILE"
        echo "  ✓ Plot saved to: $OUTPUT_FILE"
      else
        echo "  ✗ Failed to generate plot"
      fi
      
      # Add a small delay to prevent overwhelming the system
      sleep 0.5
    done
  done
  
  echo ""
  echo "Plotting complete for dataset $DATASET! All plots saved to $OUTPUT_DIR"
  
  # Generate an HTML index for easy viewing
  HTML_INDEX="$OUTPUT_DIR/index.html"
  echo "<!DOCTYPE html>
<html>
<head>
  <title>ANN Benchmark Plots for $DATASET</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { color: #333; }
    .plot-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
    .plot-card { border: 1px solid #ddd; border-radius: 4px; padding: 10px; }
    .plot-card img { width: 100%; height: auto; }
    .plot-card h3 { margin-top: 0; }
  </style>
</head>
<body>
  <h1>ANN Benchmark Plots for $DATASET</h1>
  <p>Generated on $(date)</p>
  <div class='plot-grid'>" > "$HTML_INDEX"

  # Add all plots to the HTML index
  for X_METRIC in "${METRICS[@]}"; do
    for Y_METRIC in "${METRICS[@]}"; do
      # Skip if metrics are the same
      if [ "$X_METRIC" == "$Y_METRIC" ]; then
        continue
      fi
      
      PLOT_FILE="${X_METRIC}_vs_${Y_METRIC}.png"
      if [ -f "$OUTPUT_DIR/$PLOT_FILE" ]; then
        echo "    <div class='plot-card'>
      <h3>$X_METRIC vs $Y_METRIC</h3>
      <img src='$PLOT_FILE' alt='$X_METRIC vs $Y_METRIC'>
    </div>" >> "$HTML_INDEX"
      fi
    done
  done

  echo "  </div>
</body>
</html>" >> "$HTML_INDEX"

  echo "HTML index generated at $HTML_INDEX"
done

# Create a master index file that links to all dataset pages
MASTER_INDEX="plots/index.html"
echo "<!DOCTYPE html>
<html>
<head>
  <title>ANN Benchmark Plots - All Datasets</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { color: #333; }
    .dataset-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
    .dataset-card { border: 1px solid #ddd; border-radius: 4px; padding: 15px; text-align: center; }
    .dataset-card:hover { background-color: #f5f5f5; }
    a { text-decoration: none; color: #0066cc; }
  </style>
</head>
<body>
  <h1>ANN Benchmark Plots - All Datasets</h1>
  <p>Generated on $(date)</p>
  <div class='dataset-list'>" > "$MASTER_INDEX"

# Add links to each dataset
for DATASET in "${DATASETS[@]}"; do
  echo "    <div class='dataset-card'>
    <a href='$DATASET/index.html'>
      <h2>$DATASET</h2>
      <p>View all plots</p>
    </a>
  </div>" >> "$MASTER_INDEX"
done

echo "  </div>
</body>
</html>" >> "$MASTER_INDEX"

echo ""
echo "==================================="
echo "All datasets processed successfully!"
echo "==================================="
echo "Master index page created at $MASTER_INDEX"
echo "Total plots generated: $CURRENT" 