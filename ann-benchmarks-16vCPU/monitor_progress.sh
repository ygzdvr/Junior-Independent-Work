#!/bin/bash

# Check if the Docker container is still running
echo "===== Docker Container Status ====="
docker ps | grep hnsw-analysis || echo "Container not running. Analysis may be complete or failed."

# Check the log file
echo -e "\n===== Recent Log Activity ====="
tail -20 ~/analysis.log

# Check output files
echo -e "\n===== Output Directory Status ====="
ls -la ~/analysis_results/m_analysis 2>/dev/null || echo "No output directory detected yet"

# If m_analysis directory exists, check files
if [ -d ~/analysis_results/m_analysis ]; then
    echo -e "\n===== Analysis Progress ====="
    echo "M value directories:"
    ls -la ~/analysis_results/m_analysis/ | grep "^d" | wc -l
    
    # Check for PNG files (completed analyses)
    echo "Completed plots:"
    find ~/analysis_results/m_analysis -name "*.png" | wc -l
fi

echo -e "\n===== System Resource Usage ====="
top -b -n 1 | head -20 