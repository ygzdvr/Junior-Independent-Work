#!/bin/bash

# Run all gnuplot scripts
for m in 4 8 16 32 64 96 128 192 256 512; do
    echo "Running gnuplot scripts for M=$m..."
    cd m_analysis/pdf_graphs/m_$m
    
    # Fix the output paths for each script
    sed -i 's|m_analysis/pdf_graphs/m_'$m'/|./|g' plot_search.gp
    sed -i 's|m_analysis/pdf_graphs/m_'$m'/|./|g' plot_construction.gp
    sed -i 's|m_analysis/pdf_graphs/m_'$m'/|./|g' plot_insertion.gp
    sed -i 's|m_analysis/pdf_graphs/m_'$m'/|./|g' plot_total_insertion.gp
    
    # Run the scripts
    gnuplot plot_search.gp
    gnuplot plot_construction.gp
    gnuplot plot_insertion.gp
    gnuplot plot_total_insertion.gp
    cd ../../../
done

echo "All plots generated!"
