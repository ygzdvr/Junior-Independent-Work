#!/bin/bash

# Generate PDF graphs using gnuplot from converted NumPy data
# This is a fixed version of the script to address gnuplot issues

echo "Converting NumPy data to gnuplot format..."
python convert_numpy_to_gnuplot.py

# Create output directory for PDF files
mkdir -p m_analysis/pdf_graphs

# Function to generate plots for a specific M value
generate_plots_for_m() {
    local m_value=$1
    local data_dir="m_analysis/gnuplot_data/m_${m_value}"
    local output_dir="m_analysis/pdf_graphs/m_${m_value}"
    
    # Check if data exists
    if [ ! -d "$data_dir" ]; then
        echo "No data found for M=${m_value}"
        return 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Read parameters
    if [ -f "${data_dir}/parameters.dat" ]; then
        local dim=$(grep "DIM" "${data_dir}/parameters.dat" | awk '{print $2}')
        echo "Generating plots for M=${m_value}, DIM=${dim}..."
    else
        echo "Parameter file not found for M=${m_value}"
        return 1
    fi
    
    # Generate construction time plot
    if [ -f "${data_dir}/construction.dat" ]; then
        echo "Generating construction time plot for M=${m_value}..."
        
        cat > "${output_dir}/gnuplot_construction.gp" <<EOF
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,6
set output "${output_dir}/construction_time_m${m_value}.pdf"
set title "Index Construction Time (M=${m_value}, DIM=${dim})" font "Helvetica,14"
set xlabel "Number of vectors"
set ylabel "Time (seconds)"
set logscale x
set logscale y
set grid
set key outside right top

# For O(n log n) reference curve
f(x) = a * x * log(x)
a = 5e-6

plot \\
EOF
        
        # Add EF=50 if it exists
        if [ $(head -1 ${data_dir}/construction.dat | wc -w) -ge 2 ]; then
            echo "    \"../../gnuplot_data/m_${m_value}/construction.dat\" using 1:2 with linespoints pt 7 ps 0.7 title \"EF=50\", \\" >> "${output_dir}/gnuplot_construction.gp"
        fi
        
        # Add additional EF values dynamically
        ef_values=(100 200 400 600 800 1000)
        col=3
        for ef in "${ef_values[@]}"; do
            if [ $col -le $(cat ${data_dir}/construction.dat | head -1 | wc -w) ]; then
                echo "    \"../../gnuplot_data/m_${m_value}/construction.dat\" using 1:${col} with linespoints pt 7 ps 0.7 title \"EF=${ef}\", \\" >> "${output_dir}/gnuplot_construction.gp"
                col=$((col + 1))
            fi
        done
        
        # Add the O(n log n) reference curve
        echo "    f(x) title \"O(n log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/gnuplot_construction.gp"
        
        # Run gnuplot
        gnuplot "${output_dir}/gnuplot_construction.gp"
    fi
    
    # Generate search time plot
    if [ -f "${data_dir}/search.dat" ]; then
        echo "Generating search time plot for M=${m_value}..."
        
        cat > "${output_dir}/gnuplot_search.gp" <<EOF
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,6
set output "${output_dir}/search_time_m${m_value}.pdf"
set title "Search Time vs Index Size (M=${m_value}, DIM=${dim})" font "Helvetica,14"
set xlabel "Index Size (number of vectors)"
set ylabel "Time per query (seconds)"
set logscale x
set logscale y
set grid
set key outside right top

# For O(log n) reference curve
g(x) = b * log(x)
b = 2e-6

plot \\
EOF
        
        # Add EF=50 if it exists
        if [ $(head -1 ${data_dir}/search.dat | wc -w) -ge 2 ]; then
            echo "    \"../../gnuplot_data/m_${m_value}/search.dat\" using 1:2 with linespoints pt 7 ps 0.7 title \"EF=50\", \\" >> "${output_dir}/gnuplot_search.gp"
        fi
        
        # Add additional EF values dynamically
        ef_values=(100 200 400 600 800 1000)
        col=3
        for ef in "${ef_values[@]}"; do
            if [ $col -le $(cat ${data_dir}/search.dat | head -1 | wc -w) ]; then
                echo "    \"../../gnuplot_data/m_${m_value}/search.dat\" using 1:${col} with linespoints pt 7 ps 0.7 title \"EF=${ef}\", \\" >> "${output_dir}/gnuplot_search.gp"
                col=$((col + 1))
            fi
        done
        
        # Add the O(log n) reference curve
        echo "    g(x) title \"O(log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/gnuplot_search.gp"
        
        # Run gnuplot
        gnuplot "${output_dir}/gnuplot_search.gp"
    fi
    
    # Generate insertion time plot
    if [ -f "${data_dir}/insertion.dat" ]; then
        echo "Generating insertion time plot for M=${m_value}..."
        
        cat > "${output_dir}/gnuplot_insertion.gp" <<EOF
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,6
set output "${output_dir}/insertion_time_m${m_value}.pdf"
set title "Insertion Time vs Index Size (M=${m_value}, DIM=${dim})" font "Helvetica,14"
set xlabel "Index size"
set ylabel "Time per insertion (seconds)"
set logscale x
set logscale y
set grid
set key outside right top

# For O(log n) reference curve
h(x) = c * log(x)
c = 1e-6

plot \\
EOF
        
        # Add EF=50 if it exists
        if [ $(head -1 ${data_dir}/insertion.dat | wc -w) -ge 2 ]; then
            echo "    \"../../gnuplot_data/m_${m_value}/insertion.dat\" using 1:2 with linespoints pt 7 ps 0.7 title \"EF=50\", \\" >> "${output_dir}/gnuplot_insertion.gp"
        fi
        
        # Add additional EF values dynamically
        ef_values=(100 200 400 600 800 1000)
        col=3
        for ef in "${ef_values[@]}"; do
            if [ $col -le $(cat ${data_dir}/insertion.dat | head -1 | wc -w) ]; then
                echo "    \"../../gnuplot_data/m_${m_value}/insertion.dat\" using 1:${col} with linespoints pt 7 ps 0.7 title \"EF=${ef}\", \\" >> "${output_dir}/gnuplot_insertion.gp"
                col=$((col + 1))
            fi
        done
        
        # Add the O(log n) reference curve
        echo "    h(x) title \"O(log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/gnuplot_insertion.gp"
        
        # Run gnuplot
        gnuplot "${output_dir}/gnuplot_insertion.gp"
    fi
    
    # Generate total insertion time plot
    if [ -f "${data_dir}/total_insertion.dat" ]; then
        echo "Generating total insertion time plot for M=${m_value}..."
        
        cat > "${output_dir}/gnuplot_total_insertion.gp" <<EOF
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,6
set output "${output_dir}/total_insertion_time_m${m_value}.pdf"
set title "Total Insertion Time (M=${m_value}, DIM=${dim})" font "Helvetica,14"
set xlabel "Number of vectors"
set ylabel "Total insertion time (seconds)"
set logscale x
set logscale y
set grid
set key outside right top

# For O(n log n) reference curve
j(x) = d * x * log(x)
d = 5e-6

plot \\
EOF
        
        # Add EF=50 if it exists
        if [ $(head -1 ${data_dir}/total_insertion.dat | wc -w) -ge 2 ]; then
            echo "    \"../../gnuplot_data/m_${m_value}/total_insertion.dat\" using 1:2 with linespoints pt 7 ps 0.7 title \"EF=50\", \\" >> "${output_dir}/gnuplot_total_insertion.gp"
        fi
        
        # Add additional EF values dynamically
        ef_values=(100 200 400 600 800 1000)
        col=3
        for ef in "${ef_values[@]}"; do
            if [ $col -le $(cat ${data_dir}/total_insertion.dat | head -1 | wc -w) ]; then
                echo "    \"../../gnuplot_data/m_${m_value}/total_insertion.dat\" using 1:${col} with linespoints pt 7 ps 0.7 title \"EF=${ef}\", \\" >> "${output_dir}/gnuplot_total_insertion.gp"
                col=$((col + 1))
            fi
        done
        
        # Add the O(n log n) reference curve
        echo "    j(x) title \"O(n log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/gnuplot_total_insertion.gp"
        
        # Run gnuplot
        gnuplot "${output_dir}/gnuplot_total_insertion.gp"
    fi
    
    echo "Plots for M=${m_value} generated in ${output_dir}"
}

# Create a single gnuplot script for comparing different M values
create_m_comparison_plots() {
    echo "Generating M value comparison plots..."
    local output_dir="m_analysis/pdf_graphs"
    
    # Find all M value directories
    local m_dirs=$(find m_analysis/gnuplot_data -maxdepth 1 -type d -name "m_*" | sort -V)
    
    if [ -z "$m_dirs" ]; then
        echo "No M value data found for comparison"
        return 1
    fi
    
    # Extract the dimension from the first M directory
    local first_dir=$(echo "$m_dirs" | head -1)
    local dim=256
    if [ -f "${first_dir}/parameters.dat" ]; then
        dim=$(grep "DIM" "${first_dir}/parameters.dat" | awk '{print $2}')
    fi
    
    # Create a list of M values
    local m_values=()
    for dir in $m_dirs; do
        m_values+=($(basename "$dir" | sed 's/m_//'))
    done
    
    # Compare construction times for M values (using EF=200)
    echo "Generating construction time comparison for different M values..."
    
    # Create gnuplot script
    cat > "${output_dir}/m_comparison_construction.gp" <<EOF
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,6
set output "${output_dir}/m_comparison_construction.pdf"
set title "Construction Time Comparison (DIM=${dim}, EF=200)" font "Helvetica,14"
set xlabel "Number of vectors"
set ylabel "Time (seconds)"
set logscale x
set logscale y
set grid
set key outside right top

plot \\
EOF
    
    # Add each M value to the plot
    for m in "${m_values[@]}"; do
        echo "    \"../gnuplot_data/m_${m}/construction.dat\" using 1:3 with linespoints pt 7 ps 0.7 title \"M=${m}\", \\" >> "${output_dir}/m_comparison_construction.gp"
    done
    
    # Add theoretical O(n log n) curve
    echo "    f(x) = a * x * log(x), \\" >> "${output_dir}/m_comparison_construction.gp"
    echo "    a = 5e-6, \\" >> "${output_dir}/m_comparison_construction.gp"
    echo "    f(x) title \"O(n log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/m_comparison_construction.gp"
    
    # Run gnuplot
    gnuplot "${output_dir}/m_comparison_construction.gp"
    
    # Compare search times for M values (using EF=200)
    echo "Generating search time comparison for different M values..."
    
    # Create gnuplot script
    cat > "${output_dir}/m_comparison_search.gp" <<EOF
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,6
set output "${output_dir}/m_comparison_search.pdf"
set title "Search Time Comparison (DIM=${dim}, EF=200)" font "Helvetica,14"
set xlabel "Index Size (number of vectors)"
set ylabel "Time per query (seconds)"
set logscale x
set logscale y
set grid
set key outside right top

plot \\
EOF
    
    # Add each M value to the plot
    for m in "${m_values[@]}"; do
        echo "    \"../gnuplot_data/m_${m}/search.dat\" using 1:3 with linespoints pt 7 ps 0.7 title \"M=${m}\", \\" >> "${output_dir}/m_comparison_search.gp"
    done
    
    # Add theoretical O(log n) curve
    echo "    g(x) = b * log(x), \\" >> "${output_dir}/m_comparison_search.gp"
    echo "    b = 2e-6, \\" >> "${output_dir}/m_comparison_search.gp"
    echo "    g(x) title \"O(log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/m_comparison_search.gp"
    
    # Run gnuplot
    gnuplot "${output_dir}/m_comparison_search.gp"
    
    # Compare insertion times for M values (using EF=200)
    echo "Generating insertion time comparison for different M values..."
    
    # Create gnuplot script
    cat > "${output_dir}/m_comparison_insertion.gp" <<EOF
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,6
set output "${output_dir}/m_comparison_insertion.pdf"
set title "Insertion Time Comparison (DIM=${dim}, EF=200)" font "Helvetica,14"
set xlabel "Index size"
set ylabel "Time per insertion (seconds)"
set logscale x
set logscale y
set grid
set key outside right top

plot \\
EOF
    
    # Add each M value to the plot
    for m in "${m_values[@]}"; do
        echo "    \"../gnuplot_data/m_${m}/insertion.dat\" using 1:3 with linespoints pt 7 ps 0.7 title \"M=${m}\", \\" >> "${output_dir}/m_comparison_insertion.gp"
    done
    
    # Add theoretical O(log n) curve
    echo "    h(x) = c * log(x), \\" >> "${output_dir}/m_comparison_insertion.gp"
    echo "    c = 1e-6, \\" >> "${output_dir}/m_comparison_insertion.gp"
    echo "    h(x) title \"O(log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/m_comparison_insertion.gp"
    
    # Run gnuplot
    gnuplot "${output_dir}/m_comparison_insertion.gp"
    
    echo "M comparison plots generated in ${output_dir}"
}

# Find all M values
echo "Finding available M values..."
m_values=$(find m_analysis/gnuplot_data -maxdepth 1 -type d -name "m_*" | sort -V | sed 's/.*m_//')

if [ -z "$m_values" ]; then
    echo "No data found for any M value. Run the analysis first."
    exit 1
fi

echo "Found data for M values: $m_values"

# Process each M value
for m in $m_values; do
    generate_plots_for_m "$m"
done

# Create comparison plots for different M values
create_m_comparison_plots

echo "All plots generated. PDF files are in m_analysis/pdf_graphs/" 