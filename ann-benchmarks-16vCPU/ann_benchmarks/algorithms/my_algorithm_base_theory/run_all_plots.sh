#!/bin/bash

# Create a directory for each M value and generate individual gnuplot scripts
create_m_plots() {
    local m_value=$1
    local output_dir="m_analysis/pdf_graphs/m_${m_value}"
    
    mkdir -p "$output_dir"
    
    # Get dimension from parameters.dat
    local dim=256
    if [ -f "m_analysis/gnuplot_data/m_${m_value}/parameters.dat" ]; then
        dim=$(grep "DIM" "m_analysis/gnuplot_data/m_${m_value}/parameters.dat" | awk '{print $2}')
    fi

    # Create search plot
    cat > "${output_dir}/plot_search.gp" <<EOF
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

    # Add data to the search plot
    for ef in 100 200 400 600 800 1000; do
        col=$(( (ef / 100) + 1 ))
        echo "    \"../../gnuplot_data/m_${m_value}/search.dat\" using 1:${col} with linespoints pt 7 ps 0.7 title \"EF=${ef}\", \\" >> "${output_dir}/plot_search.gp"
    done
    
    # Add reference curve
    echo "    g(x) title \"O(log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/plot_search.gp"

    # Create construction time plot
    cat > "${output_dir}/plot_construction.gp" <<EOF
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,6
set output "${output_dir}/construction_time_m${m_value}.pdf"
set title "Construction Time (M=${m_value}, DIM=${dim})" font "Helvetica,14"
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

    # Add data to the construction plot
    for ef in 100 200 400 600 800 1000; do
        col=$(( (ef / 100) + 1 ))
        echo "    \"../../gnuplot_data/m_${m_value}/construction.dat\" using 1:${col} with linespoints pt 7 ps 0.7 title \"EF=${ef}\", \\" >> "${output_dir}/plot_construction.gp"
    done
    
    # Add reference curve
    echo "    f(x) title \"O(n log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/plot_construction.gp"

    # Create insertion time plot
    cat > "${output_dir}/plot_insertion.gp" <<EOF
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

    # Add data to the insertion plot
    for ef in 100 200 400 600 800 1000; do
        col=$(( (ef / 100) + 1 ))
        echo "    \"../../gnuplot_data/m_${m_value}/insertion.dat\" using 1:${col} with linespoints pt 7 ps 0.7 title \"EF=${ef}\", \\" >> "${output_dir}/plot_insertion.gp"
    done
    
    # Add reference curve
    echo "    h(x) title \"O(log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/plot_insertion.gp"

    # Create total insertion time plot
    cat > "${output_dir}/plot_total_insertion.gp" <<EOF
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

    # Add data to the total insertion plot
    for ef in 100 200 400 600 800 1000; do
        col=$(( (ef / 100) + 1 ))
        echo "    \"../../gnuplot_data/m_${m_value}/total_insertion.dat\" using 1:${col} with linespoints pt 7 ps 0.7 title \"EF=${ef}\", \\" >> "${output_dir}/plot_total_insertion.gp"
    done
    
    # Add reference curve
    echo "    j(x) title \"O(n log n)\" with lines lt 7 lw 2 dashtype 2" >> "${output_dir}/plot_total_insertion.gp"
}

# Generate all gnuplot files for each M value
for m in 4 8 16 32 64 96 128 192 256 512; do
    echo "Creating gnuplot scripts for M=$m..."
    create_m_plots $m
done

# Create a simple wrapper script to run all gnuplot files
cat > run_gnuplot.sh <<EOF
#!/bin/bash

# Run all gnuplot scripts
for m in 4 8 16 32 64 96 128 192 256 512; do
    cd m_analysis/pdf_graphs/m_\$m
    echo "Running gnuplot scripts for M=\$m..."
    gnuplot plot_search.gp
    gnuplot plot_construction.gp
    gnuplot plot_insertion.gp
    gnuplot plot_total_insertion.gp
    cd ../../../
done

echo "All plots generated!"
EOF

chmod +x run_gnuplot.sh

# Create a single gnuplot file specifically for m=4
cat > m4_all_plots.gp <<EOF
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,6
set output "m4_search_time.pdf"
set title "Search Time vs Index Size (M=4, DIM=256)" font "Helvetica,14"
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
     "m_analysis/gnuplot_data/m_4/search.dat" using 1:3 with linespoints pt 7 ps 0.7 title "EF=100", \\
     "m_analysis/gnuplot_data/m_4/search.dat" using 1:4 with linespoints pt 7 ps 0.7 title "EF=200", \\
     "m_analysis/gnuplot_data/m_4/search.dat" using 1:5 with linespoints pt 7 ps 0.7 title "EF=400", \\
     "m_analysis/gnuplot_data/m_4/search.dat" using 1:6 with linespoints pt 7 ps 0.7 title "EF=600", \\
     "m_analysis/gnuplot_data/m_4/search.dat" using 1:7 with linespoints pt 7 ps 0.7 title "EF=800", \\
     "m_analysis/gnuplot_data/m_4/search.dat" using 1:8 with linespoints pt 7 ps 0.7 title "EF=1000", \\
     g(x) title "O(log n)" with lines lt 7 lw 2 dashtype 2
EOF

echo "Scripts created. To generate all plots, run:"
echo "./run_gnuplot.sh"
echo ""
echo "Or to generate a single plot for M=4, run:"
echo "gnuplot m4_all_plots.gp" 