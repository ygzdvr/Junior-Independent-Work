#!/usr/bin/env python3

import json
import numpy as np
import os
import tempfile
import subprocess
import math

def load_data(filename):
    """Load data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def generate_plots(data_file):
    """Generate plots from the intermediate results file"""
    # Load data
    data = load_data(data_file)
    
    # Use the same color and point type for all plots
    color = '#332288'  # deep purple
    point_type = 1
    
    # Extract data
    sizes = data['sizes']
    construction_times = data['construction_times']
    search_times = data['search_times']
    insertion_times = data['insertion_times']
    memory_usages = data['memory_usages']
    
    # Create a temporary directory for data files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create data files
        construction_data_file = os.path.join(temp_dir, 'construction_data.txt')
        search_data_file = os.path.join(temp_dir, 'search_data.txt')
        insertion_data_file = os.path.join(temp_dir, 'insertion_data.txt')
        memory_data_file = os.path.join(temp_dir, 'memory_data.txt')
        construction_nlogn_data_file = os.path.join(temp_dir, 'construction_nlogn_data.txt')
        
        # Write construction time data
        with open(construction_data_file, 'w') as f:
            for size, time in zip(sizes, construction_times):
                f.write(f"{size} {time}\n")
        
        # Write construction time data with n*log(n) for x-axis
        with open(construction_nlogn_data_file, 'w') as f:
            for size, time in zip(sizes, construction_times):
                nlogn = size * math.log10(size)  # Using base-10 log
                f.write(f"{nlogn} {time}\n")
        
        # Write search time data
        with open(search_data_file, 'w') as f:
            for size, time in zip(sizes, search_times):
                f.write(f"{size} {time}\n")
        
        # Write insertion time data (note: one less entry than sizes)
        with open(insertion_data_file, 'w') as f:
            for size, time in zip(sizes[:-1], insertion_times):
                f.write(f"{size} {time}\n")
        
        # Write memory usage data
        with open(memory_data_file, 'w') as f:
            for size, usage in zip(sizes, memory_usages):
                f.write(f"{size} {usage}\n")
        
        # Create gnuplot scripts
        # Log-log plots
        log_log_plot_configs = [
            {
                'data_file': construction_data_file,
                'output_file': 'construction_time_vs_sizes_log.pdf',
                'title': 'Construction Time vs Dataset Size (Log-Log)',
                'y_label': 'Construction Time (s)',
                'legend': 'HNSW Index Construction'
            },
            {
                'data_file': search_data_file,
                'output_file': 'search_time_vs_sizes_log.pdf',
                'title': 'Search Time vs Dataset Size (Log-Log)',
                'y_label': 'Search Time (s)',
                'legend': 'HNSW Query Performance'
            },
            {
                'data_file': insertion_data_file,
                'output_file': 'insertion_time_vs_sizes_log.pdf',
                'title': 'Insertion Time vs Dataset Size (Log-Log)',
                'y_label': 'Insertion Time (s)',
                'legend': 'HNSW Insertion Performance'
            },
            {
                'data_file': memory_data_file,
                'output_file': 'memory_usage_vs_sizes_log.pdf',
                'title': 'Memory Usage vs Dataset Size (Log-Log)',
                'y_label': 'Memory Usage (GB)',
                'legend': 'HNSW Memory Consumption'
            }
        ]
        
        # Linear-linear plots (non-log axis)
        linear_plot_configs = [
            {
                'data_file': construction_data_file,
                'output_file': 'construction_time_vs_sizes_linear.pdf',
                'title': 'Construction Time vs Dataset Size',
                'y_label': 'Construction Time (s)',
                'legend': 'HNSW Index Construction'
            },
            {
                'data_file': search_data_file,
                'output_file': 'search_time_vs_sizes_linear.pdf',
                'title': 'Search Time vs Dataset Size',
                'y_label': 'Search Time (s)',
                'legend': 'HNSW Query Performance'
            },
            {
                'data_file': insertion_data_file,
                'output_file': 'insertion_time_vs_sizes_linear.pdf',
                'title': 'Insertion Time vs Dataset Size',
                'y_label': 'Insertion Time (s)',
                'legend': 'HNSW Insertion Performance'
            },
            {
                'data_file': memory_data_file,
                'output_file': 'memory_usage_vs_sizes_linear.pdf',
                'title': 'Memory Usage vs Dataset Size',
                'y_label': 'Memory Usage (GB)',
                'legend': 'HNSW Memory Consumption'
            }
        ]
        
        # Semi-log plot for memory usage (log x-axis, linear y-axis)
        memory_semi_log_config = {
            'data_file': memory_data_file,
            'output_file': 'memory_usage_vs_sizes_semi_log.pdf',
            'title': 'Memory Usage vs Dataset Size (Log X-axis)',
            'y_label': 'Memory Usage (GB)',
            'legend': 'HNSW Memory Consumption'
        }
        
        # Semi-log plots for search and insertion time (log x-axis, linear y-axis)
        search_semi_log_config = {
            'data_file': search_data_file,
            'output_file': 'search_time_vs_sizes_semi_log.pdf',
            'title': 'Search Time vs Dataset Size (Log X-axis)',
            'y_label': 'Search Time (s)',
            'legend': 'HNSW Query Performance'
        }
        
        insertion_semi_log_config = {
            'data_file': insertion_data_file,
            'output_file': 'insertion_time_vs_sizes_semi_log.pdf',
            'title': 'Insertion Time vs Dataset Size (Log X-axis)',
            'y_label': 'Insertion Time (s)',
            'legend': 'HNSW Insertion Performance'
        }
        
        # NLogN plot for construction time
        construction_nlogn_config = {
            'data_file': construction_nlogn_data_file,
            'output_file': 'construction_time_vs_nlogn.pdf',
            'title': 'Construction Time vs N·log(N)',
            'x_label': 'N·log(N)',
            'y_label': 'Construction Time (s)',
            'legend': 'HNSW Index Construction'
        }
        
        # Generate log-log plots
        for config in log_log_plot_configs:
            gnuplot_script = f"""
set terminal pdf enhanced color font 'Helvetica,12'
set output '{config['output_file']}'
set title '{config['title']}'
set xlabel 'Dataset Size'
set ylabel '{config['y_label']}'
set logscale x
set logscale y
set grid
set key top left

# Auto-scale to fit data exactly
set autoscale xfix
set autoscale yfix
set offsets 0, 0, 0, 0

plot '{config['data_file']}' using 1:2 with linespoints \
     linecolor rgb '{color}' \
     pointtype {point_type} \
     linewidth 2 \
     pointsize 1 \
     title '{config['legend']}'
"""
            # Create temporary gnuplot script file
            script_file = os.path.join(temp_dir, f"plot_{config['output_file']}.gp")
            with open(script_file, 'w') as f:
                f.write(gnuplot_script)
            
            # Execute gnuplot
            subprocess.run(['gnuplot', script_file], check=True)
            
            print(f"Generated: {config['output_file']}")
        
        # Generate linear-linear plots
        for config in linear_plot_configs:
            gnuplot_script = f"""
set terminal pdf enhanced color font 'Helvetica,12'
set output '{config['output_file']}'
set title '{config['title']}'
set xlabel 'Dataset Size'
set ylabel '{config['y_label']}'
set grid
set key top left

# Auto-scale to fit data exactly
set autoscale xfix
set autoscale yfix
set offsets 0, 0, 0, 0

plot '{config['data_file']}' using 1:2 with linespoints \
     linecolor rgb '{color}' \
     pointtype {point_type} \
     linewidth 2 \
     pointsize 1 \
     title '{config['legend']}'
"""
            # Create temporary gnuplot script file
            script_file = os.path.join(temp_dir, f"plot_{config['output_file']}.gp")
            with open(script_file, 'w') as f:
                f.write(gnuplot_script)
            
            # Execute gnuplot
            subprocess.run(['gnuplot', script_file], check=True)
            
            print(f"Generated: {config['output_file']}")
        
        # Generate semi-log plot for memory usage
        gnuplot_script = f"""
set terminal pdf enhanced color font 'Helvetica,12'
set output '{memory_semi_log_config['output_file']}'
set title '{memory_semi_log_config['title']}'
set xlabel 'Dataset Size'
set ylabel '{memory_semi_log_config['y_label']}'
set logscale x
unset logscale y
set grid
set key top left

# Auto-scale to fit data exactly
set autoscale xfix
set autoscale yfix
set offsets 0, 0, 0, 0

plot '{memory_semi_log_config['data_file']}' using 1:2 with linespoints \
     linecolor rgb '{color}' \
     pointtype {point_type} \
     linewidth 2 \
     pointsize 1 \
     title '{memory_semi_log_config['legend']}'
"""
        # Create temporary gnuplot script file
        script_file = os.path.join(temp_dir, f"plot_{memory_semi_log_config['output_file']}.gp")
        with open(script_file, 'w') as f:
            f.write(gnuplot_script)
        
        # Execute gnuplot
        subprocess.run(['gnuplot', script_file], check=True)
        
        print(f"Generated: {memory_semi_log_config['output_file']}")
        
        # Generate semi-log plot for search time
        gnuplot_script = f"""
set terminal pdf enhanced color font 'Helvetica,12'
set output '{search_semi_log_config['output_file']}'
set title '{search_semi_log_config['title']}'
set xlabel 'Dataset Size'
set ylabel '{search_semi_log_config['y_label']}'
set logscale x
unset logscale y
set grid
set key top left

# Auto-scale to fit data exactly
set autoscale xfix
set autoscale yfix
set offsets 0, 0, 0, 0

plot '{search_semi_log_config['data_file']}' using 1:2 with linespoints \
     linecolor rgb '{color}' \
     pointtype {point_type} \
     linewidth 2 \
     pointsize 1 \
     title '{search_semi_log_config['legend']}'
"""
        # Create temporary gnuplot script file
        script_file = os.path.join(temp_dir, f"plot_{search_semi_log_config['output_file']}.gp")
        with open(script_file, 'w') as f:
            f.write(gnuplot_script)
        
        # Execute gnuplot
        subprocess.run(['gnuplot', script_file], check=True)
        
        print(f"Generated: {search_semi_log_config['output_file']}")
        
        # Generate semi-log plot for insertion time
        gnuplot_script = f"""
set terminal pdf enhanced color font 'Helvetica,12'
set output '{insertion_semi_log_config['output_file']}'
set title '{insertion_semi_log_config['title']}'
set xlabel 'Dataset Size'
set ylabel '{insertion_semi_log_config['y_label']}'
set logscale x
unset logscale y
set grid
set key top left

# Auto-scale to fit data exactly
set autoscale xfix
set autoscale yfix
set offsets 0, 0, 0, 0

plot '{insertion_semi_log_config['data_file']}' using 1:2 with linespoints \
     linecolor rgb '{color}' \
     pointtype {point_type} \
     linewidth 2 \
     pointsize 1 \
     title '{insertion_semi_log_config['legend']}'
"""
        # Create temporary gnuplot script file
        script_file = os.path.join(temp_dir, f"plot_{insertion_semi_log_config['output_file']}.gp")
        with open(script_file, 'w') as f:
            f.write(gnuplot_script)
        
        # Execute gnuplot
        subprocess.run(['gnuplot', script_file], check=True)
        
        print(f"Generated: {insertion_semi_log_config['output_file']}")
        
        # Generate n*log(n) plot for construction time
        gnuplot_script = f"""
set terminal pdf enhanced color font 'Helvetica,12'
set output '{construction_nlogn_config['output_file']}'
set title '{construction_nlogn_config['title']}'
set xlabel '{construction_nlogn_config['x_label']}'
set ylabel '{construction_nlogn_config['y_label']}'
set grid
set key top left

# Auto-scale to fit data exactly
set autoscale xfix
set autoscale yfix
set offsets 0, 0, 0, 0

plot '{construction_nlogn_config['data_file']}' using 1:2 with linespoints \
     linecolor rgb '{color}' \
     pointtype {point_type} \
     linewidth 2 \
     pointsize 1 \
     title '{construction_nlogn_config['legend']}'
"""
        # Create temporary gnuplot script file
        script_file = os.path.join(temp_dir, f"plot_{construction_nlogn_config['output_file']}.gp")
        with open(script_file, 'w') as f:
            f.write(gnuplot_script)
        
        # Execute gnuplot
        subprocess.run(['gnuplot', script_file], check=True)
        
        print(f"Generated: {construction_nlogn_config['output_file']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "analysis_results_M_48/intermediate_results_7.json"
    
    generate_plots(data_file)
    print(f"All plots generated from: {data_file}") 