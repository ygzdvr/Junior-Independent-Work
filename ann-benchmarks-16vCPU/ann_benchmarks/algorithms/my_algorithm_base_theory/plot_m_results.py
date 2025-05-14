#!/usr/bin/env python3

import json
import numpy as np
import os
import tempfile
import subprocess

def load_data(filename):
    """Load data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def generate_comparison_plots(data_files, m_values):
    """Generate plots comparing different M values from the intermediate results files"""
    # List to store data from each M value
    datasets = []
    
    # Load all datasets
    for data_file in data_files:
        datasets.append(load_data(data_file))
    
    # Colors for different M values
    colors = ['#332288', '#88CCEE', '#117733',]  # deep purple, light blue
    point_types = [1, 2, 3]
    
    # Create a temporary directory for data files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create data files for each metric and M value
        data_file_paths = {}
        
        for i, (dataset, m_value) in enumerate(zip(datasets, m_values)):
            # Extract data
            sizes = dataset['sizes']
            construction_times = dataset['construction_times']
            search_times = dataset['search_times']
            insertion_times = dataset['insertion_times']
            memory_usages = dataset['memory_usages']
            
            # Create and write data files
            construction_data_file = os.path.join(temp_dir, f'construction_data_M{m_value}.txt')
            search_data_file = os.path.join(temp_dir, f'search_data_M{m_value}.txt')
            insertion_data_file = os.path.join(temp_dir, f'insertion_data_M{m_value}.txt')
            memory_data_file = os.path.join(temp_dir, f'memory_data_M{m_value}.txt')
            
            # Store file paths
            data_file_paths[f'construction_M{m_value}'] = construction_data_file
            data_file_paths[f'search_M{m_value}'] = search_data_file
            data_file_paths[f'insertion_M{m_value}'] = insertion_data_file
            data_file_paths[f'memory_M{m_value}'] = memory_data_file
            
            # Write construction time data
            with open(construction_data_file, 'w') as f:
                for size, time in zip(sizes, construction_times):
                    f.write(f"{size} {time}\n")
            
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
        
        # Create gnuplot scripts for each type of plot
        plot_configs = [
            # Log-log plots
            {
                'output_file': 'construction_time_vs_sizes_log_m_comparison.pdf',
                'title': 'Construction Time vs Dataset Size (Log-Log)',
                'y_label': 'Construction Time (s)',
                'data_type': 'construction',
                'logscale_x': True,
                'logscale_y': True
            },
            {
                'output_file': 'search_time_vs_sizes_log_m_comparison.pdf',
                'title': 'Search Time vs Dataset Size (Log-Log)',
                'y_label': 'Search Time (s)',
                'data_type': 'search',
                'logscale_x': True,
                'logscale_y': True
            },
            {
                'output_file': 'insertion_time_vs_sizes_log_m_comparison.pdf',
                'title': 'Insertion Time vs Dataset Size (Log-Log)',
                'y_label': 'Insertion Time (s)',
                'data_type': 'insertion',
                'logscale_x': True,
                'logscale_y': True
            },
            {
                'output_file': 'memory_usage_vs_sizes_log_m_comparison.pdf',
                'title': 'Memory Usage vs Dataset Size (Log-Log)',
                'y_label': 'Memory Usage (GB)',
                'data_type': 'memory',
                'logscale_x': True,
                'logscale_y': True
            },
            
            # Linear plots
            {
                'output_file': 'construction_time_vs_sizes_linear_m_comparison.pdf',
                'title': 'Construction Time vs Dataset Size',
                'y_label': 'Construction Time (s)',
                'data_type': 'construction',
                'logscale_x': False,
                'logscale_y': False
            },
            {
                'output_file': 'search_time_vs_sizes_linear_m_comparison.pdf',
                'title': 'Search Time vs Dataset Size',
                'y_label': 'Search Time (s)',
                'data_type': 'search',
                'logscale_x': False,
                'logscale_y': False
            },
            {
                'output_file': 'insertion_time_vs_sizes_linear_m_comparison.pdf',
                'title': 'Insertion Time vs Dataset Size',
                'y_label': 'Insertion Time (s)',
                'data_type': 'insertion',
                'logscale_x': False,
                'logscale_y': False
            },
            {
                'output_file': 'memory_usage_vs_sizes_linear_m_comparison.pdf',
                'title': 'Memory Usage vs Dataset Size',
                'y_label': 'Memory Usage (GB)',
                'data_type': 'memory',
                'logscale_x': False,
                'logscale_y': False
            },
            
            # Semi-log plots (log x, linear y)
            {
                'output_file': 'memory_usage_vs_sizes_semi_log_m_comparison.pdf',
                'title': 'Memory Usage vs Dataset Size (Log X-axis)',
                'y_label': 'Memory Usage (GB)',
                'data_type': 'memory',
                'logscale_x': True,
                'logscale_y': False
            },
            {
                'output_file': 'search_time_vs_sizes_semi_log_m_comparison.pdf',
                'title': 'Search Time vs Dataset Size (Log X-axis)',
                'y_label': 'Search Time (s)',
                'data_type': 'search',
                'logscale_x': True,
                'logscale_y': False
            },
            {
                'output_file': 'insertion_time_vs_sizes_semi_log_m_comparison.pdf',
                'title': 'Insertion Time vs Dataset Size (Log X-axis)',
                'y_label': 'Insertion Time (s)',
                'data_type': 'insertion',
                'logscale_x': True,
                'logscale_y': False
            }
        ]
        
        # Generate all plots
        for config in plot_configs:
            # Create gnuplot script for this plot
            gnuplot_script = f"""
set terminal pdf enhanced color font 'Helvetica,12'
set output '{config['output_file']}'
set title '{config['title']}'
set xlabel 'Dataset Size'
set ylabel '{config['y_label']}'
"""
            
            # Set scales based on configuration
            if config['logscale_x']:
                gnuplot_script += "set logscale x\n"
            else:
                gnuplot_script += "unset logscale x\n"
                
            if config['logscale_y']:
                gnuplot_script += "set logscale y\n"
            else:
                gnuplot_script += "unset logscale y\n"
            
            gnuplot_script += """
set grid
set key top left

# Auto-scale to fit data exactly
set autoscale xfix
set autoscale yfix
set offsets 0, 0, 0, 0

plot """
            
            # Add plot commands for each M value
            plot_commands = []
            for i, m_value in enumerate(m_values):
                data_file_key = f"{config['data_type']}_M{m_value}"
                plot_command = f"'{data_file_paths[data_file_key]}' using 1:2 with linespoints \\\n" + \
                               f"     linecolor rgb '{colors[i]}' \\\n" + \
                               f"     pointtype {point_types[i]} \\\n" + \
                               f"     linewidth 2 \\\n" + \
                               f"     pointsize 1 \\\n" + \
                               f"     title 'M = {m_value}'"
                plot_commands.append(plot_command)
            
            gnuplot_script += ", \\\n".join(plot_commands)
            
            # Create and execute gnuplot script
            script_file = os.path.join(temp_dir, f"plot_{config['output_file']}.gp")
            with open(script_file, 'w') as f:
                f.write(gnuplot_script)
            
            subprocess.run(['gnuplot', script_file], check=True)
            
            print(f"Generated: {config['output_file']}")

if __name__ == "__main__":
    # Paths to data files for different M values
    data_files = [
        "analysis_results_M_24/intermediate_results_7.json",
        "analysis_results_M_48/intermediate_results_7.json",
        "analysis_results_M_96/intermediate_results_7.json"
    ]
    
    # Corresponding M values
    m_values = [24, 48, 96]
    
    generate_comparison_plots(data_files, m_values)
    print(f"All comparison plots generated from: {', '.join(data_files)}") 