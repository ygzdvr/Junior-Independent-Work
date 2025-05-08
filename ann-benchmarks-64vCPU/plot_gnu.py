#!/usr/bin/env python3

import argparse
import numpy as np
import tempfile
import os
import sys
import subprocess
from pygnuplot import gnuplot

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import (compute_metrics, create_linestyles,
                                          create_pointset, get_plot_label)
from ann_benchmarks.results import get_unique_algorithms, load_all_results

# List of algorithms to include in the plot
# Only these algorithms will be plotted if they exist in the dataset
# If empty, all algorithms will be plotted
ALGORITHMS_TO_PLOT = [
    "my-algorithm-prune-10-00",
    "my-algorithm-prune-7-50",
    "my-algorithm-prune-5-00",
    "my-algorithm-prune-2-50",
#    "my-algorithm-prune-1-25",
    "my-algorithm-base",
#    "my-algorithm-prune",
#    "my-algorithm-diversity-tight",
#    "my-algorithm-diversity-loose",
#    "hnsw(faiss)",
#    "hnsw(vespa)",
#    "hnswlib",
#    "my-algorithm-reduction-0-2",
#    "my-algorithm-reduction-0-3",
#    "my-algorithm-reduction-0-4",
#    "my-algorithm-reduction-0-6",
#    "my-algorithm-reduction-0-8",
]

def check_gnuplot_installed():
    """Check if gnuplot is installed on the system"""
    try:
        result = subprocess.run(['which', 'gnuplot'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False


def create_plot_gnu(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, batch):
    # First check if gnuplot is installed
    if not check_gnuplot_installed():
        print("Error: gnuplot is not installed on your system.")
        print("Please install gnuplot using your package manager:")
        print("  - For Ubuntu/Debian: sudo apt-get install gnuplot")
        print("  - For RHEL/CentOS/Amazon Linux: sudo yum install gnuplot")
        print("  - For macOS: brew install gnuplot")
        sys.exit(1)
        
    xm, ym = (metrics[xn], metrics[yn])
    
    # Create a temporary directory in a location that's accessible when running with sudo
    temp_dir = os.path.abspath("./temp_gnuplot_data")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    print(f"Using temporary directory: {temp_dir}")
    
    data_files = []
    
    try:
        # Filter algorithms to plot if the global list is not empty
        if ALGORITHMS_TO_PLOT:
            filtered_data = {}
            for algo in ALGORITHMS_TO_PLOT:
                if algo in all_data:
                    filtered_data[algo] = all_data[algo]
                    print(f"Including algorithm: {algo}")
                else:
                    print(f"Warning: Requested algorithm '{algo}' not found in dataset")
            
            if not filtered_data:
                print("Error: None of the specified algorithms were found in the dataset")
                print(f"Available algorithms: {list(all_data.keys())}")
                sys.exit(1)
                
            all_data = filtered_data
        
        # Sort algorithms by mean y-value for consistent legend order
        def mean_y(algo):
            xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
            return -np.log(np.array(ys)).mean() if len(ys) > 0 else float('inf')
        
        # Find data ranges to set tight axes
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        has_data = False
        
        for algo in all_data.keys():
            xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
            if len(xs) > 0:
                has_data = True
                if x_scale != 'log' and x_scale != 'logit' and x_scale[0] != 'a':
                    min_x = min(min_x, min(xs))
                    max_x = max(max_x, max(xs))
                else:
                    # For log scales, only consider positive values
                    pos_xs = [x for x in xs if x > 0]
                    if pos_xs:
                        min_x = min(min_x, min(pos_xs))
                        max_x = max(max_x, max(pos_xs))
                
                if y_scale != 'log' and y_scale != 'logit':
                    min_y = min(min_y, min(ys))
                    max_y = max(max_y, max(ys))
                else:
                    # For log scales, only consider positive values
                    pos_ys = [y for y in ys if y > 0]
                    if pos_ys:
                        min_y = min(min_y, min(pos_ys))
                        max_y = max(max_y, max(ys))
        
        if not has_data:
            print("No data points found to plot!")
            return
            
        # Define an academic color scheme (ColorBrewer-inspired)
        # These colors are colorblind-friendly and work well in publications
        academic_colors = [
            '#332288',  # deep purple
            '#88CCEE',  # light blue
            '#44AA99',  # teal
            '#117733',  # dark green
            '#CC6677',  # rose
            '#882255',  # wine
            '#AA4499',  # violet
        ]
        
        # Academic point types that are distinguishable in print
        point_types = [1, 2, 7, 5, 9, 11, 13, 3, 4]  # + (plus), x (cross), filled circle, filled square, filled triangle, filled diamond, etc.
        
        # Set up the GnuPlot script - create a new instance
        try:
            g = gnuplot.Gnuplot()
        except Exception as e:
            print(f"Error initializing gnuplot: {e}")
            print("Make sure gnuplot is installed and accessible in your PATH.")
            sys.exit(1)
        
        # Configure the plot
        g.cmd('reset')
        g.cmd('set terminal pdfcairo enhanced color font "Times-Roman,10" size 12cm,9cm dashed')
        g.cmd(f'set output "{os.path.abspath(fn_out)}"')
        
        # Set labels and title
        g.cmd(f'set xlabel "{xm["description"]}" font "Times-Roman,12"')
        g.cmd(f'set ylabel "{ym["description"]}" font "Times-Roman,12"')
        g.cmd(f'set title "{get_plot_label(xm, ym)}" font "Times-Roman,14" center')
        
        # Set grid and key (legend)
        g.cmd('set grid lc rgb "#dddddd" lt 1')
        g.cmd('set key inside bottom left font "Times-Roman,8" spacing 0.8')
        
        # Set scales and add minor tics for better readability
        if x_scale == 'log':
            g.cmd('set logscale x')
            g.cmd('set format x "10^{%L}"')
            g.cmd('set mxtics 10')  # 10 minor tics between major tics
        elif x_scale == 'logit':
            g.cmd('set logscale x')  # Approximation - gnuplot doesn't have true logit scale
            g.cmd('set mxtics 10')
        elif x_scale[0] == 'a':
            # For custom a-scale, we'll transform data directly
            alpha = float(x_scale[1:])
            g.cmd('set mxtics 5')
        else:
            g.cmd('set mxtics 5')  # 5 minor tics for linear scale
        
        if y_scale == 'log':
            g.cmd('set logscale y')
            g.cmd('set format y "10^{%L}"')
            g.cmd('set mytics 10')
        elif y_scale == 'logit':
            g.cmd('set logscale y')  # Approximation
            g.cmd('set mytics 10')
        else:
            g.cmd('set mytics 5')
        
        # Add some padding to avoid clipping data points at the edges
        x_padding = (max_x - min_x) * 0.05
        y_padding = (max_y - min_y) * 0.02  # Reduced padding for y-axis
        
        # Set axis limits to fit the data tightly
        if "lim" in xm and x_scale != "logit":
            x0, x1 = xm["lim"]
            g.cmd(f'set xrange [{max(x0, min_x - x_padding)}:{min(x1, max_x + x_padding)}]')
        elif x_scale == "logit":
            g.cmd(f'set xrange [{min_x * 0.95}:{max_x * 1.05}]')
        else:
            if min_x != float('inf') and max_x != float('-inf'):
                g.cmd(f'set xrange [{min_x - x_padding}:{max_x + x_padding}]')
        
        if "lim" in ym:
            y0, y1 = ym["lim"]
            g.cmd(f'set yrange [{max(y0, min_y * 0.9)}:{min(y1, max_y * 1.1)}]')
        else:
            if min_y != float('inf') and max_y != float('-inf'):
                g.cmd(f'set yrange [{min_y * 0.9}:{max_y * 1.1}]')
        
        # Add more tick marks on the x-axis
        if x_scale == 'log':
            # For log scale, show more intermediate tick values
            g.cmd('set xtics 2')  # Show tick mark at every power of 2
            g.cmd('set mxtics 2')  # Show minor tick between each major tick
            g.cmd('set format x "%.1f"')  # Use decimal format instead of exponents
            
            # Create custom tics for better readability
            if min_x < 0.2 and max_x > 0.9:
                g.cmd('set xtics add ("0.2" 0.2, "0.4" 0.4, "0.6" 0.6, "0.8" 0.8, "1.0" 1.0)')
            elif min_x < 0.5 and max_x > 0.9:
                g.cmd('set xtics add ("0.5" 0.5, "0.7" 0.7, "0.9" 0.9)')
            elif max_x > 10:
                # Add more labels for larger ranges
                custom_tics = []
                for i in range(2, int(max_x), 2):
                    if i < max_x:
                        custom_tics.append(f'"{i}" {i}')
                if custom_tics:
                    g.cmd('set xtics add (' + ', '.join(custom_tics) + ')')
        else:
            # Add more explicit ticks depending on the data range
            x_range = max_x - min_x
            if x_range < 0.1:
                g.cmd('set xtics 0.01')
            elif x_range < 0.5:
                g.cmd('set xtics 0.05')
            elif x_range < 1:
                g.cmd('set xtics 0.1')
            elif x_range < 5:
                g.cmd('set xtics 0.25')
            elif x_range < 10:
                g.cmd('set xtics 0.5')
            else:
                step = max(1, int(x_range / 10))
                g.cmd(f'set xtics {step}')
            
            # Create custom tics for specific ranges that make sense in nearest-neighbor search
            if min_x < 0.05 and max_x > 0.95:
                # If we're showing a 0-1 range (common for recall), add explicit tics at rounded values
                custom_tics = []
                for i in range(1, 10):
                    val = i/10
                    if val > min_x and val < max_x:
                        custom_tics.append(f'"{val:.1f}" {val}')
                if custom_tics:
                    g.cmd('set xtics add (' + ', '.join(custom_tics) + ')')
        
        # Ensure tic marks are visible
        g.cmd('set tics front')
        
        # Write data to temp files and prepare plot command
        plot_commands = []
        
        # Get sorted algorithms
        sorted_algos = sorted(all_data.keys(), key=mean_y)
        
        for i, algo in enumerate(sorted_algos):
            xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
            
            # Skip if no data
            if len(xs) == 0:
                continue
                
            # Apply custom scale transformation if needed
            if x_scale[0] == 'a':
                alpha = float(x_scale[1:])
                def transform_x(x):
                    return 1 - (1 - x) ** (1 / alpha)
                xs = [transform_x(x) for x in xs]
            
            # Write data to temp file and ensure it's properly formatted with non-zero sizes
            data_file = os.path.join(temp_dir, f"data_{i}.dat")
            with open(data_file, 'w') as f:
                for x, y in zip(xs, ys):
                    f.write(f"{x} {y}\n")
            
            # Ensure file permissions are appropriate
            os.chmod(data_file, 0o644)
            
            # Check if the file was written successfully and has content
            if os.path.getsize(data_file) > 0:
                data_files.append(data_file)
                print(f"Data file created: {data_file} (size: {os.path.getsize(data_file)} bytes)")
            else:
                print(f"Warning: Empty data file for {algo}")
                continue
            
            # Assign color and point type from our academic palette (cycle if more algorithms than colors)
            color_index = i % len(academic_colors)
            point_index = i % len(point_types)
            
            color = academic_colors[color_index]
            point_type = point_types[point_index]
            
            # Add to plot commands (use absolute path)
            abs_data_file = os.path.abspath(data_file)
            line_width = 2
            plot_commands.append(
                f'"{abs_data_file}" using 1:2 with linespoints ' +
                f'linewidth {line_width} pointtype {point_type} pointsize 0.5 ' +
                f'linecolor rgb "{color}" ' +
                f'title "{algo}"'
            )
            
            # Add raw data if requested
            if raw and len(axs) > 0:
                raw_data_file = os.path.join(temp_dir, f"raw_data_{i}.dat")
                with open(raw_data_file, 'w') as f:
                    for x, y in zip(axs, ays):
                        f.write(f"{x} {y}\n")
                # Ensure file permissions
                os.chmod(raw_data_file, 0o644)
                data_files.append(raw_data_file)
                
                # Use same color but lighter for raw data
                abs_raw_data_file = os.path.abspath(raw_data_file)
                plot_commands.append(
                    f'"{abs_raw_data_file}" using 1:2 with linespoints ' +
                    f'linewidth 1 pointtype {point_type} pointsize 0.3 ' +
                    f'linecolor rgb "{color}80" ' +  # Add 80 (half transparency) to color hex
                    f'title ""'  # No legend entry for raw data
                )
        
        # Execute the plot command
        if plot_commands:
            try:
                plot_cmd = 'plot ' + ', '.join(plot_commands)
                print(f"Running gnuplot command: {plot_cmd}")
                g.cmd(plot_cmd)
                # Explicitly close the output to ensure the file is written
                g.cmd('set output')
                print(f"Plot successfully created at: {fn_out}")
            except BrokenPipeError:
                print("Error: Connection to gnuplot failed (broken pipe).")
                print("This usually happens when gnuplot is not properly installed.")
                sys.exit(1)
            except Exception as e:
                print(f"Error during plotting: {e}")
                sys.exit(1)
        else:
            print("No data to plot!")
    
    finally:
        print(f"Temporary directory: {temp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", metavar="DATASET", default="glove-100-angular")
    parser.add_argument("--count", default=10)
    parser.add_argument(
        "--definitions", metavar="FILE", help="load algorithm definitions from FILE", default="algos.yaml"
    )
    parser.add_argument("--limit", default=-1)
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "-x", "--x-axis", help="Which metric to use on the X-axis", choices=metrics.keys(), default="k-nn"
    )
    parser.add_argument(
        "-y", "--y-axis", help="Which metric to use on the Y-axis", choices=metrics.keys(), default="qps"
    )
    parser.add_argument(
        "-X", "--x-scale", help="Scale to use when drawing the X-axis. Typically linear, logit or a2", default="linear",
    )
    parser.add_argument(
        "-Y",
        "--y-scale",
        help="Scale to use when drawing the Y-axis",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
    )
    parser.add_argument(
        "--raw", help="Show raw results (not just Pareto frontier) in faded colours", action="store_true"
    )
    parser.add_argument("--batch", help="Plot runs in batch mode", action="store_true")
    parser.add_argument("--recompute", help="Clears the cache and recomputes the metrics", action="store_true")
    parser.add_argument("--debug", help="Print debug information", action="store_true")
    args = parser.parse_args()

    if not args.output:
        args.output = "results/%s.pdf" % (args.dataset + ("-batch" if args.batch else ""))
        print("writing output to %s" % args.output)

    # Make sure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset, _ = get_dataset(args.dataset)
    count = int(args.count)
    unique_algorithms = get_unique_algorithms()
    results = load_all_results(args.dataset, count, args.batch)
    linestyles = create_linestyles(sorted(unique_algorithms))
    runs = compute_metrics(np.array(dataset["distances"]), results, args.x_axis, args.y_axis, args.recompute)
    if not runs:
        raise Exception("Nothing to plot")

    create_plot_gnu(
        runs, args.raw, args.x_scale, args.y_scale, args.x_axis, args.y_axis, args.output, linestyles, args.batch
    )