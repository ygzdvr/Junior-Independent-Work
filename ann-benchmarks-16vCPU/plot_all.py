import matplotlib as mpl
mpl.use("Agg")  # noqa
import argparse
import os
import itertools
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import (compute_metrics, create_linestyles,
                                         create_pointset, get_plot_label)
from ann_benchmarks.results import get_unique_algorithms, load_all_results

def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, batch):
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        if not ys:
            return 0
        return -np.log(np.array(ys) + 1e-10).mean()

    # Find range for logit x-scale
    min_x, max_x = 1, 0
    algorithms_with_data = []
    
    for algo in sorted(all_data.keys(), key=mean_y):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        if not xs or not ys:
            continue
        algorithms_with_data.append(algo)
        min_x = min([min_x] + [x for x in xs if x > 0])
        max_x = max([max_x] + [x for x in xs if x < 1])
        color, faded, linestyle, marker = linestyles[algo]
        (handle,) = plt.plot(
            xs, ys, "-", label=algo, color=color, ms=7, mew=3, lw=3, marker=marker
        )
        handles.append(handle)
        if raw:
            (handle2,) = plt.plot(
                axs, ays, "-", label=algo, color=faded, ms=5, mew=2, lw=2, marker=marker
            )
        labels.append(algo)

    # Don't create empty plots
    if not algorithms_with_data:
        plt.close()
        return False

    ax = plt.gca()
    ax.set_ylabel(ym["description"])
    ax.set_xlabel(xm["description"])
    
    # Custom scales of the type --x-scale a3
    if x_scale and x_scale[0] == "a":
        try:
            alpha = float(x_scale[1:])

            def fun(x):
                return 1 - (1 - x) ** (1 / alpha)

            def inv_fun(x):
                return 1 - (1 - x) ** alpha

            ax.set_xscale("function", functions=(fun, inv_fun))
            if alpha <= 3:
                ticks = [inv_fun(x) for x in np.arange(0, 1.2, 0.2)]
                plt.xticks(ticks)
            if alpha > 3:
                from matplotlib import ticker

                ax.xaxis.set_major_formatter(ticker.LogitFormatter())
                plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
        except:
            # Fallback to linear if parsing fails
            ax.set_xscale("linear")
    # Other x-scales
    elif x_scale:
        try:
            ax.set_xscale(x_scale)
        except:
            # Fallback to linear if invalid scale
            ax.set_xscale("linear")
    
    try:
        ax.set_yscale(y_scale)
    except:
        # Fallback to linear if invalid scale
        ax.set_yscale("linear")
    
    ax.set_title(get_plot_label(xm, ym))
    plt.gca().get_position()
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 9})
    plt.grid(visible=True, which="major", color="0.65", linestyle="-")
    plt.setp(ax.get_xminorticklabels(), visible=True)

    # Logit scale has to be a subset of (0,1)
    if "lim" in xm and x_scale != "logit":
        x0, x1 = xm["lim"]
        plt.xlim(max(x0, 0), min(x1, 1))
    elif x_scale == "logit":
        plt.xlim(min_x, max_x)
    if "lim" in ym:
        plt.ylim(ym["lim"])

    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    try:
        ax.spines["bottom"]._adjust_location()
    except:
        pass

    plt.savefig(fn_out, bbox_inches="tight", dpi=144)
    plt.close()
    return True

def get_appropriate_scale(metric_name):
    """Return appropriate scale for a given metric"""
    if metric_name in ['recall', 'precision', 'ratio', 'epsilon', 'k-nn']:
        return 'linear'
    elif metric_name in ['qps', 'build', 'candidates', 'distcomps', 'queriessize', 'indexsize', 'train']:
        return 'log'
    else:
        return 'linear'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", metavar="DATASET", default="glove-100-angular", 
                       help="Dataset to plot (default: glove-100-angular)")
    parser.add_argument("--count", default=10, help="Number of queries (default: 10)")
    parser.add_argument("--definitions", metavar="FILE", 
                       help="Load algorithm definitions from FILE (default: algos.yaml)", 
                       default="algos.yaml")
    parser.add_argument("--output-dir", default="plots", 
                       help="Directory to output plots (default: plots)")
    parser.add_argument("--raw", help="Show raw results (not just Pareto frontier)", 
                       action="store_true")
    parser.add_argument("--batch", help="Plot runs in batch mode", action="store_true")
    parser.add_argument("--recompute", help="Clears the cache and recomputes the metrics", 
                       action="store_true")
    parser.add_argument("--x-scale-override", 
                       help="Override default x-axis scale for all plots", default=None)
    parser.add_argument("--y-scale-override", 
                       help="Override default y-axis scale for all plots", default=None)
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Generating all metric combinations for dataset: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Total combinations: {len(metrics) * len(metrics)}")

    # Load dataset and results once
    dataset, _ = get_dataset(args.dataset)
    count = int(args.count)
    unique_algorithms = get_unique_algorithms()
    results = load_all_results(args.dataset, count, args.batch)
    linestyles = create_linestyles(sorted(unique_algorithms))
    
    # Create a summary file
    summary_file = os.path.join(dataset_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Metric plots for dataset: {args.dataset}\n")
        f.write(f"Generated plots: \n")
    
    # Compute all plots
    combinations = list(itertools.product(metrics.keys(), metrics.keys()))
    successful_plots = 0
    
    for x_axis, y_axis in tqdm(combinations, desc="Generating plots"):
        # Skip if metrics are the same
        if x_axis == y_axis:
            continue
            
        # Determine appropriate scales
        x_scale = args.x_scale_override if args.x_scale_override else get_appropriate_scale(x_axis)
        y_scale = args.y_scale_override if args.y_scale_override else get_appropriate_scale(y_axis)
        
        # Set output file name
        output_file = os.path.join(dataset_dir, f"{x_axis}_vs_{y_axis}.png")
        
        # Compute metrics for this combination
        runs = compute_metrics(np.array(dataset["distances"]), results, x_axis, y_axis, args.recompute)
        
        if runs:
            success = create_plot(
                runs, args.raw, x_scale, y_scale, x_axis, y_axis, 
                output_file, linestyles, args.batch
            )
            if success:
                successful_plots += 1
                with open(summary_file, 'a') as f:
                    f.write(f"- {x_axis} vs {y_axis}: {output_file}\n")
    
    # Update summary
    with open(summary_file, 'a') as f:
        f.write(f"\nTotal successful plots: {successful_plots}/{len(combinations)-len(metrics)}\n")
    
    print(f"Successfully generated {successful_plots} plots")
    print(f"Summary written to {summary_file}")

if __name__ == "__main__":
    main() 