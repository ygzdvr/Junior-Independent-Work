import matplotlib as mpl

mpl.use("Agg")  # noqa
import argparse

import matplotlib.pyplot as plt
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import (compute_metrics, create_linestyles,
                                           create_pointset, get_plot_label)
from ann_benchmarks.results import get_unique_algorithms, load_all_results


def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, batch, xmin):
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Store all plotted points
    all_xs = []
    all_ys = []

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        # Calculate points for sorting, but don't store them here yet
        # We'll recalculate inside the main loop to ensure consistency
        _, local_ys, _, _, _, _ = create_pointset(all_data[algo], xn, yn)
        # Handle empty ys to avoid errors
        if not local_ys:
             return 0 # Or some other default value
        return -np.log(np.maximum(1e-9, np.array(local_ys))).mean() # Avoid log(0)

    # Find range for logit x-scale
    min_x_logit, max_x_logit = 1, 0
    for algo in sorted(all_data.keys(), key=mean_y):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        # Store points for later y-axis adjustment
        all_xs.extend(xs)
        all_ys.extend(ys)
        if raw:
            all_xs.extend(axs)
            all_ys.extend(ays)

        min_x_logit = min([min_x_logit] + [x for x in xs if x > 0])
        max_x_logit = max([max_x_logit] + [x for x in xs if x < 1])
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

    ax = plt.gca()
    ax.set_ylabel(ym["description"])
    ax.set_xlabel(xm["description"])
    # Custom scales of the type --x-scale a3
    if x_scale[0] == "a":
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
            # plt.xticks(ticker.LogitLocator().tick_values(min_x_logit, max_x_logit))
            plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
    # Other x-scales
    else:
        ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(get_plot_label(xm, ym))
    plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 9})
    plt.grid(visible=True, which="major", color="0.65", linestyle="-")
    plt.setp(ax.get_xminorticklabels(), visible=True)

    # Set initial x-limits based on metric defaults or logit range
    if "lim" in xm and x_scale != "logit":
        x0, x1 = xm["lim"]
        plt.xlim(max(x0, 0), min(x1, 1))
    elif x_scale == "logit":
        plt.xlim(min_x_logit, max_x_logit)
    # else: let matplotlib decide based on data

    # Apply the custom lower x-limit if provided (--xmin)
    if xmin is not None:
        current_xlim = plt.xlim()
        print(f"Applying x-axis lower limit: {xmin}. Current limits: {current_xlim}")
        # Only set the lower bound, keep the upper bound
        plt.xlim(left=max(xmin, current_xlim[0] if current_xlim[0] is not None else xmin), right=current_xlim[1])
        print(f"New x-axis limits: {plt.xlim()}")

    # --- Adjust y-axis --- 
    final_xlim = plt.xlim()
    # Filter points that are within the final x-axis limits
    visible_ys = [y for x, y in zip(all_xs, all_ys) if x >= final_xlim[0] and x <= final_xlim[1]]

    if visible_ys:
        min_y = min(visible_ys)
        max_y = max(visible_ys)
        y_range = max_y - min_y
        # Add 5% padding, but handle cases where range is zero or very small
        padding = max(y_range * 0.05, 1e-6) 
        
        final_min_y = min_y - padding
        final_max_y = max_y + padding
        
        # Optionally respect metric limits if they exist
        if "lim" in ym:
            final_min_y = max(final_min_y, ym["lim"][0])
            final_max_y = min(final_max_y, ym["lim"][1])

        plt.ylim(final_min_y, final_max_y)
        print(f"Adjusting y-axis limits based on visible data: ({final_min_y:.3f}, {final_max_y:.3f})")
    elif "lim" in ym: # Fallback to metric limits if no data is visible
         plt.ylim(ym["lim"])
    # else: let matplotlib decide if no data and no metric limits
    # --- End y-axis adjust ---

    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines["bottom"]._adjust_location()

    plt.savefig(fn_out, bbox_inches="tight", dpi=144)
    plt.close()


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
        "--xmin",
        type=float,
        default=None,
        help="Set the minimum value for the x-axis (e.g., 100 for QPS, 0.9 for Recall)"
    )
    parser.add_argument(
        "--raw", help="Show raw results (not just Pareto frontier) in faded colours", action="store_true"
    )
    parser.add_argument("--batch", help="Plot runs in batch mode", action="store_true")
    parser.add_argument("--recompute", help="Clears the cache and recomputes the metrics", action="store_true")
    parser.add_argument(
        "--algorithms",
        metavar="ALG",
        help="Only plot results for the specified algorithms",
        nargs="*",
        default=None,
    )
    args = parser.parse_args()

    if not args.output:
        args.output = "results/%s.png" % (args.dataset + ("-batch" if args.batch else ""))
        print("writing output to %s" % args.output)

    dataset, _ = get_dataset(args.dataset)
    count = int(args.count)

    # First pass to get available algorithms without keeping file handles open
    available_algorithms_gen = (properties["algo"] for properties, _ in load_all_results(args.dataset, count, args.batch))
    available_algorithms = sorted(list(set(available_algorithms_gen)))
    print(f"Available algorithms: {available_algorithms}")
    if not available_algorithms:
         raise Exception("No results found for the specified dataset and parameters.")

    algorithms_to_plot = args.algorithms
    if not algorithms_to_plot:
        algorithms_to_plot = available_algorithms
    else:
        # Validate provided algorithm names
        valid_algorithms = []
        for algo in args.algorithms: # Iterate over user input order
            if algo in available_algorithms:
                valid_algorithms.append(algo)
            else:
                print(f"Warning: Algorithm '{algo}' not found in results, skipping.")
        algorithms_to_plot = valid_algorithms # Keep user's order if valid
        if not algorithms_to_plot:
            raise Exception("No valid algorithms specified or found to plot.")

    print(f"Plotting algorithms: {', '.join(algorithms_to_plot)}")

    # Get a fresh generator to pass to compute_metrics
    results_generator = load_all_results(args.dataset, count, args.batch)

    # Create linestyles based on the algorithms we intend to plot
    linestyles = create_linestyles(algorithms_to_plot) # Use the filtered list

    # Pass the generator and the list of algorithms to plot
    runs = compute_metrics(
        np.array(dataset["distances"]),
        results_generator,
        args.x_axis,
        args.y_axis,
        args.recompute,
        algorithms_to_plot=algorithms_to_plot # Pass the list here
    )

    # Filter runs to ensure only plotted algorithms are considered for the plot function
    # compute_metrics already returns filtered results based on algorithms_to_plot,
    # but the keys might include algos that didn't produce valid metrics.
    # create_plot iterates over runs.keys(), so ensure these keys are in linestyles.
    runs_to_plot = {algo: data for algo, data in runs.items() if algo in algorithms_to_plot}

    if not runs_to_plot:
        raise Exception("Nothing to plot after computing metrics (check data and parameters).")

    create_plot(
        runs_to_plot, args.raw, args.x_scale, args.y_scale, args.x_axis, args.y_axis, args.output, linestyles, args.batch, args.xmin
    )
