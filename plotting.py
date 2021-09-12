import math
from collections import Iterable
import os.path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.container import ErrorbarContainer
import yaml


class PruningPlotter:
    def __init__(self, nrows=1, ncols=1, width=9, height=5,
                 x_data_format=0, y_data_format=0):
        matplotlib.rcParams.update(
            {
                "font.family": "serif",
                "font.sans-serif": ["Computer Modern Roman"],
                "font.weight": "light",
                "axes.labelsize": 16,
                "font.size": 20,
                "legend.fontsize": 14,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "figure.figsize": (width, height),
                "figure.dpi": 72,
                "legend.loc": "best",
                "axes.titlepad": 20,
            }
        )

        self.fig, self.axes = plt.subplots(nrows, ncols)
        if not isinstance(self.axes, np.ndarray):
            self.axes = np.array([self.axes])
        self.axes = self.axes.flatten()
        self.set_current_axis(0)

        for ax in self.axes.flatten():
            ax.grid(True)
            ax.all_x = []  # USED FOR CALCULATING RANGE ON X AXIS
            ax.all_y = []  # USED FOR CALCULATING RANGE ON Y AXIS

            for spine in ax.spines:
                ax.spines[spine].set_visible(False)

            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")

            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_tick_params(direction="out", color="gray")

        self.label2style = {}
        self.markers = [
            "o",
            "v",
            "s",
            "p",
            "*",
            ">",
            "D",
            "d",
            "P",
        ]
        self.lines = list(plt.Line2D.lineStyles)[:4]
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.set_data_format(x_data_format, y_data_format)

    def set_data_format(self, x_data_format, y_data_format):
        # X DATA FORMAT
        # 0: Compression ratio
        # 1: Raw sparsity
        # 2: Density
        self.x_data_format = x_data_format

        # Y DATA FORMAT
        # 0: Delta of accuracy: run_accuracy - baseline accuracy
        # 1: Raw accuracy
        self.y_data_format = y_data_format

    def set_current_axis(self, idx=0):
        self.ax = self.axes[idx]

    def get_next_style(self):
        marker = self.markers.pop(0)
        self.markers.append(marker)
        line = self.lines.pop(0)
        self.lines.append(line)
        color = self.colors.pop(0)
        self.colors.append(color)

        style = {"marker": marker, "color": color, "linestyle": line}
        return style

    @staticmethod
    def get_acc_deltas(accuracies, baseline):
        return [(x / baseline - 1) * 100 for x in accuracies]

    @staticmethod
    def get_compresion_ratios(sparsities):
        return [1 / (1 - x) for x in sparsities]

    @staticmethod
    def add_formatting(data, format_str):
        return [format_str.format(x) for x in data]

    def add_pruning_results(
        self,
        *accuracies,
        sparsities,
        label="Unnamed",
        baseline_accuracy=1.0,
        error_percentile=90,
    ):
        if label in self.label2style:
            style = self.label2style[label]
        else:
            style = self.get_next_style()
            self.label2style[label] = style

        if self.x_data_format == 1:
            x_data = sparsities
        elif self.x_data_format == 2:
            x_data = [1 - sparsity for sparsity in sparsities]
        elif self.x_data_format == 0:
            x_data = self.get_compresion_ratios(sparsities)
        else:
            raise NotImplementedError
        self.ax.all_x.extend(x_data)

        if self.y_data_format == 1:
            y_data = accuracies
        elif self.y_data_format == 0:
            y_data = [self.get_acc_deltas(x, baseline_accuracy) for x in accuracies]
        else:
            raise NotImplementedError
        self.ax.all_y.extend(sum(y_data, []))

        acc_median = [np.nanmedian(x) for x in y_data]
        acc_topbar = [np.percentile(x, error_percentile) - med
                      for med, x in zip(acc_median, y_data)]
        acc_botbar = [med - np.percentile(x, 100 - error_percentile)
                      for med, x in zip(acc_median, y_data)]
        yerr = (acc_botbar, acc_topbar) if acc_topbar != acc_botbar else None
        self.ax.errorbar(
            x_data,
            acc_median,
            yerr=yerr,
            label=label,
            markersize=10,
            capsize=4,
            linewidth=2,
            capthick=2,
            **style,
        )

    def add_horizontal_line(self, height):
        xmin, xmax = min(self.ax.all_x), max(self.ax.all_x)
        self.ax.plot([xmin, xmax],
                     [height, height],
                     color='red',
                     linewidth=4,
                     alpha=0.3,
                     zorder=-1,
                     linestyle='--')
        self.ax.plot([xmin, xmax],
                     [height, height],
                     color='red',
                     linewidth=8,
                     alpha=0.2,
                     zorder=-2,
                     linestyle='-')
        plt.plot()

    def update_x(self, num_ticks=4, mode="linspace", fmt="{}", precision=2, label=""):
        xmin, xmax = min(self.ax.all_x), max(self.ax.all_x)

        if mode == "logspace":
            self.ax.set_xscale("log")
            xticks = np.logspace(
                math.log10(xmin), math.log10(xmax), num=num_ticks,
            ).round(precision)
        elif mode == "linspace":
            self.ax.set_xscale("linear")
            xticks = np.linspace(xmin, xmax, num=num_ticks, ).round(precision)
        else:
            raise NameError(f"Possible modes: logspace, linspace")

        if precision == 0:
            xticks = xticks.astype(int)
        xticks = np.unique(xticks)

        self.ax.set_xticks(xticks)
        xticks_l = self.add_formatting(xticks, fmt)
        self.ax.set_xticklabels(xticks_l)
        self.ax.set_xlabel(label)

    def update_y(self, num_ticks=4, mode="linspace", fmt="{}", precision=2, label=""):
        ymin, ymax = min(self.ax.all_y), max(self.ax.all_y)

        if mode == "logspace":
            self.ax.set_yscale("log")
            yticks = np.logspace(
                math.log10(ymin), math.log10(ymax), num=num_ticks,
            ).round(precision)
        elif mode == "linspace":
            self.ax.set_yscale("linear")
            yticks = np.linspace(ymin, ymax, num=num_ticks).round(precision)
        else:
            raise NameError(f"Possible modes: linspace, logspace")

        if precision == 0:
            yticks = yticks.astype(int)
        yticks = np.unique(yticks)

        self.ax.set_yticks(yticks)
        yticks_l = self.add_formatting(yticks, fmt)
        self.ax.set_yticklabels(yticks_l)
        self.ax.set_ylabel(label)

    def prepare(self, title="", legend=True):
        self.ax.set_title(title)
        self.ax.minorticks_off()
        if legend:
            self.ax.legend()
        plt.tight_layout()

    def show(self):
        self.fig.show()


def find_entry(name, source):
    for entry in yaml.safe_load_all(open(source, 'r')):
        if entry is None:
            continue

        if entry['name'] == name:
            break
    else:
        raise UserWarning(f"Missing name {name} in {source}!")
    return entry


def load_axis_from_recipe(plotter: PruningPlotter, path, axis=0):
    plotter.set_current_axis(idx=axis)
    settings, *data = yaml.safe_load_all(open(path, 'r'))
    plotter.set_data_format(settings['x_axis'].pop('data_format'),
                            settings['y_axis'].pop('data_format'))

    for new_trace in data:
        trace = settings.copy()
        trace.update(new_trace)
        source = trace['source']

        optional_params = {
            key: value for key, value in trace.items()
            if key in (
                'baseline_accuracy',
                'error_percentile',
            )
        }

        entry = find_entry(trace['name'], source)
        sparsities, accuracies = zip(*entry['pruning_accuracies'].items())
        plotter.add_pruning_results(
            *accuracies,
            sparsities=sparsities,
            label=trace['label'],
            **optional_params,
        )

    plotter.update_x(**settings['x_axis'])
    plotter.update_y(**settings['y_axis'])

    plotter.prepare(title=settings['title'], legend=settings.get('legend'))
    return plotter


def compose_plots(
    *names,
    nrows=1,
    ncols=None,
    height=None,
    width=None,
    path=""
):
    if ncols is None:
        ncols = len(names)
    if height is None:
        height = 5 * nrows
    if width is None:
        width = 8 * ncols
    plotter = PruningPlotter(nrows=nrows, ncols=ncols, height=height, width=width)

    for idx, name in enumerate(names):
        plotter = load_axis_from_recipe(plotter, os.path.join(path, name), idx)

    handles, labels = plotter.ax.get_legend_handles_labels()
    plotter.fig.legend(handles,
                       labels,
                       loc='lower center',
                       ncol=5,
                       bbox_to_anchor=(0.5, -0.01))
    plotter.fig.subplots_adjust(bottom=0.23)
    plotter.show()
    return plotter.fig
