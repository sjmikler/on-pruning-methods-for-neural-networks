import math
from collections import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import ErrorbarContainer
import yaml


class PruningPlotter:
    def __init__(self, nrows=1, ncols=1, width=9, height=5,
                 x_data_format=0, y_data_format=0):
        matplotlib.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"],
                # "font.weight": "light",
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
        self, *accuracies, sparsities, label="Unnamed", baseline_accuracy=1.0,
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
        acc_topbar = [np.nanmax(x) - med for med, x in zip(acc_median, y_data)]
        acc_botbar = [med - np.nanmin(x) for med, x in zip(acc_median, y_data)]
        self.ax.errorbar(
            x_data,
            acc_median,
            label=label,
            markersize=10,
            yerr=(acc_botbar, acc_topbar),
            capsize=4,
            **style,
        )

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

    def prepare(self, title=""):
        self.ax.set_title(title)
        self.ax.minorticks_off()
        self.ax.legend()

    def add_data(self, data):
        for entry in data:
            baseline_accuracy = entry.get('baseline_accuracy') or 1.0
            self.add_pruning_results(
                *entry['accuracies'],
                sparsities=entry['sparsities'],
                label=entry['name'],
                baseline_accuracy=baseline_accuracy
            )

    def show(self):
        plt.tight_layout()
        self.fig.show()


def load_axis(plotter, path, axis=0):
    plotter.set_current_axis(idx=axis)
    settings, *data = yaml.safe_load_all(open(path, 'r'))
    plotter.set_data_format(settings['x_data_format'], settings['y_data_format'])
    plotter.add_data(data)
    plotter.update_x(
        mode=settings['x_mode'],
        precision=settings['x_precision'],
        fmt=settings['x_fmt'],
        label=settings['x_label'],
    )
    plotter.update_y(
        mode=settings['y_mode'],
        precision=settings['y_precision'],
        fmt=settings['y_fmt'],
        label=settings['y_label'],
    )
    plotter.prepare(title=settings['title'])
    return plotter


plotter = PruningPlotter(x_data_format=1, y_data_format=1)

plotter.add_pruning_results(
    [0.9], [0.97, 0.93, 0.91, 0.90], [0.93, 0.9],
    sparsities=[0.1, 0.9, 0.98],
    baseline_accuracy=0.95,
    label="test1"
)

plotter.add_pruning_results(
    [0.92, 0.95], [0.93, 0.9, 0.88], [0.96],
    sparsities=[0.1, 0.2, 0.995],
    baseline_accuracy=0.95,
    label="test2"
)

plotter.update_x(
    mode="logspace", precision=3, fmt=r"{}$\times$", label="Compression ratio"
)
plotter.update_y(
    mode="linspace", num_ticks=20, precision=2, fmt="{}%", label=r"$\Delta$ Accuracy"
)
plotter.prepare("CIFAR-10 ResNet-56 Unstructured (iterative)")
plotter.show()

# %%

plotter = PruningPlotter(nrows=1, ncols=2, height=6, width=15)
plotter = load_axis(plotter, "data/ResNet20-LR-rewinding-results.yaml", 0)
plotter = load_axis(plotter, "data/ResNet56-LR-rewinding-results.yaml", 1)
plt.suptitle('CIFAR10 one-shot pruning')
plotter.show()
plotter.fig.savefig("Resnet20vsResnet50.pdf")
