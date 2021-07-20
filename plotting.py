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
        acc_topbar = [np.percentile(x, 90) - med for med, x in zip(acc_median, y_data)]
        acc_botbar = [med - np.percentile(x, 10) for med, x in zip(acc_median, y_data)]
        self.ax.errorbar(
            x_data,
            acc_median,
            label=label,
            markersize=10,
            yerr=(acc_botbar, acc_topbar),
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

    plt.margins(x=0, y=1)

    def add_many_results(self, data):
        for entry in data:
            baseline_accuracy = entry.get('baseline_accuracy') or 1.0
            self.add_pruning_results(
                *entry['accuracies'],
                sparsities=entry['sparsities'],
                label=entry['name'],
                baseline_accuracy=baseline_accuracy
            )

    def show(self):
        self.fig.show()


def load_axis(plotter, path, axis=0):
    plotter.set_current_axis(idx=axis)
    settings, *data = yaml.safe_load_all(open(path, 'r'))
    plotter.set_data_format(settings['x_data_format'], settings['y_data_format'])
    plotter.add_many_results(data)
    plotter.update_x(
        mode=settings['x_mode'],
        precision=settings['x_precision'],
        fmt=settings['x_fmt'],
        label=settings['x_label'],
        num_ticks=settings['x_num_ticks'],
    )
    plotter.update_y(
        mode=settings['y_mode'],
        precision=settings['y_precision'],
        fmt=settings['y_fmt'],
        label=settings['y_label'],
        num_ticks=settings['y_num_ticks'],
    )
    plotter.prepare(title=settings['title'], legend=settings.get('legend'))
    return plotter


# %%

plotter = PruningPlotter(x_data_format=1, y_data_format=1)

plotter.add_pruning_results(
    [0.9], [0.97, 0.93, 0.91, 0.90], [0.93, 0.9],
    sparsities=[0.1, 0.9, 0.98],
    baseline_accuracy=0.95,
    label="test1"
)

plotter.add_pruning_results(
    [0.92, 0.95], [0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.88], [0.96],
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
plotter.add_horizontal_line(height=0.91)
plotter.prepare("CIFAR-10 ResNet-56 Unstructured (iterative)")
plotter.fig.legend()
plotter.show()

# %%

plotter = PruningPlotter(nrows=1, ncols=2, height=5, width=15)
plotter = load_axis(plotter, "data/repro_plot_data/ResNet20-cifar10.yaml", 0)
# plotter.add_horizontal_line(height=0)
plotter = load_axis(plotter, "data/repro_plot_data/ResNet20-cifar10-iterative.yaml", 1)
# plotter.add_horizontal_line(height=0)

handles, labels = plotter.ax.get_legend_handles_labels()
plotter.fig.legend(handles,
                   labels,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.5, -0.01))
plotter.fig.subplots_adjust(bottom=0.23)
plotter.show()
plotter.fig.savefig("data/repro_plot_data/plots/Resnet20-1s-iterative.pdf")

# %%

plotter = PruningPlotter(nrows=1, ncols=2, height=5, width=15)
plotter = load_axis(plotter, "data/repro_plot_data/ResNet56-cifar10.yaml", 0)
plotter = load_axis(plotter, "data/repro_plot_data/ResNet56-cifar10-iterative.yaml", 1)

handles, labels = plotter.ax.get_legend_handles_labels()
plotter.fig.legend(handles,
                   labels,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.5, -0.01))
plotter.fig.subplots_adjust(bottom=0.23)
plotter.show()
plotter.fig.savefig("data/repro_plot_data/plots/Resnet56-1s-iterative.pdf")

# %%

plotter = PruningPlotter(nrows=1, ncols=2, height=5, width=15)
plotter = load_axis(plotter, "data/repro_plot_data/WRN16-8-cifar10-one-shot.yaml", 0)
# plotter.add_horizontal_line(height=0)
plotter = load_axis(plotter, "data/repro_plot_data/WRN16-8-cifar10-iterative.yaml", 1)
# plotter.add_horizontal_line(height=0)

handles, labels = plotter.ax.get_legend_handles_labels()
plotter.fig.legend(handles,
                   labels,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.5, -0.01))
plotter.fig.subplots_adjust(bottom=0.23)
plotter.show()
plotter.fig.savefig("data/repro_plot_data/plots/WRN-16-8-LR-rewinding-is-flawed.pdf")

# %%

plotter = PruningPlotter(nrows=1, ncols=1, height=5, width=8)
plotter = load_axis(plotter,
                    "data/repro_plot_data/WRN16-8-cifar10-iterative-steps.yaml", 0)
# plotter.add_horizontal_line(height=0)

handles, labels = plotter.ax.get_legend_handles_labels()
plotter.fig.legend(handles,
                   labels,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.5, -0.01))
plotter.fig.subplots_adjust(bottom=0.23)
plotter.show()
plotter.fig.savefig("data/repro_plot_data/plots/WRN-16-8-LR-rew-compare2v3.pdf")

# %%

plotter = PruningPlotter(nrows=1, ncols=1, height=5, width=8)
plotter = load_axis(plotter, "data/repro_plot_data/ResNet20-cifar10-structured.yaml", 0)

handles, labels = plotter.ax.get_legend_handles_labels()
plotter.fig.legend(handles,
                   labels,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.5, -0.01))
plotter.fig.subplots_adjust(bottom=0.23)
plotter.show()
plotter.fig.savefig("data/repro_plot_data/plots/Resnet20-structured.pdf")

# %%

plotter = PruningPlotter(nrows=1, ncols=1, height=5, width=8)
plotter = load_axis(plotter, "data/repro_plot_data/ResNet56-cifar100.yaml", 0)
# plotter.add_horizontal_line(height=0)
handles, labels = plotter.ax.get_legend_handles_labels()
plotter.fig.legend(handles,
                   labels,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.5, -0.01))
plotter.fig.subplots_adjust(bottom=0.23)
plotter.show()
plotter.fig.savefig("data/repro_plot_data/plots/Resnet56-C100.pdf")

# %%
