import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math


class PruningPlotter:
    def __init__(self, nrows=1, ncols=1, width=9, height=5):
        matplotlib.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Calibri"],
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
        if not isinstance(self.axes, list):
            self.axes = [self.axes]
        self.ax = self.axes[0]

        for ax in self.axes:
            ax.grid(True)
            ax.all_x = []
            ax.all_y = []

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

    def get_next_style(self):
        marker = self.markers.pop(0)
        self.markers.append(marker)
        line = self.lines.pop(0)
        self.lines.append(line)
        color = self.colors.pop(0)
        self.colors.append(color)

        style = {"marker": marker, "color": color, "linestyle": line}
        return style

    def set_ax(self, idx=0):
        self.ax = self.axes[idx]

    @staticmethod
    def get_acc_delta(accuracies, baseline):
        return [(x / baseline - 1) * 100 for x in accuracies]

    @staticmethod
    def get_compresion_ratios(sparsities):
        return [1 / (1 - x) for x in sparsities]

    @staticmethod
    def add_formatting(data, format_str):
        return [format_str.format(x) for x in data]

    def add_results(
        self, accuracies, sparsities, label="Unnamed", baseline_accuracy=1.0,
    ):
        if label in self.label2style:
            style = self.label2style[label]
        else:
            style = self.get_next_style()
            self.label2style[label] = style

        compression = self.get_compresion_ratios(sparsities)
        self.ax.all_x.extend(compression)

        delta_acc = self.get_acc_delta(accuracies, baseline_accuracy)
        self.ax.all_y.extend(delta_acc)

        line = plt.Line2D(compression, delta_acc, label=label, markersize=10, **style)
        self.ax.add_line(line)

    def update_x(self, num_ticks=4, mode="linspace", fmt="{}", precision=2, label=""):
        xmin, xmax = min(self.ax.all_x), max(self.ax.all_x)

        if mode == "logspace":
            self.ax.set_xscale("log")
            xticks = np.logspace(
                math.log10(xmin), math.log10(xmax), num=num_ticks,
            ).round(precision)
        elif mode == "linspace":
            self.ax.set_xscale("linear")
            xticks = np.linspace(xmin, xmax, num=num_ticks,).round(precision)
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
            yticks = np.linspace(ymin, ymax, num=num_ticks,).round(precision)
        else:
            raise NameError(f"Possible modes: logspace, linspace")

        if precision == 0:
            yticks = yticks.astype(int)
        yticks = np.unique(yticks)

        self.ax.set_yticks(yticks)
        yticks_l = self.add_formatting(yticks, fmt)
        self.ax.set_yticklabels(yticks_l)
        self.ax.set_ylabel(label)

    def update(self, title=""):
        self.ax.set_title(title)
        self.ax.minorticks_off()
        self.ax.legend()

    def show(self):
        plt.tight_layout()
        self.fig.show()


# %%


plotter = PruningPlotter()
for _ in range(4):
    plotter.add_results(
        1 - np.random.rand(6) * 0.05, np.random.rand(6), label=f"test{_}",
    )

plotter.update_x(
    mode="logspace", precision=2, fmt=r"{}$\times$", label="Compression ratio"
)
plotter.update_y(
    mode="linspace", num_ticks=20, precision=0, fmt="{}%", label=r"$\Delta$ Accuracy"
)
plotter.update("CIFAR-10 ResNet-56 Unstructured (iterative)")
plotter.show()

plotter.fig.savefig("filename.pdf", format="pdf")
