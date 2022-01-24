from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from . import PROJECT_DIR

plt.style.use(PROJECT_DIR.joinpath("plot_formatting.mplstyle"))


@dataclass
class Series:
    data: np.ndarray
    label: Optional[str] = None
    ls: str = "-"
    color: Optional[str] = None
    plot_type: str = "plot"
    alpha: float = 1.0


@dataclass
class FrequencySeries:
    magnitude: np.ndarray
    phase: np.ndarray
    label: Optional[str] = None
    ls: str = "-"
    color: Optional[str] = None
    plot_type: str = "plot"
    alpha: float = 1.0


def multiseries_plot(x: Series, *y: Series, **options) -> None:

    fig, ax = plt.subplots(1, 1)
    plot = {"plot": ax.plot, "scatter": ax.scatter}

    for n, vector in enumerate(y):
        plot_func = plot.get(vector.plot_type, plot["plot"])
        plot_func(
            x.data,
            vector.data,
            label=vector.label,
            ls=vector.ls,
            color=f"C{n}" if vector.color is not None else vector.color,
            alpha=vector.alpha,
        )

    ax.set(
        xlabel=x.label,
        ylabel=options.get("ylabel", ""),
        xscale=options.get("xscale", "linear"),
    )

    if len(y) > 1:
        ax.legend()
    fig.suptitle(options.get("title", ""))

    if options.get("show", True):
        plt.show()

    if options.get("file_path", False):
        fig.savefig(options["file_path"])
        print(f"Saved Figure: {options['file_path']}")


def multiseries_bodeplot(f: Series, *h: FrequencySeries, **options) -> None:

    fig, axes = plt.subplots(2, 1, sharex=True)
    plot_functions = {
        "plot": lambda i: axes[i].plot,
        "scatter": lambda i: axes[i].scatter,
    }

    units = (options.get("mag_unit", "dB"), options.get("phase_unit", "deg"))

    for i in (0, 1):
        ax = axes[i]
        for n, vector in enumerate(h):
            plot_func = plot_functions.get(
                vector.plot_type, plot_functions["plot"]
            )(i)
            color = f"C{n}" if vector.color is not None else vector.color
            plot_func(
                f.data,
                vector.magnitude if i == 0 else vector.phase,
                label=vector.label,
                ls=vector.ls,
                color=color,
                alpha=vector.alpha,
            )

        if options.get("grid", True):
            ax.grid(True, which="both", axis="both")

        ax.set(
            xlabel=f.label,
            ylabel=options.get("ylabel", "") + f" ({units[i]})",
            xscale=options.get("xscale", "linear"),
        )

        if len(h) > 1:
            ax.legend()

    fig.suptitle(options.get("title", ""))

    if options.get("show", True):
        plt.show()

    if options.get("file_path", False):
        fig.savefig(options["file_path"])
        print(f"Saved Figure: {options['file_path']}")
