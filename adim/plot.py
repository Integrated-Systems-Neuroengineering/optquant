from matplotlib.ticker import MaxNLocator, MultipleLocator
from .prob import log_pdf_n, EmpiricalDistribution
from .utils import compute_even_levels
from jax import numpy as jnp


def plot_quantization(
    ax_fun,
    thresholds,
    dist=None,
    color="black",
    linecolor=None,
    xticklocator=MultipleLocator(64),
    yticklocator=MaxNLocator(integer=True),
    remove_endpoints=True,
    xmin=None,
    xmax=None,
    step=True,
    twinaxis=True,
):
    if linecolor is None:
        linecolor = color

    if remove_endpoints:
        thresholds = thresholds[1:-1]

    if xmin is None:
        if dist is not None:
            xmin = dist.support.min()
        else:
            xmin = thresholds.min()
    if xmax is None:
        if dist is not None:
            xmax = dist.support.max()
        else:
            xmax = thresholds.max()

    num_levels = len(thresholds) + 1

    xx = jnp.linspace(xmin, xmax, 1001)

    if dist is not None:
        nn = dist.support
        n_cdf = dist.cdf(nn)

        if twinaxis:
            # callback for when the ylim of ax_cdf is changed
            def update_cdf_ylim(ax_fun):
                y1, y2 = ax_fun.get_ylim()
                ax_cdf.set_ylim(y1 / (num_levels - 1), y2 / (num_levels - 1))
                ax_cdf.figure.canvas.draw()

            # create twin axis for the CDF
            ax_cdf = ax_fun.twinx()
            # swap zorder so that the CDF is behind the quantization levels
            ax_fun.set_zorder(ax_cdf.get_zorder() + 1)

            # automatically update ylim of ax2 when ylim of ax1 changes.
            ax_fun.callbacks.connect("ylim_changed", update_cdf_ylim)

            ax_cdf.yaxis.label.set_color("gray")
            ax_cdf.tick_params(axis="y", colors="gray")
            ax_cdf.grid(True, zorder=0)
            ax_cdf.xaxis.grid(True, zorder=0)
            ax_fun.grid(False)
            ax_cdf.set_ylabel("cumulative prob.")
            # Set ax's patch invisible
            ax_fun.patch.set_visible(False)
            # Set axtwin's patch visible and colorize it in grey
            ax_cdf.patch.set_visible(True)
            ax_cdf.patch.set_facecolor(ax_fun.patch.get_facecolor())
            scale = 1
        else:
            ax_cdf = ax_fun
            scale = num_levels - 1

        if step:
            l1 = ax_cdf.step(nn, n_cdf * scale, color="gray", where="post", label="CDF")
        else:
            nn_mid = (nn[1:] + nn[:-1]) / 2
            n_cdf_mid = (n_cdf[1:] + n_cdf[:-1]) / 2
            l1 = ax_cdf.plot(nn_mid, n_cdf_mid * scale, color="gray", label="CDF")
    else:
        l1 = []

    l2 = ax_fun.step(
        xx,
        jnp.searchsorted(thresholds, xx),
        color=linecolor,
        linewidth=1,
        label="$Q(\\tilde y)$",
    )

    ax_fun.yaxis.label.set_color(color)
    ax_fun.tick_params(axis="y", colors=color)
    ax_fun.set_ylabel("level")

    ax_fun.xaxis.set_major_locator(xticklocator)
    ax_fun.yaxis.set_major_locator(yticklocator)
    ax_fun.set_xlim(xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.05)

    if twinaxis:
        return ax_cdf, ax_fun, l1 + l2
    else:
        return ax_fun, l1 + l2


def plot_sweep(ax, phase, scale, MIs, annotate=None, **kwargs):
    im = ax.pcolormesh(phase.ravel(), scale.ravel(), MIs, **kwargs)
    ax.set_yscale("log")

    if annotate is not None:
        phase_ann, scale_ann, MI_ann = annotate
        ax.plot(phase_ann, scale_ann, "wx")

        # annotate optimum
        ax.annotate(
            f"offset: {phase_ann:.0f}\nscale: {scale_ann:.0f}\nMI: {MI_ann:.2f} bits",
            (phase_ann, scale_ann),
            (0, 5),
            va="bottom",
            ha="center",
            textcoords="offset points",
            fontsize=10,
            color="white",
        )
    return ax, im
