from matplotlib.ticker import MaxNLocator, MultipleLocator
from .prob import log_pdf_n
from .utils import compute_even_levels
from jax import numpy as jnp


def plot_quantization(
    ax_cdf,
    vector_length,
    num_bits,
    alpha,
    beta,
    offset_fixed=0.0,
    color="black",
    linecolor=None,
    plot_cdf=True,
    xticklocator=MultipleLocator(64),
    yticklocator=MaxNLocator(integer=True),
):
    if linecolor is None:
        linecolor = color

    # compute the optimal thresholds for the given parameters
    opt_thresholds = compute_even_levels(
        2**num_bits, alpha, beta, offset_fixed=offset_fixed
    ).ravel()

    # compute the "background" probability distribution
    nn = jnp.arange(vector_length + 1) * 2 - vector_length
    # pp = np.exp(dists.Binomial(vector_length, 0.5).log_prob(xx))
    p_n = jnp.exp(
        log_pdf_n(nn, vector_length=vector_length, distribution="bipolar", fixed=0)
    )

    xx = jnp.linspace(nn[0], nn[-1], 1001)

    if plot_cdf:
        ax_fun = ax_cdf.twinx()
        l1 = ax_cdf.plot(nn, jnp.cumsum(p_n), color="gray", label="CDF")

        ax_cdf.yaxis.tick_right()
        ax_cdf.yaxis.label.set_color("gray")
        ax_cdf.tick_params(axis="y", colors="gray")
        ax_cdf.yaxis.set_label_position("right")
        ax_cdf.yaxis.grid(False)
        ax_cdf.set_ylim(-0.1, 1.1)
        ax_cdf.set_ylabel("cumulative prob.")
        ax_cdf.set_xlabel("$\\tilde y$")
        ax_cdf.set_ylim(-0.05, 1.05)
    else:
        ax_fun = ax_cdf
        l1 = []

    l2 = ax_fun.step(
        xx,
        jnp.searchsorted(opt_thresholds, xx) - 1,
        color=linecolor,
        linewidth=1,
        label="$Q(\\tilde y)$",
    )

    ax_fun.yaxis.tick_left()
    ax_fun.yaxis.label.set_color(color)
    ax_fun.tick_params(axis="y", colors=color)
    ax_fun.yaxis.set_label_position("left")
    ax_fun.set_ylabel("level")

    ax_fun.xaxis.set_major_locator(xticklocator)
    ax_fun.yaxis.set_major_locator(yticklocator)
    ax_fun.set_ylim(-0.05 * (2**num_bits - 1), 1.05 * (2**num_bits - 1))

    return ax_cdf, ax_fun, l1 + l2


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
