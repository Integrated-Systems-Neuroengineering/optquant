from matplotlib import pyplot as plt
from jax import numpy as jnp


def compute_even_levels(num_bins, alpha, beta, offset_fixed=0):
    alpha, beta, offset_fixed = jnp.broadcast_arrays(
        jnp.atleast_1d(alpha), jnp.atleast_1d(beta), jnp.atleast_1d(offset_fixed)
    )

    i = jnp.arange(num_bins + 1).reshape(*([1] * (alpha.ndim - 1)), -1)
    offset_scaled = -(num_bins + 1) / 2
    res = (
        alpha[..., jnp.newaxis] * (i + 0.5 + offset_scaled)
        + beta[..., jnp.newaxis]
        + offset_fixed[..., jnp.newaxis]
    )
    return res.at[..., 0].set(-jnp.inf).at[..., -1].set(jnp.inf)


def share_axis(ax, sharex="all", sharey="all"):
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            hidex, hidey = False, False

            if sharex != "none" and sharex is not None:
                # share with the correct axis
                if sharex == "row":
                    ax[i, j].sharex(ax[i, 0])
                elif sharex == "col":
                    ax[i, j].sharex(ax[0, j])
                    if i != 0:
                        hidex = True
                elif sharex == "all":
                    ax[j, 1].sharex(ax[0, 0])
                    if i != 0:
                        hidex = True
                else:
                    raise ValueError(f"Got unknown sharex argument '{sharex}'")

            if sharey != "none" and sharey is not None:
                if sharey == "row":
                    ax[i, j].sharey(ax[i, 0])
                    if j != 0:
                        hidey = True
                elif sharey == "col":
                    ax[i, j].sharey(ax[0, j])
                elif sharey == "all":
                    ax[j, 1].sharey(ax[0, 0])
                    if j != 0:
                        hidey = True
                else:
                    raise ValueError(f"Got unknown sharey argument '{sharey}'")

            if hidex:
                # hide the x-axis label and tick labels by setting them to invisible
                plt.setp(ax[i, j].get_xticklabels(), visible=False)
                ax[i, j].xaxis.label.set_visible(False)

            if hidey:
                # hide the y-axis label and tick labels by setting them to invisible
                plt.setp(ax[i, j].get_yticklabels(), visible=False)
                ax[i, j].yaxis.label.set_visible(False)
