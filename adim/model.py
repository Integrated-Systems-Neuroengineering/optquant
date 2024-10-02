from typing import Union
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dists
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import hpdi


def model(
    vector_length: int,
    distribution: Union[dists.Distribution, str] = "bipolar",
    noise_std: float = 0.1,
    thresholds: jnp.ndarray = jnp.array([0.5]),
):
    scale = 1.0
    offset = 0.0

    if isinstance(distribution, str):
        if distribution == "normal":
            distribution = dists.Normal(0, 1)
        elif distribution == "binary":
            distribution = dists.Bernoulli(0.5)
        elif distribution == "bipolar":
            distribution = dists.Bernoulli(0.5)
            scale = 2.0
            offset = -1.0
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    # v1, v2: the two random vectors to be correlated
    with numpyro.plate("vectors", vector_length):
        v1 = numpyro.sample("v1", distribution) * scale + offset
        v2 = numpyro.sample("v2", distribution) * scale + offset

    # Compute correct product
    n = numpyro.deterministic("n", jnp.dot(v1, v2))

    # Compute the noisy measurement
    x = numpyro.sample("x", dists.Normal(n, noise_std))

    # compute the quantized version of the noisy measurement
    y = numpyro.deterministic("y", jnp.searchsorted(thresholds, x))
