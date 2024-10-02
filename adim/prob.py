import jax
import jax.numpy as jnp
import xarray
import numpyro.distributions as dists
from functools import partial

jax.config.update("jax_enable_x64", True)


def log_pdf_n(n, vector_length: int, distribution: str = "binary", fixed: int = None):
    """
    Compute the log-probability of getting the integer sum x from summing two random vectors of length vector_length.

    This assumes i.i.d values, where:
        - the product between two binary random variables is a Bernoulli random variable with p=0.25.
        - the product between two bipolar random variables is a Rademacher random variable with p=0.5.

    If 'fixed' is specified, that many elements are assumed to be deterministically set equal in both vectors.
    """
    if fixed is None:
        fixed = 0

    if distribution == "binary":
        p = 0.25
        scale = 1
        offset = 0
    elif distribution == "bipolar":
        # for bipolar: r random matches + fixed matches => vector_length - (r + fixed) mismatches
        # => n = (r + fixed) - (vector_length - (r + fixed)) = 2(r + fixed) - vector_length
        # => r = (n + vector_length) / 2 - fixed
        p = 0.5
        scale = 2
        offset = -vector_length

    return jnp.where(
        (n - offset) % scale == 0,
        dists.Binomial(vector_length - fixed, p).log_prob((n - offset) // scale),
        -jnp.inf,
    )


# drawing n=0 from 10 bipolar values with 4 fixed draws
# should be the same as drawing 1 matching and 5 mismatching from 6 binary values
assert log_pdf_n(0, 10, "bipolar", fixed=4) == dists.Binomial(6, 0.5).log_prob(
    1
), "Value mismatch"


def log_pdf_x_n(x, n, noise_std: float = 0.1):
    """
    Compute the log-probability of observing the noisy real measurement x given the true integer product n.
    """
    return dists.Normal(n, noise_std).log_prob(x)


def log_pdf_x(
    x,
    vector_length: int,
    noise_std: float = 0.1,
    distribution: str = "bipolar",
    fixed=0,
):
    """
    Compute the log-probability of observing the noisy real measurement x.
    """

    trial_dims = jnp.broadcast_shapes(
        jnp.shape(noise_std), jnp.shape(fixed), jnp.shape(x)
    )

    if distribution == "binary":
        ns = jnp.arange(vector_length + 1)
    elif distribution == "bipolar":
        ns = jnp.arange(-vector_length, vector_length + 1, 2)

    ns = ns[:, *(jnp.newaxis,) * len(trial_dims)]
    x = jnp.atleast_1d(x)[jnp.newaxis, ...]
    noise_std = jnp.atleast_1d(noise_std)[jnp.newaxis, ...]
    fixed = jnp.atleast_1d(fixed)[jnp.newaxis, ...]

    log_probs = log_pdf_x_n(x, ns, noise_std) + log_pdf_n(
        ns, vector_length, distribution, fixed=fixed
    )

    return jax.scipy.special.logsumexp(log_probs, axis=0, keepdims=False)


def log_pdf_y_n(y, n, thresholds: jnp.ndarray, noise_std: float = 0.1):
    """
    Compute the log-probability of observing the quantized measurement y given the true integer product n.
    """

    # print(
    # f"Thresholds: {thresholds.shape}, Y: {y.shape}, Thresholds[Y]: {thresholds[y].shape}"
    # )

    shape = jnp.broadcast_shapes(
        thresholds.shape[:-1], jnp.shape(y), jnp.shape(n), jnp.shape(noise_std)
    )
    y = jnp.broadcast_to(y, shape)
    n = jnp.broadcast_to(n, shape)
    noise_std = jnp.broadcast_to(noise_std, shape)

    d = dists.Normal(n, noise_std)
    idx = jnp.ix_(*[jnp.arange(i) for i in shape])

    return jnp.log((d.cdf(thresholds[*idx, y + 1]) - d.cdf(thresholds[*idx, y])))


def log_pdf_yn(
    y,
    n,
    thresholds: jnp.ndarray,
    noise_std: float = None,
    vector_length: int = None,
    distribution: str = None,
    fixed=None,
):
    """
    Compute the joint log-probability of observing the quantized measurement y and the true integer product n.
    """

    # print("Y, N:", y.shape, n.shape)

    res1 = log_pdf_y_n(y, n, thresholds, noise_std)
    res2 = log_pdf_n(
        n, vector_length=vector_length, distribution=distribution, fixed=fixed
    )
    # print(
    #     f"log P(Y|N): {res1.shape}, log P(N): {res2.shape}, log P(Y,N): {(res1+res2).shape}"
    # )
    return res1 + res2


def log_pdf_y(
    y,
    thresholds: jnp.ndarray,
    noise_std: float = None,
    vector_length: int = None,
    distribution: str = None,
    fixed=0,
):
    """
    Compute the log-probability of observing the quantized measurement y.
    """

    if distribution == "binary":
        ns = jnp.arange(vector_length + 1)
    elif distribution == "bipolar":
        ns = jnp.arange(-vector_length, vector_length + 1, 2)

    trial_dims = jnp.broadcast_shapes(
        jnp.shape(thresholds)[:-1], jnp.shape(noise_std), jnp.shape(fixed), jnp.shape(y)
    )

    ns = ns[:, *(jnp.newaxis,) * len(trial_dims)]
    y = jnp.atleast_1d(y)[jnp.newaxis, ...]
    thresholds = jnp.atleast_1d(thresholds)[jnp.newaxis, ...]
    noise_std = jnp.atleast_1d(noise_std)[jnp.newaxis, ...]
    fixed = jnp.atleast_1d(fixed)[jnp.newaxis, ...]

    terms = log_pdf_yn(y, ns, thresholds, noise_std, vector_length, distribution, fixed)
    res = jax.scipy.special.logsumexp(terms, axis=0, keepdims=False)
    # print(y.shape, terms.shape, res.shape)
    return res


@partial(jax.jit, static_argnames=("vector_length", "distribution"))
def MIbits(
    vector_length: int,
    thresholds: jnp.ndarray,
    noise_std: float = 0,
    distribution: str = "bipolar",
    fixed=0,
):
    """
    Compute the mutual information between the quantized measurement y and the true product n.
    """

    trial_dims = jnp.broadcast_shapes(
        jnp.shape(thresholds)[:-1], jnp.shape(noise_std), jnp.shape(fixed)
    )

    # map over all possible combinations of y and n
    ys = jnp.arange(thresholds.shape[-1] - 1)

    if distribution == "binary":
        ns = jnp.arange(vector_length + 1)
    elif distribution == "bipolar":
        ns = jnp.arange(-vector_length, vector_length + 1, 2)

    # add extra dimensions to ys and ns to sum over
    ys = ys[:, jnp.newaxis, *(jnp.newaxis,) * len(trial_dims)]
    ns = ns[jnp.newaxis, :, *(jnp.newaxis,) * len(trial_dims)]
    noise_std = jnp.array(noise_std)[jnp.newaxis, jnp.newaxis, ...]
    fixed = jnp.array(fixed)[jnp.newaxis, jnp.newaxis, ...]
    thresholds = thresholds[jnp.newaxis, jnp.newaxis, ...]

    # compute the probabilities
    l_p_y_n = log_pdf_y_n(ys, ns, thresholds, noise_std)
    l_p_y = log_pdf_y(
        ys, thresholds, noise_std, vector_length, distribution, fixed=fixed
    )
    l_p_n = log_pdf_n(ns, vector_length, distribution, fixed=fixed)
    l_p_yn = l_p_n + l_p_y_n

    # compute the log of each summand term
    # log2(p_y_n/p_y) = log2(p_y_n) - log2(p_y) = (log(p_y_n) - log(p_y)) / log(2)
    summand = jnp.exp(l_p_yn) * (l_p_y_n - l_p_y) / jnp.log(2)

    # sum over all possible values of y and n where the probability is >0
    res = jnp.sum(summand, axis=(0, 1), keepdims=False, where=jnp.isfinite(summand))
    return res


def sweep(
    vector_length: int,
    num_bits: jnp.ndarray,
    phase: jnp.ndarray,
    scale: jnp.ndarray,
    noise_std: jnp.ndarray,
    distribution: str = None,
    fixed: jnp.ndarray = jnp.array([0]),
    to_xarray: bool = True,
):
    # reshape parameters:
    # - 0th dimension: number of ADC bits (added by looping)
    # - 1st dimension: number of fixed bits
    # - 2nd dimension: noise standard deviation
    # - 3rd dimension: phase offset
    # - 4th dimension: scale factor
    (
        fixed,
        noise_std,
        phase,
        scale,
    ) = jnp.ix_(fixed, noise_std, phase, scale)

    num_bits = jnp.atleast_1d(num_bits)

    MIs = []
    for b in num_bits:
        # compute thresholds
        thresholds = compute_thresholds(b, phase, scale, distribution, vector_length)

        MIs.append(
            MIbits(
                vector_length=vector_length,
                thresholds=thresholds,
                noise_std=noise_std,
                distribution=distribution,
                fixed=fixed,
            )
        )

    res = jnp.stack(MIs, axis=0)

    if to_xarray:
        res = xarray.DataArray(
            res,
            dims=("num_bits", "fixed", "noise_std", "phase", "scale"),
            coords={
                "num_bits": num_bits.ravel(),
                "fixed": fixed.ravel(),
                "noise_std": noise_std.ravel(),
                "phase": phase.ravel(),
                "scale": scale.ravel(),
            },
        )

    return res


def compute_thresholds(
    num_bits: int,
    phase: jnp.ndarray,
    scale: jnp.ndarray,
    distribution: str,
    vector_length: int,
):
    if distribution == "bipolar":
        thr = (
            (
                (
                    jnp.arange((1 << num_bits) + 1)
                    - (1 << (num_bits - 1))
                    + jnp.atleast_1d(phase)[..., jnp.newaxis]
                )
                * jnp.atleast_1d(scale)[..., jnp.newaxis]
                * 2
            )
            .at[..., -1]
            .set(jnp.inf)
            .at[..., 0]
            .set(-jnp.inf)
        )
    elif distribution == "binary":
        thr = (
            (
                (
                    jnp.arange((1 << num_bits) + 1)
                    - (1 << (num_bits - 1))
                    + vector_length / 2
                    + 0.5
                    + jnp.atleast_1d(phase)[..., jnp.newaxis]
                )
                * jnp.atleast_1d(scale)[..., jnp.newaxis]
            )
            .at[..., -1]
            .set(jnp.inf)
            .at[..., 0]
            .set(-jnp.inf)
        )
    return thr.astype(jnp.float_)
