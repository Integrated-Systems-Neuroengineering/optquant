import jax
import jax.numpy as jnp
from tqdm import tqdm
import xarray
import numpyro.distributions as dists
from functools import partial
from .utils import compute_even_levels

jax.config.update("jax_enable_x64", True)


class EmpiricalDistribution(object):
    def __init__(
        self, support, probs=None, log_probs=None, issorted=False, isnormalized=True
    ):
        """Initialize an empirical distribution.

        Args:
            support: The support of the empirical distribution.
            probs: The probabilities of the empirical distribution.
            log_probs: The log probabilities of the empirical distribution.
            issorted: Whether the support is sorted.
            isnormalized: Whether the probabilities are normalized.
        """

        if log_probs is None and probs is not None:
            probs = jnp.atleast_1d(probs)
            log_probs = jnp.log(probs)
        elif probs is None and log_probs is not None:
            log_probs = jnp.atleast_1d(log_probs)
            probs = jnp.exp(log_probs)
        else:
            raise ValueError("Exactly one of probs and log_probs must be provided.")

        support = jnp.atleast_1d(support)
        assert (
            support.shape == probs.shape == log_probs.shape
        ), "support, probs, and log_probs must have the same shape."

        if not isnormalized:
            probs /= probs.sum()
            log_probs = jnp.log(probs)

        if not issorted:
            idx = jnp.argsort(support)
            support = support[idx]
            probs = probs[idx]

        self.support = support
        self.probs = probs
        self.log_probs = log_probs
        self.empirical_cdf = jnp.cumsum(probs)
        self.mean = jnp.dot(support, probs)
        self.var = jnp.dot((support - self.mean) ** 2, probs)
        self.std = jnp.sqrt(self.var)

    def split(self, val):
        idx = self.get_left_index(val)
        d1 = EmpiricalDistribution(
            self.support[:idx],
            log_probs=self.log_probs[:idx],
            issorted=True,
            isnormalized=False,
        )
        p1 = self.cdf(self.support[idx])
        d2 = EmpiricalDistribution(
            self.support[idx:],
            log_probs=self.log_probs[idx:],
            issorted=True,
            isnormalized=False,
        )
        return (p1, d1, 1 - p1, d2)

    def entropy(self):
        return -jnp.dot(self.probs, self.log_probs)

    def get_left_index(self, x):
        return jnp.clip(
            0, jnp.searchsorted(self.support, x, side="left"), len(self.support) - 1
        )

    def prob(self, x):
        idx = self.get_left_index(x)
        return jnp.where(self.support[idx] == x, self.probs[idx], 0)

    def log_prob(self, x):
        idx = self.get_left_index(x)
        return jnp.where(self.support[idx] == x, self.log_probs[idx], -jnp.inf)

    def cdf(self, x):
        """Compute the CDF of the empirical distribution at the given points.

        Args:
            x: The points at which to evaluate the CDF.

        Returns:
            The CDF at the given points.
        """
        idx = self.get_left_index(x)
        return self.empirical_cdf[idx]

    def icdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the inverse CDF of the empirical distribution at the given points.

        Args:
            x: The points at which to evaluate the inverse CDF.

        Returns:
            The inverse CDF at the given points.
        """
        idx = jnp.searchsorted(self.empirical_cdf, x, side="right")
        idx_lower = jnp.clip(0, idx - 1, len(self.support) - 2)
        return jnp.where(
            idx == 0,
            -jnp.inf,
            jnp.where(
                idx == len(self.support),
                jnp.inf,
                (self.support[idx_lower] + self.support[idx_lower + 1]) / 2,
            ),
        )

    def transform(self: "EmpiricalDistribution", fun):
        mapped_support = fun(self.support)
        support = jnp.unique(mapped_support)
        mapped_support_index = jnp.searchsorted(support, mapped_support)
        # sum probabilities grouped by support
        probs = (
            jnp.zeros_like(support, dtype=float)
            .at[mapped_support_index]
            .add(self.probs)
        )
        return EmpiricalDistribution(support, probs / probs.sum())


def enob(vector_length):
    return jnp.log2(2 * jnp.e * jnp.pi * 0.25 * vector_length) * 0.5


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
    shape = jnp.broadcast_shapes(
        thresholds.shape[:-1], jnp.shape(y), jnp.shape(n), jnp.shape(noise_std)
    )
    y = jnp.broadcast_to(y, shape)
    n = jnp.broadcast_to(n, shape)
    noise_std = jnp.broadcast_to(noise_std, shape)

    idx = jnp.ix_(*[jnp.arange(i) for i in shape])

    upper = thresholds[*idx, y + 1]
    lower = thresholds[*idx, y]

    return jnp.where(
        noise_std <= 1e-16,
        jnp.where(
            (n >= lower) & (n < upper),
            0.0,
            -jnp.inf,
        ),
        jnp.log(((d := dists.Normal(n, noise_std)).cdf(upper) - d.cdf(lower))),
    )


def log_pdf_yn(
    P_n: EmpiricalDistribution,
    y: jnp.ndarray,
    n: jnp.ndarray,
    thresholds: jnp.ndarray,
    noise_std: float = None,
):
    """
    Compute the joint log-probability of observing the quantized measurement y and the true integer product n.
    """
    return log_pdf_y_n(y, n, thresholds, noise_std) + P_n.log_prob(n)


def log_pdf_y(
    P_n: EmpiricalDistribution,
    y: jnp.ndarray,
    thresholds: jnp.ndarray,
    noise_std: float = None,
):
    """
    Compute the log-probability of observing the quantized measurement y.
    """
    ns = P_n.support

    trial_dims = jnp.broadcast_shapes(
        jnp.shape(thresholds)[:-1], jnp.shape(noise_std), jnp.shape(y)
    )

    ns = ns[:, *(jnp.newaxis,) * len(trial_dims)]
    y = jnp.atleast_1d(y)[jnp.newaxis, ...]
    thresholds = jnp.atleast_1d(thresholds)[jnp.newaxis, ...]
    noise_std = jnp.atleast_1d(noise_std)[jnp.newaxis, ...]

    terms = log_pdf_yn(P_n, y, ns, thresholds, noise_std)
    return jax.scipy.special.logsumexp(terms, axis=0, keepdims=False)


@partial(jax.jit, static_argnames=("P_n",))
def MI(
    P_n: EmpiricalDistribution,
    thresholds: jnp.ndarray,
    noise_std: float = 0,
    transform: jnp.array = None,
):
    """
    Compute the mutual information between the quantized measurement y and the true product n.
    """

    trial_dims = jnp.broadcast_shapes(jnp.shape(thresholds)[:-1], jnp.shape(noise_std))

    # map over all possible combinations of y and n
    ys = jnp.arange(thresholds.shape[-1] - 1)
    ns = P_n.support

    if transform is not None:
        zs = jnp.arange(transform.shape[-1] - 1)
        zs = zs[jnp.newaxis, :, jnp.newaxis, *(jnp.newaxis,) * len(trial_dims)]
        transform = transform[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...]

    # add extra dimensions to ys and ns to sum over
    ys = ys[:, jnp.newaxis, jnp.newaxis, *(jnp.newaxis,) * len(trial_dims)]
    ns = ns[jnp.newaxis, jnp.newaxis, :, *(jnp.newaxis,) * len(trial_dims)]
    noise_std = jnp.array(noise_std)[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...]
    thresholds = thresholds[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...]

    # compute the probabilities
    l_p_n = P_n.log_prob(ns)

    l_p_y_n = log_pdf_y_n(ys, ns, thresholds, noise_std)
    l_p_y = log_pdf_y(P_n, ys, thresholds, noise_std)

    def entropy(log_p, **kwargs):
        return jnp.sum(
            jnp.where(jnp.isfinite(log_p), -jnp.exp(log_p) * log_p, 0.0), **kwargs
        ) / jnp.log(2)

    h_y = entropy(l_p_y, axis=0, keepdims=True)

    if transform is not None:
        l_p_z_n = log_pdf_y_n(zs, ns, transform, noise_std)
        l_p_z = log_pdf_y(P_n, zs, transform, noise_std)

        l_p_yzn = l_p_n + l_p_y_n + l_p_z_n
        l_p_yz = jax.scipy.special.logsumexp(l_p_yzn, axis=2, keepdims=True)

        h_z = entropy(l_p_z, axis=1, keepdims=True)
        h_yz = entropy(l_p_yz, axis=(0, 1), keepdims=True)
        h = h_y + h_z - h_yz
    else:
        l_p_yn = l_p_n + l_p_y_n

        h_n = entropy(l_p_n, axis=2, keepdims=True)
        h_yn = entropy(l_p_yn, axis=(0, 2), keepdims=True)
        h = h_y + h_n - h_yn

    return h.squeeze(axis=(0, 1, 2))


def sweep(
    P_n: EmpiricalDistribution,
    num_bits: jnp.ndarray,
    phase: jnp.ndarray,
    scale: jnp.ndarray,
    noise_std: jnp.ndarray,
    transform: jnp.ndarray = None,
):
    # reshape parameters:
    # - 0th dimension: number of ADC bits (added by looping)
    # - 1st dimension: noise standard deviation
    # - 2nd dimension: phase offset
    # - 3rd dimension: scale factor
    (
        noise_std,
        phase,
        scale,
    ) = jnp.ix_(jnp.atleast_1d(noise_std), jnp.atleast_1d(phase), jnp.atleast_1d(scale))

    num_bits = jnp.atleast_1d(num_bits)

    if transform is not None:
        transform = jnp.atleast_1d(transform)[
            jnp.newaxis, jnp.newaxis, jnp.newaxis, ...
        ]

    MIs = []
    for b in tqdm(num_bits):
        # compute thresholds
        # thresholds = compute_thresholds(b, phase, scale, distribution, vector_length)
        thresholds = compute_even_levels(
            2**b, alpha=scale, beta=phase, offset_fixed=P_n.support.mean()
        )

        MIs.append(
            MI(
                P_n=P_n,
                thresholds=thresholds,
                noise_std=noise_std,
                transform=transform,
            )
        )

    return jnp.stack(MIs, axis=0)


@partial(jax.jit, static_argnums=(2,))
def MI_nonoise(alpha, beta, K, values, probabilities):

    levels = compute_even_levels(
        K, alpha, beta, offset_fixed=(values[0] + values[-1]) / 2
    )
    alpha, beta = (
        jnp.atleast_1d(alpha)[..., jnp.newaxis],
        jnp.atleast_1d(beta)[..., jnp.newaxis],
    )

    cumprobs = jnp.concat([jnp.array([0.0]), jnp.cumsum(probabilities)])

    p_s = (
        cumprobs[jnp.searchsorted(values, levels[..., 1:])]
        - cumprobs[jnp.searchsorted(values, levels[..., :-1])]
    )

    return -jnp.sum(p_s * jnp.log2(p_s), where=p_s > 0, axis=-1)


p_Y = jax.scipy.stats.norm.pdf
F_Y = jax.scipy.stats.norm.cdf
prec = 1e-16


@partial(jax.jit, static_argnames=("levels"))
def H_normal_partition(a, b, levels, mu=0.0, sigma=1.0):
    inner_levels = levels - 2

    def step(y, carry):
        lower_edge = a * (y - inner_levels / 2) + b
        upper_edge = lower_edge + a
        # if the bin is very small, use a linear approximation of the integral
        term1 = jnp.where(
            a > 1e-3,
            F_Y(upper_edge, loc=mu, scale=sigma) - F_Y(lower_edge, loc=mu, scale=sigma),
            a * p_Y(lower_edge + a / 2, loc=mu, scale=sigma),
        )
        term2 = -jnp.log2(term1)
        prod = term1 * term2
        return carry + jnp.where(term1 > prec, prod, 0.0)

    # account for the entropy of the outer bins
    outer = 0.0
    term1 = F_Y(b - a * inner_levels / 2, loc=mu, scale=sigma)
    prod = term1 * -jnp.log2(term1)
    outer += jnp.where(term1 > prec, prod, 0.0)
    term1 = 1 - F_Y(b + a * inner_levels / 2, loc=mu, scale=sigma)
    prod = term1 * -jnp.log2(term1)
    outer += jnp.where(term1 > prec, prod, 0.0)
    return jax.lax.fori_loop(0, levels - 2, step, outer)


@partial(jax.jit, static_argnames=("levels"))
def dH_normal_partition(a, b, levels, mu=0.0, sigma=1.0):
    inner_levels = levels - 2

    def step(y, carry):
        lower_edge = a * (y - inner_levels / 2) + b
        upper_edge = lower_edge + a
        # if the bin is very small, use a linear approximation of the integral
        term1 = jnp.where(
            a > 1e3,
            F_Y(upper_edge, loc=mu, scale=sigma) - F_Y(lower_edge, loc=mu, scale=sigma),
            a * p_Y(lower_edge + a / 2, loc=mu, scale=sigma),
        )
        term2 = -jnp.log2(term1)
        dterm1_a = p_Y(upper_edge, loc=mu, scale=sigma) * (
            y - inner_levels / 2 + 1
        ) - p_Y(lower_edge, loc=mu, scale=sigma) * (y - inner_levels / 2)
        dterm1_b = p_Y(upper_edge, loc=mu, scale=sigma) - p_Y(
            lower_edge, loc=mu, scale=sigma
        )
        dterm1 = jnp.array([dterm1_a, dterm1_b])
        prod = dterm1 * (term2 - 1.0)
        return carry + jnp.where(term1 > prec, prod, 0.0)

    # account for the entropy of the outer bins
    outer = jnp.array([0.0, 0.0])
    term1 = F_Y(b - a * inner_levels / 2, loc=mu, scale=sigma)
    term2 = -jnp.log2(term1)

    dterm1_a = p_Y(b - a * inner_levels / 2, loc=mu, scale=sigma) * -inner_levels / 2
    dterm1_b = p_Y(b - a * inner_levels / 2, loc=mu, scale=sigma)
    dterm1 = jnp.array([dterm1_a, dterm1_b])
    prod = dterm1 * (term2 - 1.0)
    outer += jnp.where(term1 > prec, prod, 0.0)

    term1 = 1 - F_Y(b + a * inner_levels / 2, loc=mu, scale=sigma)
    term2 = -jnp.log2(term1)

    dterm1_a = -p_Y(b + a * inner_levels / 2, loc=mu, scale=sigma) * inner_levels / 2
    dterm1_b = -p_Y(b + a * inner_levels / 2, loc=mu, scale=sigma)
    dterm1 = jnp.array([dterm1_a, dterm1_b])
    prod = dterm1 * (term2 - 1.0)
    outer += jnp.where(term1 > prec, prod, 0.0)

    return jax.lax.fori_loop(0, levels - 2, step, outer)


dH_normal_parition_jax = jax.grad(H_normal_partition, argnums=(0, 1))


@partial(
    jax.jit,
    static_argnames=(
        "levels",
        "num_steps",
    ),
)
def uniform_normal_partition(levels, eta=0.1, mu=0.0, sigma=1.0, num_steps=2000):
    def cond(idab):
        i, delta, _, _ = idab
        return (i < num_steps) * (delta > 1e-8)

    def step(idab):
        i, _, a, b = idab
        # (da, db) = dH(a, b, levels=levels, mu=mu, sigma=sigma)
        (da, db) = dH_normal_partition(a, b, levels=levels, mu=mu, sigma=sigma)
        a += eta * da
        b += eta * db
        return (i + 1, jnp.sqrt(da**2 + db**2), a, b)

    eta *= sigma / levels
    a, b = 1.0 / levels * sigma, 0.0
    return jax.lax.while_loop(cond, step, (0, jnp.inf, a, b))[2:]
