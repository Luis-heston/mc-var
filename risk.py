"""Core risk maths: correlated Monte Carlo for two assets, VaR and Expected Shortfall."""
import numpy as np


def estimate_params(log_returns):
    """log_returns: (n_days, 2) array. Returns daily mean vector and covariance matrix."""
    mu = log_returns.mean(axis=0)
    cov = np.cov(log_returns, rowvar=False)
    return mu, cov


def _factor(cov):
    """Return L with L @ L.T == cov.

    Cholesky when possible; eigen-decomposition fallback for singular or near-singular
    matrices (e.g. two tickers that move almost identically).
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        return vecs * np.sqrt(np.clip(vals, 0, None))


def corr_matrix(cov):
    """Correlation matrix from a covariance matrix."""
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def simulate_terminal_returns(mu, cov, horizon_days, n_sims, seed=None):
    """Simulate cumulative log returns over the horizon for N correlated assets.

    Multivariate normal with a Cholesky factor: R = T*mu + sqrt(T) * L @ z
    Returns array (n_sims, N).
    """
    rng = np.random.default_rng(seed)
    L = _factor(cov)
    z = rng.standard_normal((n_sims, len(mu)))
    return horizon_days * mu + np.sqrt(horizon_days) * (z @ L.T)


def simulate_paths(mu, cov, horizon_days, n_paths, seed=None):
    """Daily paths of cumulative log returns, shape (n_paths, horizon_days + 1, N). Used for plotting."""
    rng = np.random.default_rng(seed)
    L = _factor(cov)
    z = rng.standard_normal((n_paths, horizon_days, len(mu)))
    daily = mu + z @ L.T
    paths = np.concatenate([np.zeros((n_paths, 1, len(mu))), np.cumsum(daily, axis=1)], axis=1)
    return paths


def portfolio_pnl(terminal_log_returns, weights, portfolio_value):
    """P&L in currency units given cumulative log returns and weights (sum to 1)."""
    growth = np.exp(terminal_log_returns)  # (n, 2) price relative
    final = portfolio_value * (growth @ np.asarray(weights))
    return final - portfolio_value


def var_es(pnl, confidence):
    """Return (VaR, ES) as positive loss numbers at the given confidence (e.g. 0.95)."""
    q = np.percentile(pnl, 100 * (1 - confidence))
    var = -q
    tail = pnl[pnl <= q]
    es = -tail.mean() if tail.size else var
    return float(var), float(es)


def parametric_var(mu, cov, weights, horizon_days, confidence, portfolio_value):
    """Delta-normal VaR on arithmetic approximation, for comparison."""
    from statistics import NormalDist

    w = np.asarray(weights)
    mean = horizon_days * (w @ mu)
    std = np.sqrt(horizon_days * (w @ cov @ w))
    z = NormalDist().inv_cdf(confidence)
    return float(portfolio_value * (z * std - mean))


def historical_var(log_returns, weights, horizon_days, confidence, portfolio_value):
    """Historical-simulation VaR using overlapping horizon-day windows (1-day if horizon=1)."""
    cum = np.cumsum(log_returns, axis=0)
    cum = np.vstack([np.zeros((1, cum.shape[1])), cum])
    window = cum[horizon_days:] - cum[:-horizon_days]
    if len(window) < 30:
        return float("nan")
    pnl = portfolio_pnl(window, weights, portfolio_value)
    return var_es(pnl, confidence)[0]
