"""Inference for the marginal and dependence models.

Method summary (deliberately explicit, because the missing-data handling is
part of the result):

*   Each site's marginal is fitted on that site's **own observed months**
    (independence of irrelevant alternatives / IFM step 1).  Three positive
    severity families are compared on the training period by AIC, then the
    selected family is refitted on the full observed history for scenario
    generation.
*   Dependence is fitted only on **listwise-complete months** across the
    selected sites.  Sites are never ranked individually on their own calendars
    and then "glued" together: the uniform scores for a month exist only when
    every selected site is observed in that month.  Pairwise counts of common
    months are reported so a reader can see how thin a pair is.
*   The rank correlation matrix is estimated with Kendall's tau and inverted
    with ``sin(pi * tau / 2)``.  Both copulas share this matrix, so the AIC
    comparison isolates the tail parameter rather than a different correlation
    estimate.  The matrix is projected to the nearest valid correlation matrix
    (symmetric, unit diagonal, positive semi-definite) and the adjustment is
    reported.
*   Two dependence hypotheses are compared: a Gaussian copula (no tail
    dependence) and a Student-t copula (symmetric tail dependence).  Selection
    happens on the training period only.  The evaluation period is used solely
    to check joint-exceedance coverage, never to choose the model.
*   Joint scenarios are regenerated for the exact membership set, with a
    signature that changes whenever the site set (or fit) changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import stats

EPS = 1e-6
U_CLIP = 1e-9


# ---------------------------------------------------------------------------
# Marginal severity models
# ---------------------------------------------------------------------------


def _dist(kind: str, params: dict[str, float]):
    if kind == "lognormal":
        return stats.lognorm(s=params["sigma"], scale=math.exp(params["mu"]))
    if kind == "gamma":
        return stats.gamma(a=params["shape"], scale=params["scale"])
    if kind == "weibull":
        return stats.weibull_min(c=params["shape"], scale=params["scale"])
    raise ValueError(kind)


def _fit_one(kind: str, x: np.ndarray) -> dict[str, Any] | None:
    if x.size < 8 or np.any(x <= 0):
        return None
    try:
        if kind == "lognormal":
            lx = np.log(x)
            mu = float(lx.mean())
            sigma = float(lx.std(ddof=0))
            params = {"mu": mu, "sigma": sigma}
        elif kind == "gamma":
            shape, _loc, scale = stats.gamma.fit(x, floc=0)
            params = {"shape": float(shape), "scale": float(scale)}
        elif kind == "weibull":
            shape, _loc, scale = stats.weibull_min.fit(x, floc=0)
            params = {"shape": float(shape), "scale": float(scale)}
        else:
            raise ValueError(kind)
        dist = _dist(kind, params)
        ll = float(np.sum(dist.logpdf(x)))
        if not np.isfinite(ll):
            return None
        return {
            "kind": kind,
            "params": params,
            "loglik": ll,
            "k": 2,
            "aic": 2 * 2 - 2 * ll,
        }
    except Exception:
        return None


def fit_marginal_select(x_train: np.ndarray, x_full: np.ndarray) -> dict[str, Any]:
    """Compare severity families on training data, refit the winner on all."""
    candidates = []
    for kind in ("lognormal", "gamma", "weibull"):
        fit = _fit_one(kind, x_train)
        if fit is not None:
            candidates.append(fit)
    if not candidates:
        raise ValueError("marginal fit failed: not enough positive observations")
    best = min(candidates, key=lambda c: c["aic"])
    full = _fit_one(best["kind"], x_full)
    if full is None:  # pragma: no cover - defensive
        raise ValueError("marginal refit failed")
    dist = _dist(full["kind"], full["params"])
    order = np.sort(x_full)
    n = order.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    fitted = dist.ppf(probs)
    ks = stats.kstest(x_full, dist.cdf)
    return {
        "kind": full["kind"],
        "params": full["params"],
        "loglik": full["loglik"],
        "aic": full["aic"],
        "train": {
            "kind": best["kind"],
            "params": best["params"],
            "aic": best["aic"],
            "candidates": [
                {"kind": c["kind"], "aic": round(float(c["aic"]), 4)}
                for c in sorted(candidates, key=lambda c: c["aic"])
            ],
        },
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "n_obs": int(n),
        "qq": [
            [round(float(a), 4), round(float(b), 4)]
            for a, b in _thin(list(zip(order, fitted)), 70)
        ],
        "dist": dist,
    }


def _thin(seq: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    if len(seq) <= limit:
        return seq
    idx = np.linspace(0, len(seq) - 1, limit).round().astype(int)
    return [seq[i] for i in idx]


# ---------------------------------------------------------------------------
# Correlation matrix legality
# ---------------------------------------------------------------------------


def _eig_min(R: np.ndarray) -> float:
    return float(np.linalg.eigvalsh((R + R.T) / 2).min())


def nearest_correlation(R: np.ndarray, tol: float = 1e-10, max_iter: int = 200):
    """Project a symmetric matrix with unit diagonal onto the PSD cone.

    Simple alternating projection: clip negative eigenvalues, then rescale the
    diagonal back to one.  Iterate until both conditions hold.  Returns the
    matrix plus adjustment diagnostics.
    """
    A = np.array(R, dtype=float, copy=True)
    np.fill_diagonal(A, 1.0)
    A = (A + A.T) / 2
    start = _eig_min(A)
    p = A.shape[0]
    for _ in range(max_iter):
        vals, vecs = np.linalg.eigh(A)
        if vals.min() >= tol and np.allclose(np.diag(A), 1.0, atol=1e-12):
            break
        vals = np.clip(vals, tol, None)
        A = (vecs * vals) @ vecs.T
        d = np.sqrt(np.clip(np.diag(A), 1e-12, None))
        A = A / np.outer(d, d)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 1.0)
    off = A[~np.eye(p, dtype=bool)]
    return A, {
        "min_eig_before": round(start, 8),
        "min_eig_after": round(_eig_min(A), 8),
        "max_abs_adjust": round(float(np.max(np.abs(A - R))), 8),
        "condition": round(float(np.linalg.cond(A)), 6),
        "adjusted": bool(np.max(np.abs(A - R)) > 1e-8),
    }


def validate_correlation(R: np.ndarray) -> dict[str, Any]:
    """Hard legality checks used both by the API and by the test-suite."""
    ok_shape = R.ndim == 2 and R.shape[0] == R.shape[1]
    finite = bool(np.all(np.isfinite(R)))
    sym = bool(finite and np.allclose(R, R.T, atol=1e-9))
    unit = bool(finite and np.allclose(np.diag(R), 1.0, atol=1e-9))
    bounded = bool(finite and np.all(np.abs(R) <= 1 + 1e-9))
    psd = bool(finite and _eig_min(R) >= -1e-8)
    return {
        "shape": ok_shape,
        "finite": finite,
        "symmetric": sym,
        "unit_diagonal": unit,
        "bounded": bounded,
        "psd": psd,
        "legal": bool(ok_shape and finite and sym and unit and bounded and psd),
    }


# ---------------------------------------------------------------------------
# Copulas
# ---------------------------------------------------------------------------


def uniform_scores(x: np.ndarray, dist) -> np.ndarray:
    return np.clip(dist.cdf(x), U_CLIP, 1 - U_CLIP)


def kendall_matrix(U: np.ndarray) -> np.ndarray:
    p = U.shape[1]
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            tau = float(stats.kendalltau(U[:, i], U[:, j]).statistic)
            if not np.isfinite(tau):
                tau = 0.0
            r = math.sin(math.pi * tau / 2.0)
            R[i, j] = R[j, i] = float(np.clip(r, -0.999, 0.999))
    return R


def gaussian_loglik(R: np.ndarray, Z: np.ndarray) -> float:
    n, p = Z.shape
    _sign, logdet = np.linalg.slogdet(R)
    if not np.isfinite(logdet):
        return -np.inf
    sol = np.linalg.solve(R, Z.T).T
    quad = float(np.sum(Z * sol))
    return float(-0.5 * (n * logdet + quad - np.sum(Z * Z)))


def t_loglik(R: np.ndarray, U: np.ndarray, nu: float) -> float:
    """Student-t copula log-likelihood with corr matrix R and df nu."""
    n, p = U.shape
    _sign, logdet = np.linalg.slogdet(R)
    if not np.isfinite(logdet):
        return -np.inf
    Z = stats.t.ppf(U, nu)
    if not np.all(np.isfinite(Z)):
        return -np.inf
    sol = np.linalg.solve(R, Z.T).T
    quad = np.sum(Z * sol, axis=1)
    c = (
        math.lgamma((nu + p) / 2)
        + (p - 1) * math.lgamma(nu / 2)
        - p * math.lgamma((nu + 1) / 2)
    )
    ll = (
        n * c
        - 0.5 * n * logdet
        - (nu + p) / 2 * np.sum(np.log1p(quad / nu))
        + (nu + 1) / 2 * np.sum(np.log1p(Z * Z / nu))
    )
    return float(ll)


def fit_t_df(R: np.ndarray, U: np.ndarray) -> dict[str, Any]:
    from scipy.optimize import minimize_scalar

    def neg(log_nu: float) -> float:
        nu = math.exp(log_nu)
        ll = t_loglik(R, U, nu)
        return -ll if np.isfinite(ll) else 1e12

    res = minimize_scalar(neg, bounds=(math.log(2.05), math.log(150.0)), method="bounded")
    best_nu = float(math.exp(res.x))
    best_ll = float(-res.fun)
    # refine around optimum on a small grid for a stable reported df
    grid = np.array([2.1, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 90, 150])
    lls = np.array([t_loglik(R, U, float(nu)) for nu in grid])
    k = int(np.argmax(lls))
    if lls[k] > best_ll:
        best_nu, best_ll = float(grid[k]), float(lls[k])
    # asymptotic independence is effectively Gaussian; cap the reported df
    # at 150 but keep the likelihood for AIC.
    return {"nu": round(best_nu, 3), "loglik": round(best_ll, 4)}


def t_tail_dependence(R: np.ndarray, nu: float) -> np.ndarray:
    p = R.shape[0]
    out = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            r = float(np.clip(R[i, j], -0.999, 0.999))
            lam = 2 * float(stats.t.cdf(-math.sqrt((nu + 1) * (1 - r) / (1 + r)), nu + 1))
            out[i, j] = out[j, i] = lam
    return out


def simulate_uniforms(
    R: np.ndarray, n: int, rng: np.random.Generator, nu: float | None
) -> np.ndarray:
    """Draw copula uniforms.  ``nu is None`` means the Gaussian copula."""
    vals, vecs = np.linalg.eigh(R)
    vals = np.clip(vals, 1e-12, None)
    factor = vecs * np.sqrt(vals)
    z = rng.standard_normal((n, R.shape[0])) @ factor.T
    if nu is None:
        return stats.norm.cdf(z)
    w = rng.chisquare(nu, size=n)
    t = z / np.sqrt(w / nu)[:, None]
    return stats.t.cdf(t, nu)


def draw_losses(U: np.ndarray, dists: Iterable) -> np.ndarray:
    return np.column_stack([d.ppf(np.clip(U[:, j], U_CLIP, 1 - U_CLIP)) for j, d in enumerate(dists)])


# ---------------------------------------------------------------------------
# Scenario summaries
# ---------------------------------------------------------------------------


def _q(a: np.ndarray, p: float) -> float:
    return float(np.quantile(a, p))


def _tail_mean(a: np.ndarray, p: float) -> float:
    q = np.quantile(a, p)
    tail = a[a >= q]
    return float(tail.mean()) if tail.size else float("nan")


def scenario_summary(
    losses: np.ndarray, codes: list[str], q95: float
) -> dict[str, Any]:
    """Summarise simulated scenarios.

    ``q95`` is the total-loss 95% quantile of the reference model, so the
    "tail" column is a like-for-like comparison across dependency hypotheses
    rather than a re-quantiling of each simulation.
    """
    total = losses.sum(axis=1)
    tail_mask = total > q95
    sites = []
    for j, code in enumerate(codes):
        col = losses[:, j]
        tail_contrib = float(col[tail_mask].mean()) if tail_mask.any() else float("nan")
        sites.append(
            {
                "code": code,
                "mean": float(col.mean()),
                "sd": float(col.std(ddof=1)),
                "share_mean": float(col.mean() / total.mean()),
                "tail_mean": tail_contrib,
                "tail_share": float(tail_contrib / total[tail_mask].mean())
                if tail_mask.any()
                else float("nan"),
                "q95": _q(col, 0.95),
            }
        )
    return {
        "n": int(losses.shape[0]),
        "total": {
            "mean": float(total.mean()),
            "sd": float(total.std(ddof=1)),
            "q50": _q(total, 0.5),
            "q90": _q(total, 0.9),
            "q95": _q(total, 0.95),
            "q99": _q(total, 0.99),
            "q995": _q(total, 0.995),
            "es95": _tail_mean(total, 0.95),
            "es99": _tail_mean(total, 0.99),
        },
        "sites": sites,
        "tail_threshold": float(q95),
        "survival": survival_curve(total),
    }


def exceedance_rates(
    losses: np.ndarray,
    thr_q90: list[float],
    thr_q95: list[float],
) -> dict[str, Any]:
    """Joint-exceedance rates at *fixed* per-site thresholds.

    Thresholds must come from outside the evaluated sample (training marginal
    quantiles), otherwise the rates are tautological.
    """
    n_site = losses.shape[1]
    q90 = losses > np.asarray(thr_q90)[None, :]
    q95 = losses > np.asarray(thr_q95)[None, :]
    c90 = q90.sum(axis=1)
    c95 = q95.sum(axis=1)
    p = n_site
    return {
        "all_q95": float(np.mean(c95 == p)),
        "all_q90": float(np.mean(c90 == p)),
        "at_least_two_q95": float(np.mean(c95 >= 2)),
        "at_least_two_q90": float(np.mean(c90 >= 2)),
        "mean_count_q90": float(c90.mean()),
        "mean_count_q95": float(c95.mean()),
        "site_q95_rate": [float(x) for x in q95.mean(axis=0)],
        "n": int(losses.shape[0]),
    }


def survival_curve(total: np.ndarray, grid: int = 60) -> list[list[float]]:
    qs = np.unique(np.quantile(total, np.linspace(0.5, 0.999, grid)))
    out = []
    for x in qs:
        out.append([float(x), float(np.mean(total > x))])
    return out


# ---------------------------------------------------------------------------
# Empirical tail dependence
# ---------------------------------------------------------------------------


def empirical_tail_dep(
    U: np.ndarray, i: int, j: int, threshold: float, B: int = 400, seed: int = 7
) -> dict[str, Any]:
    a = U[:, i] > threshold
    b = U[:, j] > threshold
    denom = int(a.sum())
    num = int(np.sum(a & b))
    lam = float(num / denom) if denom else float("nan")
    rng = np.random.default_rng(seed + i * 31 + j)
    boots = []
    n = U.shape[0]
    for _ in range(B):
        idx = rng.integers(0, n, n)
        aa = a[idx]
        bb = b[idx]
        d = int(aa.sum())
        if d:
            boots.append(float(np.sum(aa & bb) / d))
    lo = float(np.quantile(boots, 0.025)) if boots else float("nan")
    hi = float(np.quantile(boots, 0.975)) if boots else float("nan")
    return {
        "threshold": threshold,
        "n_common": int(n),
        "n_first_extreme": denom,
        "n_dual_extreme": num,
        "tail": lam,
        "tail_low": lo,
        "tail_high": hi,
        "stable": bool(num >= 10),
    }


def signature(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]
