"""End-to-end fitting pipeline for the synthetic risk panel.

The pipeline is deliberately explicit about the two failure modes the page
exists to expose:

1.  **Ragged calendars.**  Each site's marginal is fitted on that site's own
    observed months, but dependence is only ever estimated from months observed
    at *every* selected site.  Copula inputs are the marginal CDF values of the
    jointly observed months, so nothing is created by sorting sites separately
    and gluing their ranks together.
2.  **Membership changes.**  A scenario set belongs to exactly one site set.
    Adding a site changes the joint sample, the marginal refit and the tail
    structure; the artifact signature changes, and the API refuses to extend an
    older scenario set column-wise.

Model selection happens on the training period only.  The evaluation period is
scored afterwards (joint-exceedance coverage at fixed training thresholds) and
never feeds back into the choice.

All amounts are synthetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from . import stats as st
from .data import DEFAULT_STUDY, StudySpec, build_study, loss_matrix, month_ym

SCENARIO_COUNT_DEFAULT = 20000
TRAIN_FRACTION_DEFAULT = 0.70
MIN_TRAIN = 30
MIN_EVAL = 10
TAIL_PRIMARY = 0.90
TAIL_SECONDARY = 0.95


class FitError(ValueError):
    """Raised when a requested fit is not statistically or legally possible."""


@dataclass(frozen=True)
class FitResult:
    payload: dict[str, Any]
    scenarios: dict[str, np.ndarray]  # model -> n x p losses
    uniforms: dict[str, np.ndarray]  # model -> n x p copula uniforms
    codes: list[str]
    signature: str


def load_study(spec: StudySpec = DEFAULT_STUDY, seed: int = 2611) -> dict[str, Any]:
    return build_study(spec, seed=seed)


def resolved_version(study: dict[str, Any], sites: list[str]) -> dict[str, Any]:
    """Identify which declared membership version a site set corresponds to."""
    key = sorted(sites)
    for v in study["study"]["membership_versions"]:
        if sorted(v["sites"]) == key:
            return {
                "label": v["label"],
                "sites": list(v["sites"]),
                "effective_month": v["effective_month"],
                "declared": True,
            }
    return {
        "label": "custom-selection",
        "sites": sorted(sites),
        "effective_month": None,
        "declared": False,
    }


# ---------------------------------------------------------------------------
# Marginal block
# ---------------------------------------------------------------------------


def _marginal_block(
    grid: np.ndarray,
    codes: list[str],
    sel: list[int],
    train_end_month: int,
) -> dict[str, Any]:
    """Fit each site's marginal on its own observed months.

    Training marginals use the site's own observed months up to the last
    training month of the joint calendar, so the family choice and thresholds
    are strictly out-of-sample for the evaluation window.  The final marginals
    used for scenario generation are refitted on all of the site's months.
    """
    out: dict[str, Any] = {}
    for code, j in zip([codes[k] for k in sel], sel):
        col = grid[:, j]
        own = np.flatnonzero(~np.isnan(col))
        own_train = own[own <= train_end_month]
        if own_train.size < 12:
            raise FitError(f"{code}: only {own_train.size} training observations")
        fit = st.fit_marginal_select(col[own_train], col[own])
        fit["obs_months"] = [int(m) for m in own]
        fit["train_months"] = [int(m) for m in own_train]
        fit["n_train"] = int(own_train.size)
        fit["mean"] = float(st._dist(fit["kind"], fit["params"]).mean())
        fit["empirical_mean"] = float(col[own].mean())
        fit["empirical_q95"] = float(np.quantile(col[own], 0.95))
        fit["q95"] = float(st._dist(fit["kind"], fit["params"]).ppf(0.95))
        train_dist = st._dist(fit["train"]["kind"], fit["train"]["params"])
        fit["train_q90"] = float(train_dist.ppf(0.90))
        fit["train_q95"] = float(train_dist.ppf(0.95))
        fit.pop("dist")
        out[code] = fit
    return out


def _dists(marginals: dict[str, Any], codes: list[str], train: bool = False):
    if train:
        return [
            st._dist(marginals[c]["train"]["kind"], marginals[c]["train"]["params"])
            for c in codes
        ]
    return [st._dist(marginals[c]["kind"], marginals[c]["params"]) for c in codes]


# ---------------------------------------------------------------------------
# Pair diagnostics: common calendar, tail estimates and their stability
# ---------------------------------------------------------------------------


def _pair_diagnostics(
    U: np.ndarray, codes: list[str], truth_corr: dict[tuple[str, str], float]
) -> list[dict[str, Any]]:
    p = U.shape[1]
    out = []
    for i in range(p):
        for j in range(i + 1, p):
            tau = float(stats.kendalltau(U[:, i], U[:, j]).statistic)
            if not np.isfinite(tau):
                tau = 0.0
            primary = st.empirical_tail_dep(U, i, j, TAIL_PRIMARY)
            secondary = st.empirical_tail_dep(U, i, j, TAIL_SECONDARY)
            key = tuple(sorted((codes[i], codes[j])))
            out.append(
                {
                    "pair": [codes[i], codes[j]],
                    "common_months": int(U.shape[0]),
                    "kendall_tau": round(tau, 5),
                    "implied_r": round(float(math.sin(math.pi * tau / 2)), 5),
                    "truth_r": round(truth_corr[key], 4) if key in truth_corr else None,
                    "thresholds": [
                        _tail_record(primary, TAIL_PRIMARY),
                        _tail_record(secondary, TAIL_SECONDARY),
                    ],
                    "n_dual_extreme": primary["n_dual_extreme"],
                    "stable": primary["stable"],
                    "message": (
                        f"{primary['n_dual_extreme']} jointly extreme "
                        f"(u>{TAIL_PRIMARY:g}) months out of {primary['n_first_extreme']} "
                        "first-site extremes: the tail estimate rests on a handful of "
                        "observations and its interval is wide"
                        if not primary["stable"]
                        else f"{primary['n_dual_extreme']} dual extremes at "
                        f"u>{TAIL_PRIMARY:g}: estimate is supported"
                    ),
                }
            )
    return out


def _tail_record(td: dict[str, Any], threshold: float) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "n_first_extreme": td["n_first_extreme"],
        "n_dual_extreme": td["n_dual_extreme"],
        "tail": None if not np.isfinite(td["tail"]) else round(td["tail"], 4),
        "tail_ci": [round(td["tail_low"], 4), round(td["tail_high"], 4)],
    }


# ---------------------------------------------------------------------------
# Copula fit
# ---------------------------------------------------------------------------


def _fit_copula(U: np.ndarray, codes: list[str]) -> dict[str, Any]:
    R_raw = st.kendall_matrix(U)
    R, corr_diag = st.nearest_correlation(R_raw)
    legality = st.validate_correlation(R)
    if not legality["legal"]:  # pragma: no cover - defensive
        raise FitError(f"correlation matrix illegal after projection: {legality}")
    p = len(codes)
    Z = stats.norm.ppf(np.clip(U, 1e-9, 1 - 1e-9))
    ll_gauss = st.gaussian_loglik(R, Z)
    t_fit = st.fit_t_df(R, U)
    k = p * (p - 1) / 2
    aic_gauss = 2 * k - 2 * ll_gauss
    aic_t = 2 * (k + 1) - 2 * t_fit["loglik"]
    preferred = "student_t" if aic_t < aic_gauss else "gaussian"
    return {
        "matrix": R,
        "matrix_raw": R_raw,
        "diagnostics": corr_diag,
        "legality": legality,
        "gaussian_loglik": round(ll_gauss, 4),
        "gaussian_aic": round(float(aic_gauss), 4),
        "student_t_loglik": t_fit["loglik"],
        "student_t_aic": round(float(aic_t), 4),
        "student_t_nu": t_fit["nu"],
        "preferred": preferred,
        "aic_gap": round(float(abs(aic_t - aic_gauss)), 4),
        "tail_dependence": np.round(st.t_tail_dependence(R, t_fit["nu"]), 5).tolist(),
        "k": int(k),
    }


def _public_copula(cop: dict[str, Any]) -> dict[str, Any]:
    return {
        "matrix": np.round(cop["matrix"], 6).tolist(),
        "matrix_raw": np.round(cop["matrix_raw"], 6).tolist(),
        "diagnostics": cop["diagnostics"],
        "legality": cop["legality"],
        "gaussian_loglik": cop["gaussian_loglik"],
        "gaussian_aic": cop["gaussian_aic"],
        "student_t_loglik": cop["student_t_loglik"],
        "student_t_aic": cop["student_t_aic"],
        "student_t_nu": cop["student_t_nu"],
        "preferred": cop["preferred"],
        "aic_gap": cop["aic_gap"],
        "k": cop["k"],
        "tail_dependence": cop["tail_dependence"],
    }


def _target_tau(R: np.ndarray) -> np.ndarray:
    """Kendall's tau implied by an elliptical correlation matrix."""
    p = R.shape[0]
    T = np.eye(p)
    for i in range(p):
        for j in range(p):
            if i != j:
                T[i, j] = 2 / math.pi * math.asin(float(np.clip(R[i, j], -1, 1)))
    return T


# ---------------------------------------------------------------------------
# Simulation and consistency checks
# ---------------------------------------------------------------------------


def _simulate_pair(
    U_Z: np.ndarray,
    W: np.ndarray,
    nu: float | None,
    dists,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn the pre-drawn normals into copula uniforms and losses.

    ``Z`` and ``W`` are shared by both copulas (common random numbers), so the
    two scenario sets are driven by the same underlying randomness and their
    difference isolates the tail assumption.
    """
    if nu is None:
        U = stats.norm.cdf(U_Z)
    else:
        U = stats.t.cdf(U_Z / np.sqrt(W / nu)[:, None], nu)
    return U, st.draw_losses(U, dists)


def _check_marginals(
    losses: np.ndarray, codes: list[str], marginals: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    n = losses.shape[0]
    for j, code in enumerate(codes):
        dist = st._dist(marginals[code]["kind"], marginals[code]["params"])
        x = losses[:, j]
        ks = stats.kstest(x, dist.cdf)
        target = float(dist.mean())
        rel = abs(float(x.mean()) / target - 1)
        # Mean agreement is a Monte Carlo question; the tolerance scales with
        # sd/sqrt(n) so a small scenario count is not read as a distributional
        # mismatch.  KS is already sample-size aware.
        se_rel = float(x.std(ddof=1)) / math.sqrt(n) / target
        tolerance = max(0.002, 3 * se_rel)
        rows.append(
            {
                "code": code,
                "ks_stat": round(float(ks.statistic), 5),
                "ks_p": round(float(ks.pvalue), 5),
                "sim_mean": round(float(x.mean()), 4),
                "fit_mean": round(target, 4),
                "sim_q95": round(float(np.quantile(x, 0.95)), 4),
                "fit_q95": round(float(dist.ppf(0.95)), 4),
                "mean_reldiff": round(float(rel), 5),
                "mean_tolerance": round(float(tolerance), 5),
                "pass": bool(ks.pvalue > 0.01 and rel <= tolerance),
            }
        )
    return rows


def _check_correlation(losses: np.ndarray, R: np.ndarray) -> dict[str, Any]:
    n, p = losses.shape
    U = np.column_stack([stats.rankdata(losses[:, j]) / (n + 1) for j in range(p)])
    sim_tau = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            sim_tau[i, j] = sim_tau[j, i] = float(
                stats.kendalltau(U[:, i], U[:, j]).statistic
            )
    target = _target_tau(R)
    off = ~np.eye(p, dtype=bool)
    diff = np.abs(sim_tau - target)
    # Kendall's tau under the null has sd about sqrt(2(2n+5)/(9n(n-1))); allow
    # three of those plus a small floor for the copula discretisation.
    tau_se = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1))) if n > 1 else float("inf")
    tolerance = max(0.03, 3 * tau_se)
    return {
        "target_tau": target.round(4).tolist(),
        "sim_tau": sim_tau.round(4).tolist(),
        "max_abs_diff": round(float(diff[off].max()), 4),
        "mean_abs_diff": round(float(diff[off].mean()), 4),
        "tolerance": round(float(tolerance), 4),
        "pass": bool(diff[off].max() < tolerance),
    }


def _check_aggregation(
    summary: dict[str, Any], marginals: dict[str, Any], codes: list[str], model: str
) -> dict[str, Any]:
    """Two separate claims, reported separately.

    * The aggregation identity ``E[sum_j X_j] = sum_j E[X_j]`` holds for the
      simulated sample *exactly* (``site_mean_sum`` is computed from the same
      columns), whatever the copula.  It is a construction check, not a fit.
    * Whether the simulated mean agrees with the **fitted marginal** means is a
      Monte Carlo question: the tolerance must scale with ``sd / sqrt(n)``, so a
      small scenario count is not mistaken for bias.
    """
    n = int(summary["n"])
    sim_mean = float(summary["total"]["mean"])
    sim_sd = float(summary["total"]["sd"])
    site_sum = float(sum(s["mean"] for s in summary["sites"]))
    fitted_mean = float(sum(marginals[c]["mean"] for c in codes))
    identity_diff = abs(sim_mean / site_sum - 1) if site_sum else float("inf")
    fitted_diff = abs(sim_mean / fitted_mean - 1) if fitted_mean else float("inf")
    se = sim_sd / math.sqrt(max(n, 1))
    # 3 standard errors of the sample mean, with a 0.2% floor for rounding.
    tolerance = max(0.002, 3 * se / fitted_mean) if fitted_mean else float("inf")
    return {
        "model": model,
        "sim_total_mean": round(sim_mean, 4),
        "sum_of_site_means": round(fitted_mean, 4),
        "site_mean_sum": round(site_sum, 4),
        "identity_reldiff": round(float(identity_diff), 8),
        "fitted_reldiff": round(float(fitted_diff), 5),
        "tolerance": round(float(tolerance), 5),
        "monte_carlo_se": round(float(se), 4),
        "n": n,
        "pass": bool(identity_diff < 1e-9 and fitted_diff <= tolerance),
        "note": (
            "E[sum of site losses] equals the sum of site means under any "
            "dependence: that identity is exact here. Agreement with the fitted "
            "marginal means is checked within a Monte Carlo tolerance of "
            "3 sd/sqrt(n), so a small scenario count is not read as bias."
        ),
    }


def _histogram(a: np.ndarray, edges: np.ndarray) -> list[int]:
    """Counts over ``edges`` with the tails folded into the end bins.

    Total loss is heavy-tailed: a handful of joint-extreme scenarios can reach
    several times the bulk.  Bin edges are chosen to cover the 99.9% range, and
    anything beyond is counted in the last bin (``>= edges[-1]``) rather than
    dropped, so the histogram still accounts for every scenario.
    """
    clipped = np.clip(a, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return [int(x) for x in counts]


def _count_distribution(losses: np.ndarray, thr: list[float]) -> list[float]:
    over = losses > np.asarray(thr)[None, :]
    c = over.sum(axis=1)
    n = losses.shape[0]
    return [round(float(np.mean(c == k)), 5) for k in range(losses.shape[1] + 1)]


def _model_block(
    model: str,
    label: str,
    nu: float | None,
    losses: np.ndarray,
    codes: list[str],
    marginals: dict[str, Any],
    R: np.ndarray,
    total_threshold: float,
    thresholds: dict[str, list[float]],
    observed_rates: dict[str, Any],
    edges: np.ndarray,
) -> dict[str, Any]:
    summary = st.scenario_summary(losses, codes, total_threshold)
    rates = st.exceedance_rates(losses, thresholds["q90"], thresholds["q95"])
    checks = {
        "marginals": _check_marginals(losses, codes, marginals),
        "correlation": _check_correlation(losses, R),
        "aggregation": _check_aggregation(summary, marginals, codes, model),
    }
    checks["all_pass"] = bool(
        all(r["pass"] for r in checks["marginals"])
        and checks["correlation"]["pass"]
        and checks["aggregation"]["pass"]
    )
    keys = ("at_least_two_q95", "all_q95", "at_least_two_q90", "all_q90")
    return {
        "label": label,
        "nu": nu,
        "summary": summary,
        "exceedance": rates,
        "checks": checks,
        "count_distribution_q95": _count_distribution(losses, thresholds["q95"]),
        "count_distribution_q90": _count_distribution(losses, thresholds["q90"]),
        "histogram": _histogram(losses.sum(axis=1), edges),
        "vs_observed": {
            "sim": {k: round(float(rates[k]), 5) for k in keys},
            "observed": {k: round(float(observed_rates[k]), 5) for k in keys},
            "ratio": {
                k: (round(float(rates[k]) / observed_rates[k], 3) if observed_rates[k] > 0 else None)
                for k in keys
            },
        },
    }


# ---------------------------------------------------------------------------
# Evaluation-period coverage
# ---------------------------------------------------------------------------


def _binomial_ci(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    lo, hi = stats.beta.ppf(
        [0.025, 0.975], [k + 0.5, k + 0.5], [n - k + 0.5, n - k + 0.5]
    )
    return [round(float(lo), 4), round(float(hi), 4)]


def _evaluation(
    obs_eval: np.ndarray,
    train_dists,
    codes: list[str],
    cop_train: dict[str, Any],
    scenario_count: int,
    seed: int,
    eval_range: list[str],
) -> dict[str, Any]:
    """Score both dependence hypotheses on the held-out period.

    Thresholds are *training-period marginal* 90th / 95th percentiles, so each
    site should exceed with probability 10% / 5%.  Nothing here feeds the
    selection: the selected model is fixed by the training AIC.
    """
    n = obs_eval.shape[0]
    thr90 = np.array([d.ppf(0.90) for d in train_dists])
    thr95 = np.array([d.ppf(0.95) for d in train_dists])
    over90 = obs_eval > thr90
    over95 = obs_eval > thr95
    c90 = over90.sum(axis=1)
    c95 = over95.sum(axis=1)
    p = len(codes)

    site_rows = []
    for j, code in enumerate(codes):
        k95 = int(over95[:, j].sum())
        k90 = int(over90[:, j].sum())
        ci95 = _binomial_ci(k95, n)
        ci90 = _binomial_ci(k90, n)
        site_rows.append(
            {
                "code": code,
                "exceed_q95": k95,
                "rate_q95": round(k95 / n, 4),
                "ci_q95": ci95,
                "exceed_q90": k90,
                "rate_q90": round(k90 / n, 4),
                "ci_q90": ci90,
                "covers_q95": bool(ci95[0] <= 0.05 <= ci95[1]),
                "covers_q90": bool(ci90[0] <= 0.10 <= ci90[1]),
            }
        )

    # Model-implied rates use the *training* copula and *training* marginals.
    rng = np.random.default_rng(seed)
    vals, vecs = np.linalg.eigh(cop_train["matrix"])
    factor = vecs * np.sqrt(np.clip(vals, 1e-12, None))
    Z = rng.standard_normal((scenario_count, p)) @ factor.T
    W = rng.chisquare(max(cop_train["student_t_nu"], 2.01), size=scenario_count)
    implied = {}
    for name, nu in (("gaussian", None), ("student_t", cop_train["student_t_nu"])):
        if nu is None:
            U = stats.norm.cdf(Z)
        else:
            U = stats.t.cdf(Z / np.sqrt(W / nu)[:, None], nu)
        sim = st.draw_losses(U, train_dists)
        o95 = sim > thr95
        o90 = sim > thr90
        implied[name] = {
            "at_least_two_q95": float(np.mean(o95.sum(axis=1) >= 2)),
            "all_q95": float(np.mean(o95.sum(axis=1) == p)),
            "at_least_two_q90": float(np.mean(o90.sum(axis=1) >= 2)),
            "all_q90": float(np.mean(o90.sum(axis=1) == p)),
        }

    def joint_block(k: int, model_rate: float) -> dict[str, Any]:
        ci = _binomial_ci(k, n)
        return {
            "observed": k,
            "rate": round(k / n, 4),
            "ci": ci,
            "model_rate": round(model_rate, 5),
            "model_expected": round(model_rate * n, 3),
            "covers": bool(ci[0] <= model_rate <= ci[1]),
        }

    joint = {}
    for name in ("gaussian", "student_t"):
        joint[name] = {
            "at_least_two_q95": joint_block(int((c95 >= 2).sum()), implied[name]["at_least_two_q95"]),
            "all_q95": joint_block(int((c95 == p).sum()), implied[name]["all_q95"]),
            "at_least_two_q90": joint_block(int((c90 >= 2).sum()), implied[name]["at_least_two_q90"]),
            "all_q90": joint_block(int((c90 == p).sum()), implied[name]["all_q90"]),
        }
    return {
        "n_eval": int(n),
        "range": eval_range,
        "sites": site_rows,
        "joint": joint,
        "model_implied": {
            k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in implied.items()
        },
        "thresholds": {
            "q90": [round(float(x), 4) for x in thr90],
            "q95": [round(float(x), 4) for x in thr95],
        },
        "note": (
            "Thresholds are training-period marginal quantiles. The evaluation "
            "window is short, so the exact binomial intervals are wide; a wide "
            "interval is reported rather than a spurious pass/fail. Selection "
            "was made on the training period only."
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fit(
    study: dict[str, Any],
    sites: list[str],
    scenario_count: int = SCENARIO_COUNT_DEFAULT,
    seed: int = 2611,
    train_fraction: float = TRAIN_FRACTION_DEFAULT,
) -> FitResult:
    all_codes = [s["code"] for s in study["sites"]]
    sites = list(dict.fromkeys(sites))
    if len(sites) < 2:
        raise FitError("select at least two sites to model dependence")
    unknown = [s for s in sites if s not in all_codes]
    if unknown:
        raise FitError(f"unknown sites: {unknown}")

    sel = [all_codes.index(s) for s in sites]
    grid, _idx, codes = loss_matrix(study)
    sub = grid[:, sel]
    complete = ~np.isnan(sub).any(axis=1)
    common_idx = np.flatnonzero(complete)
    n_common = common_idx.size
    if n_common < MIN_TRAIN + MIN_EVAL:
        raise FitError(
            f"only {n_common} jointly observed months; need at least "
            f"{MIN_TRAIN + MIN_EVAL}. Widen the calendar or drop a sparse site."
        )
    cut = int(math.floor(n_common * train_fraction))
    cut = max(MIN_TRAIN, min(cut, n_common - MIN_EVAL))
    train_idx, eval_idx = common_idx[:cut], common_idx[cut:]
    common_months = month_ym(study, common_idx)
    train_months = month_ym(study, train_idx)
    eval_months = month_ym(study, eval_idx)

    marginals = _marginal_block(grid, codes, sel, int(train_idx.max()))

    # ---- Training-only dependence fit: used for selection and evaluation ----
    train_dists = _dists(marginals, sites, train=True)
    U_train = np.column_stack(
        [st.uniform_scores(sub[train_idx, j], train_dists[j]) for j in range(len(sites))]
    )
    cop_train = _fit_copula(U_train, sites)

    # ---- Full-sample dependence fit: used for the reported scenarios --------
    full_dists = _dists(marginals, sites)
    U_full = np.column_stack(
        [st.uniform_scores(sub[common_idx, j], full_dists[j]) for j in range(len(sites))]
    )
    cop_full = _fit_copula(U_full, sites)

    truth_corr: dict[tuple[str, str], float] = {}
    truth = study["study"].get("truth", {})
    for row in truth.get("correlations", []):
        truth_corr[tuple(sorted(row["pair"]))] = float(row["corr"])
    pairs = _pair_diagnostics(U_full, sites, truth_corr)

    # ---- Scenarios, with common random numbers across the two models -------
    p = len(sites)
    vals, vecs = np.linalg.eigh(cop_full["matrix"])
    factor = vecs * np.sqrt(np.clip(vals, 1e-12, None))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((scenario_count, p)) @ factor.T
    W = rng.chisquare(max(cop_full["student_t_nu"], 2.01), size=scenario_count)

    simulated: dict[str, dict[str, np.ndarray]] = {}
    for name, nu in (("gaussian", None), ("student_t", cop_full["student_t_nu"])):
        U, losses = _simulate_pair(Z, W, nu, full_dists)
        simulated[name] = {"uniforms": U, "losses": losses}

    total_threshold = float(np.quantile(simulated["gaussian"]["losses"].sum(axis=1), 0.95))

    thresholds = {
        "q90": [
            float(st._dist(marginals[c]["kind"], marginals[c]["params"]).ppf(0.90))
            for c in sites
        ],
        "q95": [float(marginals[c]["q95"]) for c in sites],
    }
    # Empirical benchmark on the same jointly observed months, at the same
    # thresholds.  This is the "observed" column the simulations are checked
    # against; it is not used to fit anything.
    obs = sub[common_idx]
    observed_rates = st.exceedance_rates(obs, thresholds["q90"], thresholds["q95"])
    observed_summary = st.scenario_summary(obs, sites, total_threshold)
    observed_rates["count_distribution_q95"] = _count_distribution(obs, thresholds["q95"])
    observed_rates["count_distribution_q90"] = _count_distribution(obs, thresholds["q90"])

    g_total = simulated["gaussian"]["losses"].sum(axis=1)
    t_total = simulated["student_t"]["losses"].sum(axis=1)
    o_total = obs.sum(axis=1)
    # Edges span the bulk of all three samples; the extreme upper tail is folded
    # into the final bin by ``_histogram`` so the shape stays readable and the
    # counts still sum to the sample size.
    lo = float(min(g_total.min(), t_total.min(), o_total.min()))
    hi = float(
        max(
            np.quantile(g_total, 0.999),
            np.quantile(t_total, 0.999),
            np.quantile(o_total, 0.999),
        )
    )
    if hi <= lo:  # pragma: no cover - defensive
        hi = lo + 1.0
    edges = np.linspace(lo, hi, 61)

    models = {}
    for name, label, nu in (
        ("gaussian", "Gaussian copula", None),
        (
            "student_t",
            f"Student-t copula (ν = {cop_full['student_t_nu']:g})",
            cop_full["student_t_nu"],
        ),
    ):
        models[name] = _model_block(
            name,
            label,
            nu,
            simulated[name]["losses"],
            sites,
            marginals,
            cop_full["matrix"],
            total_threshold,
            thresholds,
            observed_rates,
            edges,
        )

    # ---- Tail-exceedance difference between the two hypotheses -------------
    tail_diff = {}
    g = models["gaussian"]["exceedance"]
    t = models["student_t"]["exceedance"]
    for key in ("at_least_two_q95", "all_q95", "at_least_two_q90", "all_q90"):
        gv, tv = float(g[key]), float(t[key])
        tail_diff[key] = {
            "gaussian": round(gv, 5),
            "student_t": round(tv, 5),
            "absolute": round(tv - gv, 5),
            "relative": None if gv == 0 else round(tv / gv, 3),
            "observed": round(float(observed_rates[key]), 5),
        }

    evaluation = _evaluation(
        sub[eval_idx],
        train_dists,
        sites,
        cop_train,
        min(scenario_count, 10000),
        seed + 1,
        [eval_months[0], eval_months[-1]],
    )

    # ---- Selection summary -------------------------------------------------
    selection = {
        "marginal": [
            {
                "code": c,
                "selected": marginals[c]["train"]["kind"],
                "candidates": marginals[c]["train"]["candidates"],
            }
            for c in sites
        ],
        "copula": {
            "selected": cop_train["preferred"],
            "selected_label": "Student-t copula"
            if cop_train["preferred"] == "student_t"
            else "Gaussian copula",
            "gaussian_aic": cop_train["gaussian_aic"],
            "student_t_aic": cop_train["student_t_aic"],
            "aic_gap": cop_train["aic_gap"],
            "student_t_nu": cop_train["student_t_nu"],
            "note": (
                "Selection uses the training period only. The evaluation period "
                "is scored afterwards and does not influence the choice."
            ),
        },
    }

    version = resolved_version(study, sites)
    signature_src = {
        "study": study["study"]["id"],
        "sites": sorted(sites),
        "version": version["label"],
        "marginals": {
            c: {
                "kind": marginals[c]["kind"],
                "params": {
                    k: round(float(v), 8) for k, v in marginals[c]["params"].items()
                },
            }
            for c in sites
        },
        "corr": np.round(cop_full["matrix"], 8).tolist(),
        "nu": cop_full["student_t_nu"],
        "n": scenario_count,
        "seed": seed,
        "train_fraction": train_fraction,
    }
    sig = st.signature(signature_src)

    payload = {
        "study_id": study["study"]["id"],
        "title": study["study"]["title"],
        "synthetic": True,
        "currency": study["study"]["currency"],
        "sites": sites,
        "site_meta": {x["code"]: x for x in study["sites"] if x["code"] in sites},
        "membership": version,
        "calendar": {
            "all_months": study["study"]["months"],
            "joint_months": int(n_common),
            "train_months": int(train_idx.size),
            "eval_months": int(eval_idx.size),
            "train_range": [train_months[0], train_months[-1]],
            "eval_range": [eval_months[0], eval_months[-1]],
            "common_range": [common_months[0], common_months[-1]],
            "per_site_observed": {
                c: int(np.sum(~np.isnan(grid[:, j]))) for c, j in zip(sites, sel)
            },
            "observed_series": [
                {
                    "month": month_ym(study, [m])[0],
                    "values": {
                        c: (None if np.isnan(grid[m, j]) else round(float(grid[m, j]), 3))
                        for c, j in zip(sites, sel)
                    },
                }
                for m in np.flatnonzero(~np.isnan(sub).any(axis=1))
            ],
            "site_series": {
                c: [
                    [month_ym(study, [m])[0], round(float(grid[m, j]), 3)]
                    for m in np.flatnonzero(~np.isnan(grid[:, j]))
                ]
                for c, j in zip(sites, sel)
            },
            "common_series": [
                {
                    "month": common_months[k],
                    "total_observed": round(float(np.nansum(obs[k])), 3),
                }
                for k in range(n_common)
            ],
        },
        "marginals": marginals,
        "copula": _public_copula(cop_full),
        "copula_train": _public_copula(cop_train),
        "pairs": pairs,
        "model_selection": selection,
        "truth": {
            "copula": truth.get("copula"),
            "df": truth.get("df"),
            "correlations": truth.get("correlations", []),
            "note": (
                "Ground truth of the synthetic generator. A correct fit should "
                "recover the correlation matrix and a small degrees-of-freedom "
                "parameter; the Gaussian copula cannot reproduce the tail "
                "dependence by construction."
            ),
        },
        "thresholds": {
            "site_q90": [round(x, 4) for x in thresholds["q90"]],
            "site_q95": [round(x, 4) for x in thresholds["q95"]],
            "total_q95": round(total_threshold, 4),
        },
        "models": models,
        "tail_difference": tail_diff,
        "observed": {
            "summary": observed_summary,
            "exceedance": {
                k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in observed_rates.items()
            },
            "histogram": _histogram(obs.sum(axis=1), edges),
            "n_months": int(n_common),
            "note": (
                "Observed jointly complete months, scored at the fitted marginal "
                "thresholds. Small counts mean the joint rate is imprecise."
            ),
        },
        "evaluation": evaluation,
        "scenario_count": scenario_count,
        "seed": seed,
        "train_fraction": train_fraction,
        "histogram_edges": [round(float(e), 3) for e in edges],
        "signature": sig,
        "signature_src": signature_src,
    }

    return FitResult(
        payload=payload,
        scenarios={name: m["losses"] for name, m in simulated.items()},
        uniforms={name: m["uniforms"] for name, m in simulated.items()},
        codes=sites,
        signature=sig,
    )


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def scenario_csv(result: FitResult, model: str, uniforms: bool = False) -> str:
    import csv
    import io

    losses = result.scenarios[model]
    U = result.uniforms[model]
    buf = io.StringIO()
    w = csv.writer(buf)
    header = ["scenario", *result.codes, "total"]
    if uniforms:
        header += [f"u_{c}" for c in result.codes]
    w.writerow(header)
    for i in range(losses.shape[0]):
        row = (
            [i + 1]
            + [round(float(x), 5) for x in losses[i]]
            + [round(float(losses[i].sum()), 5)]
        )
        if uniforms:
            row += [round(float(x), 6) for x in U[i]]
        w.writerow(row)
    return buf.getvalue()
