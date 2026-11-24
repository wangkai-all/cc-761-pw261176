"""Synthetic loss-record generator (fictional data only).

The generator builds a monthly panel of loss records for a handful of fictional
sites.  Each site's loss follows a log-normal severity distribution; the
dependence between sites is a **Student-t copula** with a known correlation
matrix and a known small degrees-of-freedom parameter, so the ground truth has
genuine joint upper-tail dependence.

This choice is deliberate: it means the page's two dependence hypotheses
(Gaussian vs Student-t copula) are not equally true.  The Gaussian copula
preserves the same rank correlation but has no tail dependence, so it
understates how often several sites blow through their limits in the same
month.  The Student-t copula should be selected on the training period, and the
difference in joint-exceedance rates is a real difference in the generating
process rather than a numerical artefact.

Statistical design points that mirror the requirements:

*   Sites do **not** share a calendar.  Missing months are genuinely absent
    from the record rather than zero, and each site's marginal is fitted on its
    own observed months only.  Any dependence estimate then uses *jointly
    observed* months and says so explicitly.
*   One site joins late, so the joint calendar is ragged and the membership
    version changes mid-record.
*   One pair is deliberately weakly correlated, so even with a t copula its
    joint upper tail has very few observations.  Pairwise tail estimates for it
    must be flagged as unstable instead of being reported as if they were
    reliable.
*   A scenario set belongs to exactly one site set.  Adding a site changes the
    joint sample and the fit, so old scenarios cannot simply be extended with
    another column.

All amounts are synthetic and denominated in thousands of a fictional currency
unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from .rng import rng_for


@dataclass(frozen=True)
class SiteSpec:
    code: str
    name: str
    region: str
    # Marginal log-normal severity: loss = exp(mu + sigma * z).
    mu: float
    sigma: float
    # Calendar: first / last month index (inclusive), plus per-site missingness
    # probability for interior months.
    start_month: int
    end_month: int
    missing_prob: float


@dataclass(frozen=True)
class StudySpec:
    study_id: str
    title: str
    start_ym: str  # "YYYY-MM"
    months: int
    sites: tuple[SiteSpec, ...]
    # Truth: Student-t copula with this df and correlation matrix.
    copula_df: float
    correlations: dict[tuple[str, str], float]
    # Membership versions: (label, site codes, effective month).
    membership_versions: tuple[tuple[str, tuple[str, ...], int], ...] = field(
        default_factory=tuple
    )

    def matrix(self) -> np.ndarray:
        codes = [s.code for s in self.sites]
        p = len(codes)
        R = np.eye(p)
        for (a, b), r in self.correlations.items():
            i, j = codes.index(a), codes.index(b)
            R[i, j] = R[j, i] = r
        return R


def _ym_add(start_ym: str, k: int) -> str:
    y, m = (int(x) for x in start_ym.split("-"))
    idx = y * 12 + (m - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


DEFAULT_STUDY = StudySpec(
    study_id="fictional-basin-1985-2024",
    title="Synthetic multi-site loss panel",
    start_ym="1985-01",
    months=480,
    sites=(
        SiteSpec(
            code="harbor",
            name="Harbor Terminal",
            region="North",
            mu=4.35,
            sigma=0.34,
            start_month=0,
            end_month=479,
            missing_prob=0.02,
        ),
        SiteSpec(
            code="inland",
            name="Inland Rail Yard",
            region="Central",
            mu=4.05,
            sigma=0.38,
            start_month=0,
            end_month=479,
            missing_prob=0.03,
        ),
        SiteSpec(
            code="coast",
            name="Coastal Plant",
            region="South",
            mu=4.5,
            sigma=0.32,
            start_month=0,
            end_month=479,
            missing_prob=0.015,
        ),
        SiteSpec(
            code="ridge",
            name="Ridge Storage",
            region="East",
            mu=3.9,
            sigma=0.42,
            start_month=0,
            end_month=479,
            missing_prob=0.04,
        ),
        SiteSpec(
            code="delta",
            name="Delta Fabrication",
            region="West",
            mu=4.15,
            sigma=0.36,
            # Joins the study 24 months in: the joint calendar is ragged, and
            # scenarios built for the four-site set cannot simply take an extra
            # column.
            start_month=24,
            end_month=479,
            missing_prob=0.02,
        ),
    ),
    copula_df=3.5,
    # Strong co-movement among coast / ridge / harbor, but harbor and inland are
    # only weakly linked.  Same copula family, different correlation strength:
    # the weakly linked pair therefore has a very thin joint upper tail.
    correlations={
        ("harbor", "inland"): 0.08,
        ("harbor", "coast"): 0.50,
        ("harbor", "ridge"): 0.44,
        ("harbor", "delta"): 0.40,
        ("inland", "coast"): 0.30,
        ("inland", "ridge"): 0.26,
        ("inland", "delta"): 0.24,
        ("coast", "ridge"): 0.55,
        ("coast", "delta"): 0.48,
        ("ridge", "delta"): 0.38,
    },
    membership_versions=(
        ("core-four", ("harbor", "inland", "coast", "ridge"), 0),
        ("all-five", ("harbor", "inland", "coast", "ridge", "delta"), 24),
    ),
)


def build_study(spec: StudySpec = DEFAULT_STUDY, seed: int = 2611) -> dict[str, Any]:
    """Generate the full synthetic panel and its calendar.

    Returns a JSON-serialisable dict: ``months``, ``sites`` and ``records``
    (one row per observed site-month with a synthetic loss amount).
    """
    n = spec.months
    codes = [s.code for s in spec.sites]
    R = spec.matrix()
    g = rng_for(spec.study_id, "copula", seed=seed)

    # Multivariate t draws with correlation R.
    vals, vecs = np.linalg.eigh(R)
    factor = vecs * np.sqrt(np.clip(vals, 0, None))
    Z = g.standard_normal((n, len(codes))) @ factor.T
    W = g.chisquare(spec.copula_df, size=n)
    T = Z / np.sqrt(W / spec.copula_df)[:, None]
    U = stats.t.cdf(T, spec.copula_df)

    rows: list[dict[str, Any]] = []
    site_meta: list[dict[str, Any]] = []
    for j, site in enumerate(spec.sites):
        sg = rng_for(spec.study_id, site.code, "calendar", seed=seed)
        # Marginally log-normal, driven by the t-copula uniform.
        loss = stats.lognorm.ppf(
            np.clip(U[:, j], 1e-12, 1 - 1e-12), s=site.sigma, scale=np.exp(site.mu)
        )
        observed = sg.random(n) >= site.missing_prob
        observed[: site.start_month] = False
        observed[site.end_month + 1 :] = False

        for m in np.flatnonzero(observed):
            rows.append(
                {
                    "site_code": site.code,
                    "month_index": int(m),
                    "month": _ym_add(spec.start_ym, int(m)),
                    "loss": round(float(loss[m]), 3),
                }
            )
        site_meta.append(
            {
                "code": site.code,
                "name": site.name,
                "region": site.region,
                "mu": site.mu,
                "sigma": site.sigma,
                "start_month": site.start_month,
                "end_month": site.end_month,
                "observed_months": int(observed.sum()),
            }
        )

    months = [
        {"index": m, "ym": _ym_add(spec.start_ym, m)} for m in range(spec.months)
    ]
    return {
        "study": {
            "id": spec.study_id,
            "title": spec.title,
            "start_ym": spec.start_ym,
            "months": spec.months,
            "synthetic": True,
            "currency": "thousand units (synthetic)",
            "truth": {
                "copula": "student_t",
                "df": spec.copula_df,
                "correlations": [
                    {"pair": list(k), "corr": v}
                    for k, v in sorted(spec.correlations.items())
                ],
            },
            "membership_versions": [
                {"label": label, "sites": list(codes), "effective_month": eff}
                for label, codes, eff in spec.membership_versions
            ],
        },
        "months": months,
        "sites": site_meta,
        "records": rows,
    }


def loss_matrix(study: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pivot records into a month x site matrix, NaN where unobserved."""
    codes = [s["code"] for s in study["sites"]]
    n_months = study["study"]["months"]
    grid = np.full((n_months, len(codes)), np.nan)
    col = {c: j for j, c in enumerate(codes)}
    for r in study["records"]:
        grid[r["month_index"], col[r["site_code"]]] = r["loss"]
    return grid, np.arange(n_months), codes


def month_ym(study: dict[str, Any], index: np.ndarray) -> list[str]:
    by_idx = {m["index"]: m["ym"] for m in study["months"]}
    return [by_idx[int(i)] for i in index]
