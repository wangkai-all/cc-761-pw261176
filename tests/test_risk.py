"""Statistical and API tests for the synthetic multi-site risk lab.

Run with ``.venv/bin/python -m pytest -q`` from the repository root.
"""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from server import risk
from server import stats as st
from server.data import DEFAULT_STUDY, build_study, loss_matrix
from server.main import app

CORE = ["harbor", "inland", "coast", "ridge"]
ALL5 = ["harbor", "inland", "coast", "ridge", "delta"]


@pytest.fixture(scope="session")
def study():
    return risk.load_study(DEFAULT_STUDY)


@pytest.fixture(scope="session")
def fit_core(study):
    return risk.fit(study, CORE, scenario_count=8000, seed=2611)


@pytest.fixture(scope="session")
def fit_all(study):
    return risk.fit(study, ALL5, scenario_count=8000, seed=2611)


@pytest.fixture()
def client():
    return TestClient(app)


def _common(study, sites):
    """Reconstruct common months and the fitted marginals' copula inputs."""
    grid, _, codes = loss_matrix(study)
    sel = [codes.index(c) for c in sites]
    sub = grid[:, sel]
    idx = np.flatnonzero(~np.isnan(sub).any(axis=1))
    return grid, codes, sub, idx


# ---------------------------------------------------------------------------
# Generator and calendar
# ---------------------------------------------------------------------------


def test_generator_is_deterministic_and_synthetic(study):
    again = build_study(DEFAULT_STUDY, seed=2611)
    assert [r["loss"] for r in study["records"][:50]] == [
        r["loss"] for r in again["records"][:50]
    ]
    assert study["study"]["synthetic"] is True
    assert all(r["loss"] > 0 for r in study["records"])


def test_calendars_are_ragged_and_missing_is_absent_not_zero(study):
    grid, _, codes = loss_matrix(study)
    observed = (~np.isnan(grid)).sum(axis=0)
    assert len(set(observed.tolist())) == len(codes), "sites must not share a calendar"
    delta = codes.index("delta")
    assert np.isnan(grid[:24, delta]).all(), "late site must have no early records"
    assert np.all(grid[~np.isnan(grid)] > 0)
    # A missing month is an absent record, not a zero row.
    assert sum(1 for r in study["records"]) == int(observed.sum())


def test_common_months_are_the_intersection(study, fit_core):
    grid, _, _, idx = _common(study, CORE)
    assert fit_core.payload["calendar"]["joint_months"] == idx.size
    assert idx.size < min((~np.isnan(grid[:, j])).sum() for j in range(len(CORE)))
    # First and last common month are reported verbatim.
    months = [m["ym"] for m in study["months"]]
    span = fit_core.payload["calendar"]["common_range"]
    assert span == [months[idx[0]], months[idx[-1]]]


def test_copula_inputs_come_from_jointly_observed_months(study, fit_core):
    """Recompute the uniform scores exactly as the pipeline documents.

    Membership: the copula is fed CDF values on common months only.  Rebuilding
    those scores must reproduce the reported correlation matrix, which proves
    the fit is not a gluing of separately ranked site series.
    """
    _, _, sub, idx = _common(study, CORE)
    dists = [st._dist(fit_core.payload["marginals"][c]["kind"],
                      fit_core.payload["marginals"][c]["params"]) for c in CORE]
    U = np.column_stack([st.uniform_scores(sub[idx, j], dists[j]) for j in range(len(CORE))])
    R, _ = st.nearest_correlation(st.kendall_matrix(U))
    reported = np.array(fit_core.payload["copula"]["matrix"])
    assert np.allclose(R, reported, atol=1e-6)
    assert U.min() > 0 and U.max() < 1


def test_ranking_each_site_separately_would_change_the_answer(study, fit_core):
    """The rejected approach: rank each site on its own calendar, then glue."""
    grid, _, sub, idx = _common(study, CORE)
    dists = [st._dist(fit_core.payload["marginals"][c]["kind"],
                      fit_core.payload["marginals"][c]["params"]) for c in CORE]
    proper = np.column_stack([st.uniform_scores(sub[idx, j], dists[j]) for j in range(len(CORE))])
    # Rank over each site's *own* observed record, then line the ranks up by
    # row order of the common months.  This is the "hard splice" the page warns
    # about: it invents a dependence estimate out of an alignment of ranks.
    spliced = np.column_stack([
        stats.rankdata(grid[~np.isnan(grid[:, j]), j])[: idx.size] / (idx.size + 1)
        for j in range(len(CORE))
    ])
    r_proper, _ = st.nearest_correlation(st.kendall_matrix(proper))
    r_spliced, _ = st.nearest_correlation(st.kendall_matrix(spliced))
    assert not np.allclose(r_proper, r_spliced, atol=1e-3)
    # And the reference matrix is the documented one.
    assert np.allclose(r_proper, np.array(fit_core.payload["copula"]["matrix"]), atol=1e-6)


def test_marginals_use_own_observed_months(study, fit_core):
    joint = fit_core.payload["calendar"]["joint_months"]
    months = [m["ym"] for m in study["months"]]
    train_end = months.index(fit_core.payload["calendar"]["train_range"][1])
    for code in CORE:
        m = fit_core.payload["marginals"][code]
        assert m["n_obs"] == len(m["obs_months"])
        assert len(m["obs_months"]) > joint, "marginal keeps its own extra months"
        assert m["n_train"] == len(m["train_months"])
        assert m["train_months"] == sorted(m["train_months"])
        # Training marginals stop at the last common training month, so the
        # evaluation window cannot leak into the fit or its thresholds.
        assert max(m["train_months"]) <= train_end


def test_site_added_late_shrinks_joint_sample(fit_core, fit_all):
    assert fit_all.payload["calendar"]["joint_months"] < fit_core.payload["calendar"]["joint_months"]
    assert fit_all.payload["calendar"]["eval_months"] < fit_core.payload["calendar"]["eval_months"]
    assert fit_all.payload["membership"]["label"] == "all-five"
    assert fit_core.payload["membership"]["label"] == "core-four"
    assert fit_all.payload["membership"]["effective_month"] == 24


# ---------------------------------------------------------------------------
# Correlation matrix legality
# ---------------------------------------------------------------------------


def test_fitted_correlation_matrix_is_legal(fit_core, fit_all):
    for result in (fit_core, fit_all):
        for key in ("copula", "copula_train"):
            R = np.array(result.payload[key]["matrix"])
            check = st.validate_correlation(R)
            assert check["legal"], (key, check)
            assert check["symmetric"] and check["unit_diagonal"] and check["psd"]
            assert np.max(np.abs(np.diag(R) - 1)) < 1e-9
            assert np.all(np.abs(R) <= 1 + 1e-9)
            assert result.payload[key]["legality"]["legal"] is True


def test_illegal_matrix_is_projected_and_flagged():
    bad = np.array([[1.0, 0.95, -0.95], [0.95, 1.0, 0.95], [-0.95, 0.95, 1.0]])
    assert not st.validate_correlation(bad)["psd"]
    fixed, diag = st.nearest_correlation(bad)
    assert st.validate_correlation(fixed)["legal"]
    assert diag["adjusted"] is True
    assert diag["max_abs_adjust"] > 0
    assert diag["min_eig_before"] < 0 <= diag["min_eig_after"]


def test_projected_matrix_is_a_no_op_when_already_legal():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    R = A @ A.T
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    fixed, diag = st.nearest_correlation(R)
    assert np.allclose(fixed, R, atol=1e-9)
    assert diag["adjusted"] is False


# ---------------------------------------------------------------------------
# Tail diagnostics and unstable estimates
# ---------------------------------------------------------------------------


def test_thin_joint_tail_is_exposed(fit_core):
    pairs = {tuple(sorted(p["pair"])): p for p in fit_core.payload["pairs"]}
    thin = pairs[("harbor", "inland")]
    assert thin["common_months"] == fit_core.payload["calendar"]["joint_months"]
    assert thin["n_dual_extreme"] < 10
    assert thin["stable"] is False
    assert "handful" in thin["message"]
    lo, hi = thin["thresholds"][0]["tail_ci"]
    assert hi - lo > 0.15, "the interval must visibly reflect the thin sample"


def test_strong_pair_is_not_flagged(fit_core):
    pairs = {tuple(sorted(p["pair"])): p for p in fit_core.payload["pairs"]}
    strong = pairs[("coast", "ridge")]
    assert strong["n_dual_extreme"] >= 10
    assert strong["stable"] is True
    assert "supported" in strong["message"]


def test_pair_diagnostic_reports_truth_correlation(fit_core):
    pairs = {tuple(sorted(p["pair"])): p for p in fit_core.payload["pairs"]}
    assert len(pairs) == len(CORE) * (len(CORE) - 1) // 2
    truth_map = {tuple(sorted(k)): v for k, v in DEFAULT_STUDY.correlations.items()}
    for key, row in pairs.items():
        assert row["truth_r"] == pytest.approx(truth_map[key], abs=1e-4)


# ---------------------------------------------------------------------------
# Model selection / evaluation separation
# ---------------------------------------------------------------------------


def test_selection_uses_training_fit_only(fit_core):
    sel = fit_core.payload["model_selection"]["copula"]
    train = fit_core.payload["copula_train"]
    assert sel["selected"] == train["preferred"]
    assert sel["gaussian_aic"] == train["gaussian_aic"]
    assert sel["student_t_aic"] == train["student_t_aic"]
    assert sel["selected"] == "student_t", "generator has genuine tail dependence"
    assert sel["student_t_nu"] == train["student_t_nu"]
    ev = fit_core.payload["evaluation"]
    q95 = [fit_core.payload["marginals"][c]["train_q95"] for c in CORE]
    assert np.allclose(ev["thresholds"]["q95"], q95, rtol=1e-3)


def test_evaluation_thresholds_are_training_marginal_quantiles(study):
    """The evaluation window must not leak into thresholds or selection."""
    for frac in (0.6, 0.7):
        res = risk.fit(study, CORE, scenario_count=2000, seed=2611, train_fraction=frac)
        ev = res.payload["evaluation"]
        for j, c in enumerate(CORE):
            m = res.payload["marginals"][c]
            assert ev["thresholds"]["q90"][j] == pytest.approx(m["train_q90"], rel=1e-3)
            assert ev["thresholds"]["q95"][j] == pytest.approx(m["train_q95"], rel=1e-3)
        ev_months = res.payload["calendar"]["eval_range"]
        assert res.payload["calendar"]["train_range"][1] < ev_months[0]


def test_evaluation_reports_intervals_and_model_rates(fit_core):
    ev = fit_core.payload["evaluation"]
    assert ev["n_eval"] > 30
    assert ev["n_eval"] == fit_core.payload["calendar"]["eval_months"]
    for name in ("gaussian", "student_t"):
        for key in ("at_least_two_q95", "all_q95", "at_least_two_q90", "all_q90"):
            blk = ev["joint"][name][key]
            assert blk["ci"][0] <= blk["rate"] <= blk["ci"][1]
            assert 0 <= blk["model_rate"] <= 1
            assert blk["model_expected"] == pytest.approx(
                blk["model_rate"] * ev["n_eval"], abs=0.05
            )
            assert blk["observed"] <= ev["n_eval"]
    for site in ev["sites"]:
        assert site["ci_q95"][0] <= site["rate_q95"] <= site["ci_q95"][1]
        assert site["ci_q90"][0] <= site["rate_q90"] <= site["ci_q90"][1]
        assert site["covers_q95"] == (site["ci_q95"][0] <= 0.05 <= site["ci_q95"][1])


def test_observed_tail_benchmark_uses_the_same_thresholds(fit_core):
    obs = fit_core.payload["observed"]["exceedance"]
    thr = fit_core.payload["thresholds"]["site_q95"]
    for code, t in zip(CORE, thr):
        assert t == pytest.approx(fit_core.payload["marginals"][code]["q95"], rel=1e-4)
    assert obs["n"] == fit_core.payload["calendar"]["joint_months"]
    # The observed benchmark is a rate per jointly observed month.
    assert sum(fit_core.payload["observed"]["exceedance"]["count_distribution_q95"]) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Simulation consistency: same marginals, different tail dependence
# ---------------------------------------------------------------------------


def test_both_models_share_marginals_and_correlation(fit_core):
    g = fit_core.payload["models"]["gaussian"]
    t = fit_core.payload["models"]["student_t"]
    assert g["label"] != t["label"]
    assert g["nu"] is None and t["nu"] is not None
    # Same marginals: the "fit" side of every check is identical by construction.
    mg = {r["code"]: (r["fit_mean"], r["fit_q95"]) for r in g["checks"]["marginals"]}
    mt = {r["code"]: (r["fit_mean"], r["fit_q95"]) for r in t["checks"]["marginals"]}
    assert mg == mt
    # Same rank correlation target.
    assert np.allclose(g["checks"]["correlation"]["target_tau"],
                       t["checks"]["correlation"]["target_tau"])
    # Simulated site means agree because the copula preserves the marginals,
    # even though the copulas differ in the tail.
    gm = [s["mean"] for s in g["summary"]["sites"]]
    tm = [s["mean"] for s in t["summary"]["sites"]]
    assert np.allclose(gm, tm, rtol=5e-3)


def test_simulation_checks_pass(fit_core):
    for name in ("gaussian", "student_t"):
        checks = fit_core.payload["models"][name]["checks"]
        assert checks["correlation"]["pass"], checks["correlation"]
        assert checks["aggregation"]["pass"], checks["aggregation"]
        assert all(r["pass"] for r in checks["marginals"]), checks["marginals"]
        assert checks["all_pass"] is True


def test_student_t_has_more_joint_tail_mass(fit_core):
    td = fit_core.payload["tail_difference"]
    assert td["all_q95"]["student_t"] > td["all_q95"]["gaussian"]
    assert td["at_least_two_q95"]["student_t"] >= td["at_least_two_q95"]["gaussian"]
    g = fit_core.payload["models"]["gaussian"]["summary"]["total"]
    t = fit_core.payload["models"]["student_t"]["summary"]["total"]
    assert g["mean"] == pytest.approx(t["mean"], rel=2e-3)
    assert t["q995"] > g["q995"]
    assert t["es99"] > g["es99"]
    # The difference is about the joint tail, not the central mass.
    assert t["q50"] == pytest.approx(g["q50"], rel=1e-2)


def test_common_random_numbers_pair_the_models(fit_core):
    g = fit_core.scenarios["gaussian"].sum(axis=1)
    t = fit_core.scenarios["student_t"].sum(axis=1)
    rho = stats.spearmanr(g, t).statistic
    assert rho > 0.9, "the two sets must share their underlying randomness"


def test_aggregation_identity_holds(fit_core):
    """E[sum_j X_j] == sum_j E[X_j] under any dependence."""
    for name in ("gaussian", "student_t"):
        agg = fit_core.payload["models"][name]["checks"]["aggregation"]
        assert agg["identity_reldiff"] < 1e-9
        assert agg["sim_total_mean"] == pytest.approx(agg["site_mean_sum"], rel=1e-9)
        # And it agrees with the fitted marginal means within Monte Carlo error.
        assert agg["fitted_reldiff"] <= agg["tolerance"]
        assert agg["pass"] is True


def test_small_scenario_counts_are_not_flagged_as_bias(study):
    """Tolerances must be Monte Carlo aware, not fixed thresholds."""
    res = risk.fit(study, ["harbor", "coast", "ridge"], scenario_count=1500, seed=5)
    for name in ("gaussian", "student_t"):
        checks = res.payload["models"][name]["checks"]
        agg = checks["aggregation"]
        assert agg["fitted_reldiff"] <= agg["tolerance"], agg
        assert agg["pass"] is True
        assert checks["correlation"]["max_abs_diff"] < checks["correlation"]["tolerance"]
        for row in checks["marginals"]:
            assert row["mean_reldiff"] <= row["mean_tolerance"], row
        assert checks["all_pass"] is True


def test_minimum_scenario_count_passes_every_check(study):
    res = risk.fit(study, CORE, scenario_count=1000, seed=2611)
    for name in ("gaussian", "student_t"):
        assert res.payload["models"][name]["checks"]["all_pass"] is True


def test_histogram_folds_mass_beyond_the_last_edge():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    counts = risk._histogram(np.array([-5.0, 0.5, 1.5, 2.5, 99.0, 1000.0]), edges)
    # Underflow joins the first bin, overflow the last; nothing is dropped.
    assert counts == [2, 1, 3]
    assert sum(counts) == 6


def test_site_contributions_sum_to_one(fit_core):
    summary = fit_core.payload["models"]["student_t"]["summary"]
    assert sum(s["share_mean"] for s in summary["sites"]) == pytest.approx(1.0, abs=1e-5)
    assert sum(s["tail_share"] for s in summary["sites"]) == pytest.approx(1.0, abs=1e-5)


def test_count_distributions_and_histograms_are_complete(fit_core):
    p = fit_core.payload
    for name in ("gaussian", "student_t"):
        for key in ("count_distribution_q90", "count_distribution_q95"):
            dist = p["models"][name][key]
            assert len(dist) == len(CORE) + 1
            assert sum(dist) == pytest.approx(1.0, abs=1e-4)
        hist = p["models"][name]["histogram"]
        assert len(hist) == len(p["histogram_edges"]) - 1
        assert sum(hist) == p["scenario_count"]
    surv = p["models"]["gaussian"]["summary"]["survival"]
    assert len(surv) > 10
    assert all(surv[i][1] >= surv[i + 1][1] for i in range(len(surv) - 1))


def test_truth_is_recovered(fit_all):
    truth = {tuple(sorted(row["pair"])): row["corr"] for row in fit_all.payload["truth"]["correlations"]}
    R = np.array(fit_all.payload["copula"]["matrix"])
    for i, a in enumerate(ALL5):
        for j in range(i + 1, len(ALL5)):
            b = ALL5[j]
            assert R[i, j] == pytest.approx(truth[tuple(sorted((a, b)))], abs=0.08)
    assert fit_all.payload["copula"]["student_t_nu"] == pytest.approx(3.5, abs=0.8)


# ---------------------------------------------------------------------------
# Membership changes and artifact signatures
# ---------------------------------------------------------------------------


def test_signature_changes_with_membership(fit_core, fit_all):
    assert fit_core.signature != fit_all.signature
    assert fit_core.payload["signature_src"]["version"] == "core-four"
    assert fit_all.payload["signature_src"]["version"] == "all-five"
    assert set(fit_all.payload["signature_src"]["sites"]) == set(ALL5)


def test_scenarios_are_not_a_prefix_of_each_other(fit_core, fit_all):
    assert fit_core.scenarios["gaussian"].shape[1] == len(CORE)
    assert fit_all.scenarios["gaussian"].shape[1] == len(ALL5)
    # Adding a site changes the common sample, so the 4x4 block is refitted.
    R4 = np.array(fit_core.payload["copula"]["matrix"])
    R5 = np.array(fit_all.payload["copula"]["matrix"])[: len(CORE), : len(CORE)]
    assert not np.allclose(R4, R5, atol=1e-3)


def test_same_membership_is_reproducible(study, fit_core):
    again = risk.fit(study, CORE, scenario_count=8000, seed=2611)
    assert again.signature == fit_core.signature
    assert np.allclose(again.scenarios["gaussian"], fit_core.scenarios["gaussian"])
    changed = risk.fit(study, CORE, scenario_count=8000, seed=7)
    assert changed.signature != fit_core.signature


def test_fit_rejects_too_few_sites_and_unknown_codes(study):
    with pytest.raises(risk.FitError):
        risk.fit(study, ["harbor"], scenario_count=1000)
    with pytest.raises(risk.FitError):
        risk.fit(study, ["harbor", "atlantis"], scenario_count=1000)


def test_api_rejects_column_wise_extension(client):
    core = client.post("/api/fit", json={"sites": CORE, "scenarios": 1200}).json()
    resp = client.post(
        "/api/fit",
        json={"sites": ALL5, "scenarios": 1200, "parent_signature": core["signature"]},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "membership_changed"
    assert detail["previous_version"] == "core-four"
    assert detail["next_version"] == "all-five"
    assert detail["previous_signature"] == core["signature"]
    # Same membership with a parent signature is fine (a deliberate refit).
    ok = client.post(
        "/api/fit",
        json={"sites": CORE, "scenarios": 1200, "parent_signature": core["signature"]},
    )
    assert ok.status_code == 200


def test_api_unknown_signature_is_404(client):
    assert client.get("/api/scatter?signature=deadbeef&model=gaussian&a=harbor&b=coast").status_code == 404
    assert client.get("/api/scenarios.csv?signature=deadbeef&model=gaussian").status_code == 404


def test_scatter_rejects_pairs_outside_the_set(client):
    res = client.post("/api/fit", json={"sites": CORE, "scenarios": 1200}).json()
    sig = res["signature"]
    assert client.get(f"/api/scatter?signature={sig}&model=gaussian&a=harbor&b=delta").status_code == 422
    assert client.get(f"/api/scatter?signature={sig}&model=gaussian&a=harbor&b=harbor").status_code == 422
    good = client.get(f"/api/scatter?signature={sig}&model=student_t&a=harbor&b=coast&n=200")
    assert good.status_code == 200
    body = good.json()
    assert len(body["points"]) == 200
    assert body["joint_q95"] >= 0


# ---------------------------------------------------------------------------
# Downloads and API surface
# ---------------------------------------------------------------------------


def test_downloads_have_matching_rows_and_totals(client):
    res = client.post("/api/fit", json={"sites": CORE, "scenarios": 1500}).json()
    sig = res["signature"]
    csv_resp = client.get(f"/api/scenarios.csv?signature={sig}&model=student_t")
    assert csv_resp.status_code == 200
    rows = list(csv.reader(io.StringIO(csv_resp.text)))
    header, body = rows[0], rows[1:]
    assert header == ["scenario", *CORE, "total"]
    assert len(body) == 1500
    for row in body[:25]:
        vals = [float(x) for x in row[1 : 1 + len(CORE)]]
        assert float(row[-1]) == pytest.approx(sum(vals), abs=1e-3)

    params = client.get(f"/api/params.json?signature={sig}")
    payload = json.loads(params.text)
    assert payload["signature"] == sig
    assert payload["sites"] == CORE
    assert np.array(payload["copula"]["matrix"]).shape == (4, 4)


def test_uniform_download_adds_copula_columns(client):
    res = client.post("/api/fit", json={"sites": CORE, "scenarios": 1200}).json()
    sig = res["signature"]
    text = client.get(
        f"/api/scenarios.csv?signature={sig}&model=gaussian&uniforms=true"
    ).text
    header = next(csv.reader(io.StringIO(text)))
    assert header[-len(CORE):] == [f"u_{c}" for c in CORE]


def test_params_download_contains_legality_and_calendar(client):
    res = client.post("/api/fit", json={"sites": CORE, "scenarios": 1200}).json()
    payload = client.get(f"/api/params.json?signature={res['signature']}").json()
    assert payload["copula"]["legality"]["legal"] is True
    assert payload["calendar"]["joint_months"] > 0
    assert payload["model_selection"]["copula"]["selected"] in ("gaussian", "student_t")
    assert "nu" in payload["signature_src"]
    assert payload["scenario_set"]["signature"] == res["signature"]
    # A JSON round-trip must not choke on numpy values added later.
    json.dumps(payload)


def test_study_endpoint_returns_raw_records(client):
    body = client.get("/api/study").json()
    assert body["study"]["synthetic"] is True
    assert len(body["sites"]) == 5
    assert body["study"]["membership_versions"][0]["label"] == "core-four"
    assert body["records"][0]["site_code"] in {s["code"] for s in body["sites"]}


def test_records_download_matches_study(client):
    body = client.get("/api/study").json()
    rows = list(csv.reader(io.StringIO(client.get("/api/records.csv").text)))
    assert rows[0] == ["site_code", "month", "month_index", "loss_synthetic"]
    assert len(rows) - 1 == len(body["records"])


def test_health(client):
    body = client.get("/api/health").json()
    assert body == {"ok": True, "synthetic": True}


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def test_t_copula_loglik_prefers_t_for_heavy_tail_data():
    rng = np.random.default_rng(3)
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    vals, vecs = np.linalg.eigh(R)
    f = vecs * np.sqrt(vals)
    Z = rng.standard_normal((4000, 2)) @ f.T
    nu_true = 4.0
    T = Z / np.sqrt(rng.chisquare(nu_true, 4000) / nu_true)[:, None]
    U = stats.t.cdf(T, nu_true)
    Rk = st.kendall_matrix(U)
    assert st.t_loglik(Rk, U, nu_true) > st.gaussian_loglik(Rk, stats.norm.ppf(U))


def test_uniform_scores_are_marginal_cdf_values():
    dist = st._dist("lognormal", {"mu": 1.0, "sigma": 0.5})
    x = np.array([0.5, 1.0, 3.0, 10.0, 200.0])
    u = st.uniform_scores(x, dist)
    assert np.allclose(u, dist.cdf(x))
    assert u.min() > 0 and u.max() < 1


def test_simulated_uniforms_are_uniform(fit_core):
    for name in ("gaussian", "student_t"):
        U = fit_core.uniforms[name]
        assert U.shape == (8000, len(CORE))
        for j in range(U.shape[1]):
            assert stats.kstest(U[:, j], "uniform").pvalue > 1e-3


def test_copula_uniforms_have_the_right_tail_ordering(fit_core):
    """The t copula must produce more simultaneous uniforms near 1."""
    g = fit_core.uniforms["gaussian"]
    t = fit_core.uniforms["student_t"]
    assert np.mean((t > 0.95).sum(axis=1) >= 2) > np.mean((g > 0.95).sum(axis=1) >= 2)


def test_template_renders_the_page(client):
    html = client.get("/").text
    assert "多地点损失依赖" in html
    assert "app.js" in html
