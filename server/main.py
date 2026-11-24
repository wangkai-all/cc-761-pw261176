"""FastAPI application for the synthetic multi-site risk page."""

from __future__ import annotations

import csv
import io
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import risk
from .data import DEFAULT_STUDY

app = FastAPI(title="Synthetic multi-site risk lab", version="1.0.0")

MAX_SCENARIOS = 50000
MIN_SCENARIOS = 1000
CACHE_LIMIT = 6

_cache: "OrderedDict[str, risk.FitResult]" = OrderedDict()
_cache_lock = threading.Lock()
_study_cache: dict[str, Any] = {}


def study() -> dict[str, Any]:
    if "v" not in _study_cache:
        _study_cache["v"] = risk.load_study(DEFAULT_STUDY)
    return _study_cache["v"]


def _remember(result: risk.FitResult) -> None:
    with _cache_lock:
        _cache[result.signature] = result
        _cache.move_to_end(result.signature)
        while len(_cache) > CACHE_LIMIT:
            _cache.popitem(last=False)


def _result(signature: str) -> risk.FitResult:
    with _cache_lock:
        r = _cache.get(signature)
        if r is not None:
            _cache.move_to_end(signature)
            return r
    raise HTTPException(
        status_code=404,
        detail={
            "code": "unknown_signature",
            "message": (
                "this scenario set is no longer in memory; refit to regenerate "
                "it. Scenario sets are never stitched together from an older fit."
            ),
        },
    )


class FitRequest(BaseModel):
    sites: list[str] = Field(min_length=2)
    scenarios: int = Field(default=20000, ge=MIN_SCENARIOS, le=MAX_SCENARIOS)
    seed: int = Field(default=2611, ge=0, le=2**31 - 1)
    train_fraction: float = Field(default=0.7, ge=0.4, le=0.9)
    parent_signature: str | None = None


@app.get("/api/study")
def get_study() -> dict[str, Any]:
    s = study()
    return {
        "study": s["study"],
        "sites": s["sites"],
        "months": s["months"],
        "records": s["records"],
        "limits": {"min_scenarios": MIN_SCENARIOS, "max_scenarios": MAX_SCENARIOS},
    }


@app.post("/api/fit")
def post_fit(req: FitRequest) -> Any:
    s = study()
    sites = list(dict.fromkeys(req.sites))

    # Membership guard: a scenario set belongs to exactly one site set.  If the
    # caller tries to build on an existing signature with a different site set,
    # reject the "append a column" move outright and require a full refit.
    if req.parent_signature:
        prev = _result(req.parent_signature)
        if sorted(prev.codes) != sorted(sites):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "membership_changed",
                    "message": (
                        "The selected sites differ from the parent scenario set. "
                        "A new site changes the jointly observed sample, the "
                        "marginal refits and the dependence structure, so the "
                        "scenarios must be regenerated rather than extended "
                        "column by column."
                    ),
                    "previous_sites": prev.codes,
                    "next_sites": sorted(sites),
                    "previous_version": prev.payload["membership"]["label"],
                    "next_version": risk.resolved_version(s, sites)["label"],
                    "previous_signature": prev.signature,
                },
            )

    try:
        result = risk.fit(
            s,
            sites,
            scenario_count=req.scenarios,
            seed=req.seed,
            train_fraction=req.train_fraction,
        )
    except risk.FitError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "fit_failed", "message": str(exc)}
        )
    _remember(result)

    payload = dict(result.payload)
    payload["versions"] = {
        v["label"]: v["sites"] for v in s["study"]["membership_versions"]
    }
    payload["scenario_set"] = {
        "signature": result.signature,
        "sites": result.codes,
        "rows": int(result.scenarios["gaussian"].shape[0]),
        "models": list(result.scenarios.keys()),
    }
    return payload


@app.get("/api/scatter")
def get_scatter(
    signature: str,
    a: str,
    b: str,
    model: str = Query(default="student_t", pattern="^(gaussian|student_t)$"),
    n: int = Query(default=700, ge=50, le=3000),
):
    """Sampled scenario pairs for two sites.

    Points are sampled evenly in the order of the pair's total loss, so the
    displayed cloud keeps its upper tail instead of re-sampling only the bulk.
    """
    import numpy as np

    r = _result(signature)
    if a not in r.codes or b not in r.codes:
        raise HTTPException(status_code=422, detail="pair not in this scenario set")
    if a == b:
        raise HTTPException(status_code=422, detail="pick two different sites")
    losses = r.scenarios[model]
    ia, ib = r.codes.index(a), r.codes.index(b)
    total = losses[:, ia] + losses[:, ib]
    order = np.argsort(total)
    pick = np.linspace(0, order.size - 1, min(n, order.size)).round().astype(int)
    idx = order[pick]
    p = r.payload
    q95 = [p["marginals"][a]["q95"], p["marginals"][b]["q95"]]
    joint = float(np.mean((losses[:, ia] > q95[0]) & (losses[:, ib] > q95[1])))
    return {
        "a": a,
        "b": b,
        "model": model,
        "points": [
            [round(float(losses[i, ia]), 3), round(float(losses[i, ib]), 3)]
            for i in idx
        ],
        "q95": [round(float(q95[0]), 3), round(float(q95[1]), 3)],
        "joint_q95": round(joint, 5),
    }


@app.get("/api/scenarios.csv")
def download_scenarios(
    signature: str,
    model: str = Query(pattern="^(gaussian|student_t)$"),
    uniforms: bool = False,
):
    r = _result(signature)
    body = risk.scenario_csv(r, model, uniforms=uniforms)
    fn = f"scenarios_{model}_{signature}.csv"
    return PlainTextResponse(
        body,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fn}"'},
    )


@app.get("/api/params.json")
def download_params(signature: str):
    r = _result(signature)
    payload = dict(r.payload)
    # Keep the artifact self-describing: it must say which scenario set it came
    # from and which membership versions exist, so a consumer cannot mistake it
    # for a set that was extended column by column.
    payload["scenario_set"] = {
        "signature": r.signature,
        "sites": r.codes,
        "rows": int(r.scenarios["gaussian"].shape[0]),
        "models": list(r.scenarios.keys()),
    }
    payload["versions"] = {
        v["label"]: v["sites"] for v in study()["study"]["membership_versions"]
    }
    body = json.dumps(payload, indent=2, default=str)
    fn = f"params_{signature}.json"
    return PlainTextResponse(
        body,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fn}"'},
    )


@app.get("/api/records.csv")
def download_records():
    s = study()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["site_code", "month", "month_index", "loss_synthetic"])
    for r in s["records"]:
        w.writerow([r["site_code"], r["month"], r["month_index"], r["loss"]])
    return PlainTextResponse(
        buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="synthetic_records.csv"'},
    )


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "synthetic": True})


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run("server.main:app", host="127.0.0.1", port=8123)
