from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lca_lcc_evaluator import BiocharLcaLccEvaluator  # noqa: E402


class EvaluationRequest(BaseModel):
    candidate: dict[str, Any] = Field(default_factory=dict)
    prediction_percent: float
    options: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="PFAS Biochar OpenLCA Evaluator",
    version="1.0.0",
    description="HTTP evaluator for PFAS biochar LCA/LCC calculations used by the AISCIA PFAS platform.",
)
evaluator = BiocharLcaLccEvaluator()


def _configured_mode() -> str:
    mode = os.environ.get("OPENLCA_EVALUATOR_MODE", "").strip().lower()
    if mode:
        return mode
    if os.environ.get("OPENLCA_REST_URL", "").strip():
        return "openlca_rest"
    if os.environ.get("OPENLCA_IPC_PORT", "").strip():
        return "openlca_ipc"
    return "proxy"


def _require_auth(authorization: str | None) -> None:
    expected = os.environ.get("OPENLCA_EVALUATOR_API_KEY", "").strip()
    if not expected:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid evaluator API key.")


def _effective_options(options: dict[str, Any]) -> dict[str, Any]:
    out = dict(options or {})
    mode = str(out.get("environmental_mode") or "").strip().lower()
    if mode in {"openlca_cloud", "cloud", "remote", ""}:
        out["environmental_mode"] = _configured_mode()
    out.setdefault("fallback_to_proxy", os.environ.get("OPENLCA_FALLBACK_TO_PROXY", "true").lower() != "false")
    if os.environ.get("OPENLCA_REST_URL"):
        out["openlca_rest_url"] = os.environ["OPENLCA_REST_URL"].strip()
    if os.environ.get("OPENLCA_IPC_PORT"):
        out["ipc_port"] = int(os.environ["OPENLCA_IPC_PORT"])
    if os.environ.get("OPENLCA_PRODUCT_SYSTEM"):
        out["product_system"] = os.environ["OPENLCA_PRODUCT_SYSTEM"].strip()
    if os.environ.get("OPENLCA_IMPACT_METHOD"):
        out["impact_method"] = os.environ["OPENLCA_IMPACT_METHOD"].strip()
    return out


@app.get("/health")
def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    mode = _configured_mode()
    status: dict[str, Any]
    if mode == "openlca_rest":
        status = evaluator.openlca_rest_status({"openlca_rest_url": os.environ.get("OPENLCA_REST_URL", "")})
    elif mode == "openlca_ipc":
        status = {"available": True, "mode": "openlca_ipc", "message": "Evaluator configured for local IPC.", "port": int(os.environ.get("OPENLCA_IPC_PORT", "8080"))}
    else:
        status = {"available": True, "mode": "proxy", "message": "Proxy LCA/LCC evaluator is available."}
    status["service"] = "pfas-openlca-evaluator"
    status["configured_mode"] = mode
    status["product_system"] = os.environ.get("OPENLCA_PRODUCT_SYSTEM", "")
    status["impact_method"] = os.environ.get("OPENLCA_IMPACT_METHOD", "")
    return status


@app.get("/metadata")
def metadata(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    return evaluator.metadata()


@app.post("/evaluate")
def evaluate(payload: EvaluationRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    options = _effective_options(payload.options)
    try:
        result = evaluator.evaluate(payload.candidate, payload.prediction_percent, options)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    result["evaluator_service_mode"] = options.get("environmental_mode")
    return {"ok": True, "evaluation": result}
