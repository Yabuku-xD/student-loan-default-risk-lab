"""
Lightweight FastAPI service for serving risk scores and watchlists.

Run with:
    uvicorn src.api.server:app --reload
"""

# I'm keeping this service tiny on purpose—future me can grow it when needed.
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
# TODO: let callers override METRICS_PATH once we deploy multiple environments.


app = FastAPI(title="Student Loan Default Risk API", version="1.0.0")


def _load_metrics() -> Dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}")
    # Loading eagerly because I like catching missing artifacts immediately.
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_metrics() -> Dict:
    return _load_metrics()


@lru_cache(maxsize=32)
def get_scores(model_name: str) -> pd.DataFrame:
    metrics = get_metrics()
    watchlists = metrics.get("watchlists", {})
    if model_name not in watchlists:
        raise KeyError(f"Model '{model_name}' not present in watchlists.")
    scores_rel = watchlists[model_name]["scores"]
    scores_path = BASE_DIR / scores_rel
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")
    # Note to self: parquet keeps things lean; please resist the CSV urge.
    return pd.read_parquet(scores_path)


@app.get("/health")
def health() -> Dict[str, str]:
    # Keeping the health endpoint boring so uptime checks stay cheap.
    return {"status": "ok"}


@app.get("/models")
def list_models() -> List[Dict[str, str]]:
    metrics = get_metrics()
    results = []
    for name, stats in metrics.get("overall", {}).items():
        thresholds = metrics.get("thresholds", {}).get(name, {})
        results.append(
            {
                "model": name,
                "accuracy": stats.get("accuracy"),
                "precision": stats.get("precision"),
                "recall": stats.get("recall"),
                "roc_auc": stats.get("roc_auc"),
                "recommended_threshold": thresholds.get("recommended_threshold"),
            }
        )
    return results


@app.get("/watchlist/{model_name}")
def watchlist(model_name: str, limit: int = 100) -> List[Dict[str, str]]:
    try:
        df = get_scores(model_name)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    df_sorted = df.sort_values("risk_score", ascending=False).head(limit)
    # TODO: add pagination once the UI team actually needs more than top N.
    return df_sorted.to_dict(orient="records")


@app.get("/institution/{unitid}")
def institution(unitid: int) -> Dict[str, Dict]:
    metrics = get_metrics()
    response: Dict[str, Dict] = {}

    for model_name in metrics.get("watchlists", {}).keys():
        try:
            df = get_scores(model_name)
        except FileNotFoundError:
            continue
        match = df[df["UNITID"] == unitid]
        if match.empty:
            continue
        response[model_name] = match.iloc[0].to_dict()

    if not response:
        raise HTTPException(status_code=404, detail=f"UNITID {unitid} not found in scores.")
    return response

