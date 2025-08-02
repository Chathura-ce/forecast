# forecast-api/main.py

import logging
from datetime import date, timedelta
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from model_wrapper import ForecastWrapper, make_features

# ── App & Logger ────────────────────────────────────────────────
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# ── Config ──────────────────────────────────────────────────────
MODELS_DIR = "models"  # beans.pkl, nadu.pkl, etc live here


# ── Schemas ──────────────────────────────────────────────────────
class HistoryPoint(BaseModel):
    ds: date
    y: float

class PredictRequest(BaseModel):
    item: str
    periods: int = 4
    history: Optional[List[HistoryPoint]] = None


# ── /predict Endpoint ─────────────────────────────────────────────
@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    # 1) Parse & validate history
    if req.history is None:
        raise HTTPException(400, "Missing `history` in request body.")
    try:
        recs = [hp.dict() for hp in req.history]
        hist_df = pd.DataFrame.from_records(recs)
        if not {"ds","y"}.issubset(hist_df.columns):
            raise ValueError(f"Needs 'ds' + 'y'; got {hist_df.columns.tolist()}")
        hist_df["ds"] = pd.to_datetime(hist_df["ds"])
        hist_df["y"]  = hist_df["y"].astype(float)
    except Exception as e:
        logger.exception("Error parsing history")
        raise HTTPException(400, f"Invalid history payload: {e}")

    # 2) Load wrapper
    try:
        wrapper: ForecastWrapper = joblib.load(f"{MODELS_DIR}/{req.item}.pkl")
    except FileNotFoundError:
        raise HTTPException(404, f"No model for item '{req.item}'")

    # 3) Forecast
    ser = wrapper.predict(hist_df, req.periods)

    # 4) Format & return
    out = []
    for ds, v in ser.items():
        # JSON doesn’t support NaN or inf—map those to null
        if not np.isfinite(v):
            yhat = None
        else:
            yhat = round(float(v), 2)
        out.append({
            "ds": ds.strftime("%Y-%m-%d"),
            "yhat": yhat
        })
    return {"item": req.item, "predictions": out}
