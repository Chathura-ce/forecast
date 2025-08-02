# main.py

import logging
from datetime import date, timedelta
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────
MODELS_DIR = "models"
W_PROPHET  = 0.6
W_LGBM     = 0.4

# ── App & logger ──────────────────────────────────────────
app = FastAPI()
logger = logging.getLogger("uvicorn.error")


# ── Pydantic schemas ──────────────────────────────────────
class HistoryPoint(BaseModel):
    ds: date
    y:  float

class PredictRequest(BaseModel):
    item:    str
    periods: int = 4
    history: Optional[List[HistoryPoint]] = None


# ── Helpers ───────────────────────────────────────────────
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lag1"]  = out["y"].shift(1)
    out["lag2"]  = out["y"].shift(2)
    out["lag4"]  = out["y"].shift(4)
    out["month"] = out["ds"].dt.month
    wk = out["ds"].dt.isocalendar().week.astype(int)
    out["sin52"] = np.sin(2 * np.pi * wk / 52)
    out["cos52"] = np.cos(2 * np.pi * wk / 52)
    return out.dropna()


# ── Endpoint ─────────────────────────────────────────────
@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    # 1) Parse & validate the history payload
    if req.history is None:
        raise HTTPException(400, "Missing `history` in request body.")
    try:
        # Convert list of HistoryPoint into a DataFrame
        records = [hp.dict() for hp in req.history]
        hist = pd.DataFrame.from_records(records)
        # Ensure required columns exist
        if not {"ds", "y"}.issubset(hist.columns):
            raise ValueError(f"History must contain 'ds' and 'y' keys; got {list(hist.columns)}")
        # Cast types
        hist["ds"] = pd.to_datetime(hist["ds"])
        hist["y"]  = hist["y"].astype(float)
    except Exception as e:
        # Log full stack trace for debugging
        logger.exception("Error parsing history payload")
        raise HTTPException(400, f"Invalid history payload: {e}")

    # 2) Load the Prophet model
    try:
        m = joblib.load(f"{MODELS_DIR}/{req.item}_prophet.pkl")
    except FileNotFoundError:
        raise HTTPException(404, f"No trained model found for item '{req.item}'")

    # 3) Load LightGBM models (optional)
    try:
        lgb_models = joblib.load(f"{MODELS_DIR}/{req.item}_lgbm.pkl")
        use_lgb     = True
    except FileNotFoundError:
        lgb_models = {}
        use_lgb     = False

    # 4) Explicitly build the next `periods` Mondays
    last_date = hist["ds"].max()
    future_dates = pd.DataFrame({
        "ds": [ last_date + timedelta(weeks=i+1) for i in range(req.periods) ]
    })

    # Prophet predictions on those dates
    ph = m.predict(future_dates)[["ds", "yhat"]].set_index("ds")["yhat"]

    # 5) Optional recursive LightGBM ensembling
    if use_lgb:
        feats    = [c for c in make_features(hist).columns if c not in ("ds","y")]
        hist_df  = hist.copy()
        lgb_preds = []
        for ds in ph.index:
            ff = make_features(hist_df)
            yhat = lgb_models[0.5].predict(ff.iloc[[-1]][feats])[0]
            lgb_preds.append(yhat)
            hist_df = pd.concat([
                hist_df,
                pd.DataFrame({"ds":[ds], "y":[yhat]})
            ], ignore_index=True)
        lgb_ser = pd.Series(lgb_preds, index=ph.index)
        ens     = W_PROPHET * ph + W_LGBM * lgb_ser
    else:
        ens = ph

    # 6) Serialize and return exactly `periods` points
    out = [
        {"ds": ds.strftime("%Y-%m-%d"), "yhat": round(float(v), 2)}
        for ds, v in ens.items()
    ]
    return {"item": req.item, "predictions": out}
