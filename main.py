# main.py

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, timedelta

# ── Config ────────────────────────────────────────────────
MODELS_DIR = "models"
W_PROPHET  = 0.6
W_LGBM     = 0.4

app = FastAPI()

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
def predict(req: PredictRequest):
    # 1) Build history DataFrame from request
    if req.history is None:
        raise HTTPException(400, "You must send `history` in the body.")
    hist = pd.DataFrame([h.dict() for h in req.history])
    hist["ds"] = pd.to_datetime(hist["ds"])

    # 2) Load Prophet model
    try:
        m = joblib.load(f"{MODELS_DIR}/{req.item}_prophet.pkl")
    except FileNotFoundError:
        raise HTTPException(500, f"Prophet model missing for item `{req.item}`")

    # 3) Load LightGBM models if present
    try:
        lgb_models = joblib.load(f"{MODELS_DIR}/{req.item}_lgbm.pkl")
        use_lgb     = True
    except:
        lgb_models = {}
        use_lgb     = False

    # 4) Explicitly build the next N weekly dates and forecast
    last_date    = hist["ds"].max()
    future_dates = pd.DataFrame({
        "ds": [last_date + timedelta(weeks=i + 1) for i in range(req.periods)]
    })
    # Prophet prediction on exactly those dates
    ph_preds = m.predict(future_dates)[["ds", "yhat"]].set_index("ds")["yhat"]

    # 5) Optionally ensemble with LightGBM
    if use_lgb:
        feats    = [c for c in make_features(hist).columns if c not in ("ds", "y")]
        hist_df  = hist.copy()
        lgb_preds = []
        for ds in ph_preds.index:
            ff   = make_features(hist_df)
            yhat = lgb_models[0.5].predict(ff.iloc[[-1]][feats])[0]
            lgb_preds.append(yhat)
            hist_df = pd.concat([hist_df, pd.DataFrame({"ds":[ds], "y":[yhat]})],
                                ignore_index=True)
        lgb_ser = pd.Series(lgb_preds, index=ph_preds.index)
        ens     = W_PROPHET * ph_preds + W_LGBM * lgb_ser
    else:
        ens = ph_preds

    # 6) Serialize and return exactly `periods` forecast points
    out = [
        {"ds": ds.strftime("%Y-%m-%d"), "yhat": float(val)}
        for ds, val in ens.items()
    ]
    return {"item": req.item, "predictions": out}
