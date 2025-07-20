# main.py
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import date

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
def weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    wk = (df
          .set_index("ds")["y"]
          .resample("W-MON").median()
          .interpolate("linear"))
    return wk.to_frame("y").rename_axis("ds").reset_index()

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lag1"]  = out["y"].shift(1)
    out["lag2"]  = out["y"].shift(2)
    out["lag4"]  = out["y"].shift(4)
    out["month"] = out["ds"].dt.month
    wk = out["ds"].dt.isocalendar().week.astype(int)
    out["sin52"] = np.sin(2*np.pi*wk/52)
    out["cos52"] = np.cos(2*np.pi*wk/52)
    return out.dropna()

# ── Endpoint ─────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictRequest):
    # 1. Build weekly history
    if req.history is not None:
        # use the passed-in history
        hist = pd.DataFrame([h.dict() for h in req.history])
        hist["ds"] = pd.to_datetime(hist["ds"])
    else:
        # fallback to reading CSV
        df = pd.read_csv("all_data.csv", parse_dates=["report_date"])
        df = df.query("item == @req.item") \
               .rename(columns={"report_date":"ds", "narahenpita_retail_today":"y"})
        hist = weekly_series(df)

    if hist.empty:
        raise HTTPException(404, f"No history for item {req.item}")

    # 2. Load models
    try:
        m = joblib.load(f"{MODELS_DIR}/{req.item}_prophet.pkl")
    except FileNotFoundError:
        raise HTTPException(500, "Prophet model missing")

    try:
        lgb_models = joblib.load(f"{MODELS_DIR}/{req.item}_lgbm.pkl")
        use_lgb = True
    except:
        lgb_models = {}
        use_lgb = False

    # 3. Prophet forecast (next req.periods weeks)
    future = m.make_future_dataframe(periods=req.periods, freq="W-MON")
    future = future[future["ds"] > hist["ds"].max()]
    ph     = m.predict(future).set_index("ds")["yhat"]

    # 4. Optional LightGBM recursive
    if use_lgb:
        feats   = [c for c in make_features(hist).columns if c not in ("ds","y")]
        hist_df = hist.copy()
        lgb_preds = []
        for ds in ph.index:
            feat_full = make_features(hist_df)
            row       = feat_full.iloc[[-1]][feats]
            yhat      = lgb_models[0.5].predict(row)[0]
            lgb_preds.append(yhat)
            hist_df = pd.concat([hist_df, pd.DataFrame({"ds":[ds],"y":[yhat]})],
                                 ignore_index=True)
        lgb = pd.Series(lgb_preds, index=ph.index)
        ens = W_PROPHET*ph + W_LGBM*lgb
    else:
        ens = ph

    # 5. Return JSON
    out = [{"ds": ds.strftime("%Y-%m-%d"), "yhat": float(val)} for ds,val in ens.items()]
    return {"item": req.item, "predictions": out}
