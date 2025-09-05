# main.py
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Literal
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging

from model_wrapper import ModelBundle, weekly_average

# ---- load model once on startup ----
# adjust the path if needed (relative to where you run uvicorn)
BUNDLE = ModelBundle("models/lstm_730in30out.h5")

app = FastAPI(title="Food Price Forecast API", version="1.0")
logger = logging.getLogger("uvicorn.error")

# ---------- schemas ----------
class HistRow(BaseModel):
    ds: str     # 'YYYY-MM-DD'
    y: float

class PredictReq(BaseModel):
    item: Optional[str] = None
    history: List[HistRow]
    horizon: int = 30
    aggregate: Literal["daily","weekly"] = "weekly"

# ---------- helpers ----------
def build_daily_series(history_rows: List[HistRow]) -> pd.Series:
    df = pd.DataFrame([r.model_dump() for r in history_rows])
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"]  = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds","y"]).sort_values("ds")

    s = df.set_index("ds")["y"]

    # 1) Treat zeros/negatives as missing
    s = s.mask(s <= 0, other=np.nan)

    # 2) Optional: clip extreme outliers (robust winsorize)
    q1, q99 = s.quantile([0.01, 0.99])
    s = s.clip(lower=q1, upper=q99)

    # 3) Reindex to daily & fill only short gaps (<=7d)
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = (s.reindex(idx)
           .interpolate("time", limit=7)
           .ffill(limit=7).bfill(limit=7))   # avoid month-long plateaus
    return s


def required_points() -> int:
    # If your ModelBundle exposes steps_in, use it; otherwise default to 730
    return getattr(BUNDLE, "steps_in", 730)

# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"ok": True, "need_points": required_points(), "default_horizon": 30}

@app.post("/predict")
def predict(req: PredictReq):
    try:
        s = build_daily_series(req.history)

        need = required_points()
        if len(s) < need:
            raise ValueError(f"need at least {need} daily points; got {len(s)}")

        # keep last 'need' points for the model
        s = s.iloc[-need:]

        H = int(max(1, min(req.horizon, 60)))
        fc_daily = BUNDLE.predict_next(s, horizon=H)

        # guard against NaN/Inf (will crash JSON otherwise)
        if not np.isfinite(fc_daily.values).all():
            raise ValueError("model returned non-finite values (NaN/Inf).")

        if req.aggregate == "weekly":
            fc_week = weekly_average(fc_daily)
            out = [{"ds": d.strftime("%Y-%m-%d"), "yhat": float(v)} for d, v in fc_week.items()]
        else:
            out = [{"ds": d.strftime("%Y-%m-%d"), "yhat": float(v)} for d, v in fc_daily.items()]

        return {"predictions": out}

    except Exception as e:
        # Log full traceback server-side; return readable message to Laravel
        logger.exception("predict failed")
        raise HTTPException(status_code=400, detail=str(e))
