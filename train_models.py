# forecast-api/train_models.py

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import lightgbm as lgb

from model_wrapper import ForecastWrapper, make_features

# 1) Configuration
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Map each item → model type
ITEM_MODEL = {
    "Beans":     "prophet",
    "Nadu":      "arima",
    "Salaya":    "ensemble",
    "Egg":       "lgbm",
    "Kelawalla": "prophet",
    "Coconut":   "ensemble",
}

# 2) Load & weekly‐resample data
df = pd.read_csv("all_data.csv")
df = df.rename(columns={
    "report_date": "ds",
    "narahenpita_retail_today": "y"
})
df["ds"] = pd.to_datetime(df["ds"], infer_datetime_format=True)

weeks = []
for item in df["item"].unique():
    s  = df[df["item"]==item].set_index("ds")["y"].sort_index()
    wk = (s.resample("W-MON").median()
           .interpolate("linear")
           .fillna(method="bfill")
           .fillna(method="ffill"))
    tmp = wk.reset_index().rename(columns={0:"y"})
    tmp["item"] = item
    weeks.append(tmp)

weekly_df = pd.concat(weeks).sort_values(["item","ds"]).reset_index(drop=True)

# 3) Train & pickle wrappers
for item, mtype in ITEM_MODEL.items():
    print(f"Training {item} [{mtype}]...")
    data = weekly_df[weekly_df["item"]==item][["ds","y"]]

    p_model = None
    l_model = None
    a_model = None

    # Prophet
    if mtype in ("prophet", "ensemble"):
        p_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        p_model.fit(data)

    # LightGBM
    if mtype in ("lgbm", "ensemble"):
        feat = make_features(data)
        X, y = feat.drop(["ds","y"], axis=1), feat["y"]
        l_model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5,
            learning_rate=0.05, num_leaves=31,
            min_data_in_leaf=20, n_estimators=200
        )
        l_model.fit(X, y)

    # ARIMA
    if mtype == "arima":
        a_model = ARIMA(data.set_index("ds")["y"], order=(1,1,1)).fit()

    # Wrap & dump
    wrapper = ForecastWrapper(
        model_type=mtype,
        prophet=p_model,
        lgbm=l_model,
        arima=a_model,
        w_prophet=0.6
    )
    joblib.dump(wrapper, f"{MODELS_DIR}/{item}.pkl")
    print(f" → Saved {MODELS_DIR}/{item}.pkl")

print("All models trained and saved.")
