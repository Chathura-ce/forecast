# forecast-api/model_wrapper.py

import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with ['ds','y'], returns
    a DataFrame with columns: ds, y, lag1, lag2, lag4, month, sin52, cos52
    """
    out = df.copy()
    out["lag1"]  = out["y"].shift(1)
    out["lag2"]  = out["y"].shift(2)
    out["lag4"]  = out["y"].shift(4)
    out["month"] = out["ds"].dt.month

    wk = out["ds"].dt.isocalendar().week.astype(int)
    out["sin52"] = np.sin(2 * np.pi * wk / 52)
    out["cos52"] = np.cos(2 * np.pi * wk / 52)

    return out.dropna().reset_index(drop=True)


class ForecastWrapper:
    """
    Wraps one of: 'prophet', 'lgbm', 'arima', 'ensemble'
    """
    def __init__(self, model_type, prophet=None, lgbm=None, arima=None, w_prophet=0.6):
        self.type      = model_type
        self.prophet   = prophet
        self.lgbm      = lgbm
        self.arima     = arima
        self.w_prophet = w_prophet
        self.w_lgbm    = 1 - w_prophet

    def predict(self, hist_df: pd.DataFrame, periods: int) -> pd.Series:
        # 1) Build future dates (next N Mondays)
        last   = hist_df["ds"].max()
        future = pd.DataFrame({
            "ds": [ last + pd.Timedelta(weeks=i+1) for i in range(periods) ]
        }).set_index("ds")

        # 2) ARIMA-only
        if self.type == "arima":
            preds = self.arima.forecast(periods)
            return pd.Series(preds, index=future.index)

        # 3) LGBM-only (use make_features to build correct feature set)
        if self.type == "lgbm":
            hist_tmp = hist_df.copy()
            lgb_preds = []
            for ds in future.index:
                feat_df = make_features(hist_tmp)
                X_row   = feat_df.drop(["ds","y"], axis=1).iloc[[-1]]
                yhat_l  = self.lgbm.predict(X_row)[0]
                lgb_preds.append(yhat_l)
                # append prediction back into history
                hist_tmp = pd.concat([
                    hist_tmp,
                    pd.DataFrame({"ds":[ds], "y":[yhat_l]})
                ], ignore_index=True)
            return pd.Series(lgb_preds, index=future.index)

        # 4) Prophet (for 'prophet' & 'ensemble')
        ph = (
            self.prophet
                .predict(future.reset_index())
                .set_index("ds")["yhat"]
        )
        if self.type == "prophet":
            return ph

        # 5) Ensemble: recursive LGBM + blend
        hist_tmp = hist_df.copy()
        lgb_preds = []
        for ds in ph.index:
            feat_df = make_features(hist_tmp)
            X_row   = feat_df.drop(["ds","y"], axis=1).iloc[[-1]]
            yhat_l  = self.lgbm.predict(X_row)[0]
            lgb_preds.append(yhat_l)
            hist_tmp = pd.concat([
                hist_tmp,
                pd.DataFrame({"ds":[ds], "y":[yhat_l]})
            ], ignore_index=True)
        lg = pd.Series(lgb_preds, index=ph.index)

        # blend Prophet + LGBM
        return self.w_prophet * ph + self.w_lgbm * lg
