# model_wrapper.py
import numpy as np
import pandas as pd
import tensorflow as tf

WIN_STATS     = 730      # window for mean/std calculation
H_DEFAULT     = 30
SD_FLOOR_ABS  = 5.0
SD_FLOOR_REL  = 0.015
CLIP_NORM     = 10.0

def sd_eff(mu: float, sd: float) -> float:
    return max(float(sd), SD_FLOOR_ABS, abs(float(mu)) * SD_FLOOR_REL)

class ModelBundle:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        # infer required input steps (timesteps) and output length
        in_shape  = self.model.input_shape   # (None, steps_in, 1)
        out_shape = self.model.output_shape  # (None, steps_out) or (None, steps_out, 1)
        self.steps_in  = int(in_shape[1])
        self.steps_out = int(out_shape[-1])

    def predict_next(
        self,
        daily_series: pd.Series,           # pd.DatetimeIndex (daily), float values
        horizon: int = H_DEFAULT
    ) -> pd.Series:

        if len(daily_series) < max(WIN_STATS, self.steps_in):
            raise ValueError(
                f"Need at least {max(WIN_STATS, self.steps_in)} daily points, got {len(daily_series)}"
            )

        # --- stats on a long window for stable μ/σ ---
        hist_stats = daily_series.iloc[-WIN_STATS:]
        mu  = float(hist_stats.mean())
        sd  = float(hist_stats.std())
        sde = sd_eff(mu, sd)

        # --- model input = last `steps_in` points, normalized the same way ---
        x_seq = daily_series.iloc[-self.steps_in:]
        x = ((x_seq.values - mu) / sde).reshape(1, self.steps_in, 1).astype("float32")

        y_norm = self.model.predict(x, verbose=0).reshape(-1)
        y_norm = np.clip(y_norm, -CLIP_NORM, CLIP_NORM)

        # some models emit more than we asked; cap to horizon
        steps = int(max(1, min(horizon, len(y_norm), self.steps_out)))
        y_hat = (y_norm[:steps] * sde + mu).astype("float64")

        start = daily_series.index[-1] + pd.Timedelta(days=1)
        idx   = pd.date_range(start, periods=steps, freq="D")
        return pd.Series(y_hat, index=idx, name="yhat")

def weekly_average(daily_fc: pd.Series) -> pd.Series:
    df = daily_fc.to_frame("yhat").copy()
    df["week"] = df.index - pd.to_timedelta(df.index.dayofweek, unit="D")
    return df.groupby("week")["yhat"].mean()
