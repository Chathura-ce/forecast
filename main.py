 # …
 @app.post("/predict")
 def predict(req: PredictRequest):
     # 1) build hist from req.history …
     # 2) load Prophet model …
     # 3) load LightGBM …

-    # 4. build a full future+history frame, predict, and SLICE
-    full = m.make_future_dataframe(periods=req.periods, freq="W-MON")
-    preds = m.predict(full)[["ds","yhat"]]
-    tail = preds.tail(req.periods).set_index("ds")["yhat"]
+    # 4. EXPLICITLY build the next N weekly Mondays
+    from datetime import timedelta
+    last_date = hist["ds"].max()
+    future_dates = pd.DataFrame({
+        "ds": [
+            last_date + timedelta(weeks=i+1)
+            for i in range(req.periods)
+        ]
+    })
+    # Prophet predict on exactly those dates
+    ph = m.predict(future_dates)[["ds","yhat"]].set_index("ds")["yhat"]

     # 5. optionally ensemble with LightGBM …
-    if use_lgb:
-        feats   = [c for c in make_features(hist).columns if c not in ("ds","y")]
-        hist_df = hist.copy()
-        lgb_preds = []
-        for ds in tail.index:
-            ff = make_features(hist_df)
-            yhat = lgb_models[0.5].predict(ff.iloc[[-1]][feats])[0]
-            lgb_preds.append(yhat)
-            hist_df = pd.concat([hist_df, pd.DataFrame({"ds":[ds],"y":[yhat]})],
-                                ignore_index=True)
-        lgb_ser = pd.Series(lgb_preds, index=tail.index)
-        ens     = W_PROPHET*tail + W_LGBM*lgb_ser
-    else:
-        ens = tail
+    if use_lgb:
+        feats   = [c for c in make_features(hist).columns if c not in ("ds","y")]
+        hist_df = hist.copy()
+        lgb_preds = []
+        for ds in ph.index:
+            ff   = make_features(hist_df)
+            yhat = lgb_models[0.5].predict(ff.iloc[[-1]][feats])[0]
+            lgb_preds.append(yhat)
+            hist_df = pd.concat([
+                hist_df,
+                pd.DataFrame({"ds":[ds],"y":[yhat]})
+            ], ignore_index=True)
+        lgb_ser = pd.Series(lgb_preds, index=ph.index)
+        ens     = W_PROPHET*ph + W_LGBM*lgb_ser
+    else:
+        ens = ph

     # 6. serialize and return exactly `periods` points
     out = [{"ds": ds.strftime("%Y-%m-%d"), "yhat": float(v)} for ds,v in ens.items()]
     return {"item": req.item, "predictions": out}
