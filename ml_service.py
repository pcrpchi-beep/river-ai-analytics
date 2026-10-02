"""แบบจำลองทำนายระดับน้ำล่วงหน้า (ไฟล์ใหม่)"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LAGS = [1, 2, 3, 6, 18, 36, 72, 144]   # x10 นาที = 10,20,30,60 นาที, 3,6,12,24 ชม.
HORIZON = 6                             # ทำนายล่วงหน้า 6 ก้าว = 1 ชั่วโมง

def make_features(s: pd.DataFrame) -> pd.DataFrame:
    x = s.set_index("datetimes")["water_level"].resample("10min").mean()
    f = pd.DataFrame({"y_now": x})
    for k in LAGS:
        f[f"lag_{k * 10}min"] = x.shift(k)
    hr = x.index.hour + x.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    f["target"] = x.shift(-HORIZON)
    return f.dropna()

@st.cache_data
def train_and_evaluate(s: pd.DataFrame):
    f = make_features(s)
    if len(f) < 500:
        return None
    cut = int(len(f) * 0.8)                 # แบ่งตามเวลา ไม่สุ่ม
    tr, te = f.iloc[:cut], f.iloc[cut:]
    X, y = f.drop(columns="target").columns, "target"
    models = {"Baseline (ค่าล่าสุด)": None,
              "Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=14,
                                                     n_jobs=-1, random_state=42)}
    rows, preds, imp = [], {}, None
    for name, m in models.items():
        if m is None:
            p = te["y_now"].to_numpy()
        else:
            # ทำนาย "ส่วนต่าง" จากค่าปัจจุบัน แล้วบวกกลับ (ช่วยให้โมเดลไม่ติดช่วงค่าที่เคยเห็น)
            m.fit(tr[X], tr[y] - tr["y_now"]); p = m.predict(te[X]) + te["y_now"].to_numpy()
        preds[name] = p
        rows.append({"โมเดล": name, "MAE": mean_absolute_error(te[y], p),
                     "RMSE": mean_squared_error(te[y], p) ** 0.5, "R²": r2_score(te[y], p)})
        if name == "Random Forest":
            imp = pd.Series(m.feature_importances_, index=X).sort_values()
    result = te[[y]].copy()
    for k, v in preds.items():
        result[k] = v
    return {"metrics": pd.DataFrame(rows).round(4), "test": result, "importance": imp,
            "n_train": len(tr), "n_test": len(te)}
