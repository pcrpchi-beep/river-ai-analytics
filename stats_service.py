"""วิเคราะห์ทางสถิติ (เทียบกับ stats_service.py)"""
import pandas as pd

def daily_table(s: pd.DataFrame) -> pd.DataFrame:
    d = (s.set_index("datetimes")["water_level"].resample("D")
           .agg(["mean", "max", "min"]).dropna())
    d["range"] = d["max"] - d["min"]          # ช่วงน้ำขึ้นน้ำลงรายวัน
    d["ma7"] = d["mean"].rolling(7).mean()
    d["ma30"] = d["mean"].rolling(30).mean()
    return d.reset_index().rename(columns={"datetimes": "date"})

def calculate_statistics(s: pd.DataFrame) -> dict:
    lv = s["water_level"]
    daily = daily_table(s)
    p95, p99 = lv.quantile(0.95), lv.quantile(0.99)
    latest = float(lv.iloc[-1])
    status = "เฝ้าระวัง" if latest >= p99 else "สูงผิดปกติ" if latest >= p95 else "ปกติ"
    return {
        "daily": daily, "latest": latest, "latest_time": s["datetimes"].iloc[-1],
        "change_10m": float(lv.diff().iloc[-1]), "mean": float(lv.mean()),
        "max": float(lv.max()), "min": float(lv.min()),
        "volatility": float(lv.diff().std()), "avg_range": float(daily["range"].mean()),
        "p95": float(p95), "p99": float(p99), "status": status,
    }
