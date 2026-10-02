"""โหลดและทำความสะอาดข้อมูล (เทียบกับ gold_service.py)"""
import pandas as pd
import streamlit as st

CSV_PATH = "data/chaophraya_level2569.csv"

@st.cache_data
def load_data(path: str = CSV_PATH):
    """คืนค่า (df ที่สะอาดแล้ว, dict สรุปการทำความสะอาด)"""
    raw = pd.read_csv(path)
    n_raw = len(raw)
    # รูปแบบเวลาปนกัน (มี 'T' คั่น) -> ทำให้เหมือนกัน
    raw["datetimes"] = pd.to_datetime(raw["datetimes"].str.replace("T", " "))
    df = (raw.groupby(["stn_id", "stn_name", "datetimes"], as_index=False)["water_level"]
             .mean()  # รวมแถวซ้ำด้วยค่าเฉลี่ย
             .sort_values(["stn_id", "datetimes"]).reset_index(drop=True))
    info = {"raw_rows": n_raw, "clean_rows": len(df), "removed": n_raw - len(df)}
    return df, info

def station_options(df: pd.DataFrame) -> dict:
    return dict(df[["stn_id", "stn_name"]].drop_duplicates().values)

def coverage(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["stn_id", "stn_name"])["datetimes"].agg(["min", "max", "count"]).reset_index()
    g.columns = ["รหัส", "สถานี", "เริ่ม", "สิ้นสุด", "จำนวนแถว"]
    return g
