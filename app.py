import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from data_service import load_data, station_options, coverage
from stats_service import calculate_statistics
from ml_service import train_and_evaluate
from ai_service import ask_river_ai

st.set_page_config(page_title="Chao Phraya AI Analytics", page_icon="🌊", layout="wide")
st.title("🌊 Chao Phraya River AI Analytics")

df, info = load_data()
stations = station_options(df)

st.sidebar.header("ตัวเลือก")
sid = st.sidebar.selectbox("สถานี", list(stations), format_func=lambda k: f"{k} · {stations[k]}")
s_all = df[df["stn_id"] == sid]
d0, d1 = s_all["datetimes"].min().date(), s_all["datetimes"].max().date()
rng = st.sidebar.date_input("ช่วงวันที่", (d0, d1), min_value=d0, max_value=d1)
s = s_all
if len(rng) == 2:
    s = s_all[(s_all["datetimes"].dt.date >= rng[0]) & (s_all["datetimes"].dt.date <= rng[1])]
if len(s) < 10:
    st.warning("ข้อมูลในช่วงที่เลือกน้อยเกินไป"); st.stop()

stats = calculate_statistics(s)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Compare", "Analytics", "Forecast", "AI Insights"])

with tab1:
    c = st.columns(5)
    c[0].metric("ระดับล่าสุด (ม.)", f"{stats['latest']:.2f}", f"{stats['change_10m']:+.3f}")
    c[1].metric("สถานะ", stats["status"])
    c[2].metric("ช่วงขึ้น-ลงเฉลี่ย/วัน", f"{stats['avg_range']:.2f}")
    c[3].metric("Volatility", f"{stats['volatility']:.4f}")
    c[4].metric("สูงสุด / ต่ำสุด", f"{stats['max']:.2f} / {stats['min']:.2f}")
    st.caption(f"ข้อมูลล่าสุดของสถานีนี้: {stats['latest_time']} | เกณฑ์ P95={stats['p95']:.2f}, P99={stats['p99']:.2f} (คำนวณจากข้อมูลเอง ไม่ใช่เกณฑ์ทางการ)")
    fig = go.Figure(go.Scattergl(x=s["datetimes"], y=s["water_level"], mode="lines", name="ระดับน้ำ"))
    fig.add_hline(y=stats["p95"], line_dash="dash", line_color="orange", annotation_text="P95")
    fig.add_hline(y=stats["p99"], line_dash="dash", line_color="red", annotation_text="P99")
    fig.update_layout(height=420, yaxis_title="ม.รทก.")
    st.plotly_chart(fig, use_container_width=True)
    
with tab2:
    picks = st.multiselect("เลือกสถานีเทียบกัน", list(stations), default=list(stations)[:3],
                           format_func=lambda k: f"{k} · {stations[k]}")
    mode = st.radio("ข้อมูล", ["รายวัน (ค่าเฉลี่ย)", "ละเอียด 10 นาที"], horizontal=True)
    fig = go.Figure()
    for k in picks:
        t = df[df["stn_id"] == k].set_index("datetimes")["water_level"]
        if mode.startswith("รายวัน"):
            t = t.resample("D").mean()
        fig.add_trace(go.Scattergl(x=t.index, y=t.values, mode="lines", name=f"{k} {stations[k]}"))
    fig.update_layout(height=450, yaxis_title="ม.รทก.")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    d = stats["daily"]
    st.dataframe(d.round(3), use_container_width=True)
    st.download_button("ดาวน์โหลด CSV", d.to_csv(index=False).encode("utf-8-sig"), f"{sid}_daily.csv", "text/csv")
    fig = go.Figure()
    for col, nm in [("mean", "ค่าเฉลี่ยรายวัน"), ("ma7", "MA7"), ("ma30", "MA30")]:
        fig.add_scatter(x=d["date"], y=d[col], name=nm)
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(go.Figure(go.Bar(x=d["date"], y=d["range"])).update_layout(
        height=300, title="ช่วงน้ำขึ้น-น้ำลงรายวัน (ม.)"), use_container_width=True)

with tab4:
    st.write("ทำนายระดับน้ำล่วงหน้า **1 ชั่วโมง** ใช้ข้อมูลทั้งสถานี แบ่งตามเวลา 80/20")
    ml = train_and_evaluate(s_all)
    if ml is None:
        st.info("ข้อมูลสถานีนี้น้อยเกินไปสำหรับสร้างแบบจำลอง (เช่น S4)")
    else:
        st.caption(f"Train {ml['n_train']:,} แถว | Test {ml['n_test']:,} แถว")
        st.dataframe(ml["metrics"], use_container_width=True)
        t = ml["test"].tail(300)
        fig = go.Figure()
        fig.add_scatter(x=t.index, y=t["target"], name="ค่าจริง")
        fig.add_scatter(x=t.index, y=t["Random Forest"], name="Random Forest")
        fig.add_scatter(x=t.index, y=t["Baseline (ค่าล่าสุด)"], name="Baseline", line=dict(dash="dot"))
        fig.update_layout(height=380, title="ค่าจริง vs ค่าทำนาย (300 จุดสุดท้ายของชุดทดสอบ)")
        st.plotly_chart(fig, use_container_width=True)
        imp = ml["importance"]
        st.plotly_chart(go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h")).update_layout(
            height=350, title="ความสำคัญของฟีเจอร์ (Random Forest)"), use_container_width=True)

with tab5:
    ml = train_and_evaluate(s_all)
    presets = {"Trend": "สรุปแนวโน้มระดับน้ำ", "Volatility": "ระดับน้ำผันผวนแค่ไหน",
               "Full summary": "สรุปสถานการณ์ทั้งหมดและข้อควรระวัง"}
    cols = st.columns(len(presets))
    for col, (n, t) in zip(cols, presets.items()):
        if col.button(n):
            st.session_state["q"] = t
    q = st.text_input("คำถาม", key="q")
    if st.button("Analyze river") and q:
        with st.spinner("Gemini กำลังวิเคราะห์..."):
            st.markdown(ask_river_ai(f"{sid} {stations[sid]}", stats, ml, q))
