"""เชื่อมต่อ Gemini (เทียบกับ ai_service.py)"""
import os

def get_secret(name, default=""):
    v = os.getenv(name)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default

def ask_river_ai(station, stats, ml, question):
    key = get_secret("GEMINI_API_KEY")
    if not key:
        return "ยังไม่ได้ตั้งค่า GEMINI_API_KEY (ดูไฟล์ .env.example)"
    from google import genai
    best = ""
    if ml:
        m = ml["metrics"].sort_values("MAE").iloc[0]
        best = f"โมเดลทำนายล่วงหน้า 1 ชม. ที่ดีที่สุด: {m['โมเดล']} (MAE {m['MAE']} ม., R² {m['R²']})"
    prompt = f"""คุณเป็นผู้ช่วยวิเคราะห์ระดับน้ำแม่น้ำเจ้าพระยาตอนล่าง ตอบภาษาไทย กระชับ
ใช้เฉพาะตัวเลขที่ให้ ห้ามแต่งข้อมูล ระบุว่าเป็นการประมาณการเชิงสถิติ ไม่ใช่คำเตือนภัยทางการ
หมายเหตุ: ระดับน้ำอ้างอิงระดับน้ำทะเลปานกลาง ติดลบได้ และสถานีท้ายน้ำได้รับอิทธิพลน้ำขึ้นน้ำลง

สถานี: {station}
ค่าล่าสุด {stats['latest']:.2f} ม. ({stats['latest_time']}) เปลี่ยนใน 10 นาที {stats['change_10m']:+.3f} ม.
ค่าเฉลี่ย {stats['mean']:.2f} สูงสุด {stats['max']:.2f} ต่ำสุด {stats['min']:.2f} ม.
ความผันผวน (SD การเปลี่ยนแปลงต่อ 10 นาที) {stats['volatility']:.4f}
ช่วงน้ำขึ้นน้ำลงเฉลี่ยรายวัน {stats['avg_range']:.2f} ม.
เกณฑ์ P95 = {stats['p95']:.2f}, P99 = {stats['p99']:.2f} ม. สถานะ: {stats['status']}
{best}

คำถาม: {question}"""
    try:
        c = genai.Client(api_key=key)
        return c.models.generate_content(
            model=get_secret("GEMINI_MODEL", "gemini-3.6-flash"), contents=prompt).text
    except Exception as e:
        return f"เรียก Gemini ไม่สำเร็จ: {e}"
