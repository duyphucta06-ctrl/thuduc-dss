import datetime
import requests
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ thống Hỗ trợ Quyết định Tưới tiêu - TP. Thủ Đức",
    page_icon="🌱",
    layout="wide"
)

LAT, LON = 10.8494, 106.7537
MODEL_FILE = "rf_et0_gee.pkl"
DATA_FILE = "clean_dataset.csv"
FEATURES = ["temp_max", "temp_min", "temp_mean", "humidity_mean", "radiation_sum", "precipitation_sum", "month", "is_rainy_season"]

# --- 1. TẢI DỮ LIỆU & TRAIN MÔ HÌNH NẾU CHƯA CÓ FILE PKL ---
@st.cache_data
def load_data_and_model():
    try:
        df = pd.read_csv(DATA_FILE)
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        df = None

    try:
        model = joblib.load(MODEL_FILE)
    except Exception:
        # Nếu chưa có file .pkl, tự huấn luyện Random Forest trên clean_dataset.csv
        if df is not None:
            from sklearn.ensemble import RandomForestRegressor
            X = df[FEATURES]
            y = df["et0_gee"]
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
        else:
            model = None

    return df, model

# --- 2. CẬP NHẬT THỜI TIẾT THỜI GIAN THỰC TỪ OPEN-METEO ---
# ttl=3600: Tự động xóa cache và làm mới dữ liệu thời tiết sau mỗi 1 giờ
@st.cache_data(ttl=3600)
def fetch_live_weather_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "precipitation_probability_max",
            "shortwave_radiation_sum",
        ],
        "timezone": "Asia/Bangkok",
        "forecast_days": 1,
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            d = res.json()["daily"]
            return {
                "temp_max": d["temperature_2m_max"][0],
                "temp_min": d["temperature_2m_min"][0],
                "temp_mean": d["temperature_2m_mean"][0],
                "humidity_mean": d["relative_humidity_2m_mean"][0],
                "precipitation_sum": d["precipitation_sum"][0],
                "pop_max": d["precipitation_probability_max"][0], # % mưa dự báo
                "radiation_sum": d["shortwave_radiation_sum"][0],
            }
    except Exception as e:
        st.error(f"Lỗi kết nối API Open-Meteo: {e}")
    return None

# --- GIAO DIỆN CHÍNH ---
st.title("🌱 Dashboard Hỗ trợ Quyết định Tưới tiêu - TP. Thủ Đức")
st.caption("Dữ liệu tự động cập nhật từ Open-Meteo API & Mô hình ML dự báo ET₀")

df_hist, model = load_data_and_model()
weather_live = fetch_live_weather_forecast()

# KHI MỞ WEB: Hiển thị các thông số thời tiết hôm nay
col1, col2, col3, col4 = st.columns(4)
if weather_live:
    col1.metric("🌧️ % Mưa dự báo", f"{weather_live['pop_max']} %")
    col2.metric("💧 Lượng mưa dự báo", f"{weather_live['precipitation_sum']} mm")
    col3.metric("🌡️ Nhiệt độ trung bình", f"{weather_live['temp_mean']} °C")
    col4.metric("☀️ Bức xạ mặt trời", f"{weather_live['radiation_sum']} MJ/m²")

st.markdown("---")

col_left, col_right = st.columns([1, 1.2])

# --- CỘT TRÁI: DỰ BÁO TƯỚI HÔM NAY ---
with col_left:
    st.subheader("🤖 Dự báo ET₀ ML & Nhu cầu tưới")
    kc = st.slider("Hệ số cây trồng (Kc):", 0.4, 1.3, 1.0, 0.05)
    
    if weather_live and model:
        now = datetime.datetime.now()
        input_data = pd.DataFrame([{
            "temp_max": weather_live["temp_max"],
            "temp_min": weather_live["temp_min"],
            "temp_mean": weather_live["temp_mean"],
            "humidity_mean": weather_live["humidity_mean"],
            "radiation_sum": weather_live["radiation_sum"],
            "precipitation_sum": weather_live["precipitation_sum"],
            "month": now.month,
            "is_rainy_season": 1 if 5 <= now.month <= 11 else 0
        }])
        
        et0_pred = model.predict(input_data)[0]
        etc = et0_pred * kc
        rain_eff = weather_live["precipitation_sum"] * 0.8  # 80% mưa hiệu quả
        i_rec = max(0.0, etc - rain_eff)
        
        st.success(f"**ET₀ Dự báo ML:** {et0_pred:.2f} mm/ngày")
        st.info(f"**Nhu cầu nước cây trồng (ETc):** {etc:.2f} mm/ngày")
        
        if i_rec > 0:
            st.warning(f"💧 **Nước cần tưới bù:** {i_rec:.2f} mm/ngày (Tương đương {i_rec * 10:.1f} m³/ha)")
        else:
            st.success("🌧️ **Không cần tưới:** Lượng mưa hôm nay đã đủ đáp ứng nhu cầu nước!")

# --- CỘT PHẢI: BIỂU ĐỒ SO SÁNH THEO ĐỘ CHIA VÀ CÁC NĂM ---
with col_right:
    st.subheader("📈 So sánh ET₀ GEE vs ET₀ ML")
    
    if df_hist is not None and model is not None:
        scale_option = st.radio(
            "Độ chia thời gian:",
            ["Tối đa (Theo Toàn bộ các năm)", "1 Năm (12 Tháng)", "10 Ngày gần nhất"],
            horizontal=True
        )
        
        df_plot = df_hist.copy()
        df_plot["et0_ml"] = model.predict(df_plot[FEATURES])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        if scale_option == "10 Ngày gần nhất":
            # Lấy 10 dòng cuối cùng trong dataset
            df_sub = df_plot.tail(10)
            x_dates = df_sub["date"].dt.strftime('%m-%d')
            ax.plot(x_dates, df_sub["et0_gee"], marker='o', label="ET0 GEE", color="green")
            ax.plot(x_dates, df_sub["et0_ml"], marker='x', linestyle="--", label="ET0 ML", color="orange")
            ax.set_xlabel("Ngày (Tháng-Ngày)")
            
        elif scale_option == "1 Năm (12 Tháng)":
            # ĐỘNG: Lấy danh sách các năm có trong dữ liệu để người dùng chọn
            available_years = sorted(df_plot["date"].dt.year.unique(), reverse=True)
            selected_year = st.selectbox("Chọn năm hiển thị:", available_years)
            
            # Lọc theo năm được chọn và tính trung bình theo tháng
            df_year = df_plot[df_plot["date"].dt.year == selected_year]
            df_sub = df_year.groupby(df_year["date"].dt.month)[["et0_gee", "et0_ml"]].mean().reset_index()
            
            ax.plot(df_sub["date"], df_sub["et0_gee"], marker='o', label="ET0 GEE", color="green")
            ax.plot(df_sub["date"], df_sub["et0_ml"], marker='x', linestyle="--", label="ET0 ML", color="orange")
            ax.set_xlabel(f"Tháng (Năm {selected_year})")
            ax.set_xticks(range(1, 13))
            
        else:
            # Tối đa: Nhóm theo toàn bộ các năm có trong bộ dữ liệu
            df_sub = df_plot.groupby(df_plot["date"].dt.year)[["et0_gee", "et0_ml"]].mean().reset_index()
            ax.plot(df_sub["date"], df_sub["et0_gee"], marker='o', label="ET0 GEE", color="green")
            ax.plot(df_sub["date"], df_sub["et0_ml"], marker='x', linestyle="--", label="ET0 ML", color="orange")
            ax.set_xlabel("Năm")
            ax.set_xticks(df_sub["date"])

        ax.set_ylabel("ET0 (mm/ngày)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
